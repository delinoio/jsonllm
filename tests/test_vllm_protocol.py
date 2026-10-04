import json
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from jsonllm.backends.vllm import Predictor
from jsonllm.fast_inference import decode_result
from jsonllm.prompts import PROMPT_VERSION, VALUE_VERSION


class Tokenizer:
    eos_token_id = 248046

    def encode(self, text, **kwargs):
        return [{"A": 32, "B": 33}[text]]


def client(handler):
    predictor = Predictor.__new__(Predictor)
    predictor.tokenizer = Tokenizer()
    predictor.client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    predictor.pool = ThreadPoolExecutor(max_workers=2)
    predictor.local = threading.local()
    return predictor


def example(schema, choices=None):
    return {
        "schema": schema,
        "choices": choices or {},
        "prompt_version": VALUE_VERSION,
        "prefill": "",
    }


def test_single_step_candidates_and_record_cache_isolation():
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "text": "B",
                        "logprobs": {"top_logprobs": [{"token_id:32": -2, "token_id:33": -1}]},
                    }
                ],
                "usage": {"completion_tokens": 1},
            },
        )

    predictor = client(handler)
    ex = example({"type": "boolean"}, {"A": False, "B": True})
    try:
        predictor.begin_record()
        results = predictor.infer_many([ex, ex], [[1], [2]], 256)
        predictor.begin_record()
        predictor.infer_many([ex], [[1]], 256)
    finally:
        predictor.close()
    assert requests[0]["cache_salt"] == requests[1]["cache_salt"]
    assert requests[0]["cache_salt"] != requests[2]["cache_salt"]
    assert all(r["max_tokens"] == 1 and r["allowed_token_ids"] == [32, 33] for r in requests)
    probabilities = results[0]["metrics"]["probabilities"]
    assert sum(probabilities) == pytest.approx(1)
    assert probabilities[1] > probabilities[0]
    assert decode_result(ex, results[0]["text"])["value"] is True


@pytest.mark.parametrize("version", [PROMPT_VERSION, VALUE_VERSION])
def test_number_stream_waits_for_eos_not_first_valid_prefix(version):
    def handler(request):
        body = json.loads(request.content)
        if version == VALUE_VERSION:
            assert body["structured_outputs"]["json"]["type"] == "number"
        else:
            assert "structured_outputs" not in body
        assert body["stop_token_ids"] == [248046]
        parts = ["1", "2", ".", "5"] + (["}"] if version == PROMPT_VERSION else [])
        chunks = [{"choices": [{"text": text, "finish_reason": None}]} for text in parts]
        chunks += [
            {"choices": [{"text": "", "finish_reason": "stop"}]},
            {"choices": [], "usage": {"completion_tokens": 5}},
        ]
        return httpx.Response(
            200,
            text="\n\n".join("data: " + json.dumps(c) for c in chunks) + "\n\ndata: [DONE]\n\n",
        )

    predictor = client(handler)
    ex = example({"type": "number"})
    ex["prompt_version"] = version
    if version == PROMPT_VERSION:
        ex.update(field="number", prefill='{"number":')
    try:
        result = predictor.infer(ex, [1], 256, "record")
    finally:
        predictor.close()
    assert result["text"] == ("12.5}" if version == PROMPT_VERSION else "12.5")
    assert decode_result(ex, result["text"])["value"] == 12.5
    assert result["metrics"]["output_tokens"] == 5
