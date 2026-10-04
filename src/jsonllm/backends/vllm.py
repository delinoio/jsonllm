"""Typed requests to a separately pinned local vLLM service."""

import json
import math
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx

from ..prompts import SHARED_VERSION, VALUE_VERSION
from ..schema import value_schema


class Predictor:
    def __init__(self, model, revision, *, url="http://127.0.0.1:8000", timeout=120):
        from ..training import load_tokenizer

        self.tokenizer = load_tokenizer(model, revision)
        self.client = httpx.Client(
            base_url=url,
            timeout=timeout,
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=256),
        )
        self.pool = ThreadPoolExecutor(max_workers=128)
        self.local = threading.local()

    def close(self):
        self.pool.shutdown()
        self.client.close()

    def synchronize(self):
        # A completed HTTP response includes completion of the remote GPU work.
        pass

    def begin_record(self):
        # A record's fields share a prefix namespace; other records/repeats cannot hit it.
        self.local.salt = uuid.uuid4().hex

    def infer_many(self, examples, inputs, max_tokens):
        salt = self.local.salt
        jobs = [
            self.pool.submit(self.infer, e, ids, max_tokens, salt)
            for e, ids in zip(examples, inputs, strict=True)
        ]
        return [f.result() for f in jobs]

    def infer(self, example, ids, max_tokens, salt):
        labels = list(example["choices"])
        candidate_ids = [self.tokenizer.encode(s, add_special_tokens=False)[0] for s in labels]
        payload = {
            "model": "typellm",
            "prompt": ids,
            "temperature": 0.0,
            "max_tokens": 1 if labels else max_tokens,
            "cache_salt": salt,
            "stream": not labels,
        }
        if labels:
            payload.update(
                allowed_token_ids=candidate_ids,
                logprobs=len(labels),
                logprob_token_ids=candidate_ids,
                return_tokens_as_token_ids=True,
            )
        else:
            # Merged Qwen exports can retain the base generation config's EOS while
            # SFT ends with the tokenizer's chat EOS. Honor the trained terminator.
            payload["stop_token_ids"] = [self.tokenizer.eos_token_id]
            payload["stream_options"] = {"include_usage": True}
            if example["prompt_version"] in {VALUE_VERSION, SHARED_VERSION}:
                payload["structured_outputs"] = {"json": value_schema(example["schema"])}
        started = time.perf_counter()
        first, parts, usage, finish, probabilities = None, [], {}, None, None
        if labels:
            response = self.client.post("/v1/completions", json=payload)
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            parts = [choice["text"]]
            usage, finish = body["usage"], "selection"
            logits = choice["logprobs"]["top_logprobs"][0]
            scores = [
                logits.get(f"token_id:{token}", logits.get(label))
                for token, label in zip(candidate_ids, labels, strict=True)
            ]
            if any(s is None or not math.isfinite(s) for s in scores):
                raise ValueError("vLLM omitted candidate log probabilities")
            weights = [math.exp(s - max(scores)) for s in scores]
            probabilities = [w / sum(weights) for w in weights]
        else:
            with self.client.stream("POST", "/v1/completions", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices", []):
                        if choice.get("text"):
                            if first is None:
                                first = time.perf_counter() - started
                            parts.append(choice["text"])
                        finish = choice.get("finish_reason") or finish
        elapsed = time.perf_counter() - started
        if "completion_tokens" not in usage or finish is None:
            raise ValueError("Incomplete inference response or token usage")
        return {
            "text": "".join(parts),
            "metrics": {
                "kind": "selection" if labels else "generation",
                "input_tokens": len(ids),
                "output_tokens": 0 if labels else usage["completion_tokens"],
                "selected_tokens": int(bool(labels)),
                "seconds": elapsed,
                "generation_seconds": elapsed,
                "ttft_seconds": first,
                "finish_reason": finish,
                "probabilities": probabilities,
            },
        }
