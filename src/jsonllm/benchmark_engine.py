"""Matched eager CUDA ablations and an explicitly separate vLLM baseline."""

import time
import uuid

from .benchmark_data import inference_record
from .io import dumps
from .metrics import strict_json
from .schema import dependency_order, validate_questions, validate_value, value_schema

METHODS = ("whole_json", "serial_fields", "batch_fields", "shared_fields", "vllm_json")


def object_schema(record):
    return {
        "type": "object",
        "properties": {k: value_schema(v) for k, v in record["questions"].items()},
        "required": list(record["questions"]),
        "additionalProperties": False,
    }


def whole_prompt(record, tokenizer):
    fields = {
        name: {
            "rule": field["instructions"],
            **value_schema(field),
            **({"depends_on": field["depends_on"]} if field.get("depends_on") else {}),
        }
        for name, field in record["questions"].items()
    }
    messages = [
        {"role": "user", "content": record["context"].rstrip()},
        {
            "role": "user",
            "content": "Return one JSON object with exactly these fields. "
            "Follow each rule. Resolve depends_on using your own earlier field values. "
            "No explanations.\n" + dumps(fields),
        },
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tokenizer.encode(rendered, add_special_tokens=False)


def validate_object(record, result):
    if not isinstance(result, dict) or set(result) != set(record["questions"]):
        raise ValueError("wrong_fields")
    for name, field in record["questions"].items():
        validate_value(field, result[name])


def field_inference(record, predictor, *, serial=False, max_tokens=128, deadline=None):
    """Only predicted ancestors cross the dependency boundary."""
    record = inference_record(record)
    validate_questions(record["questions"])
    pending = dependency_order(record["questions"])
    values, first = {}, None
    started = time.perf_counter()
    session = predictor.open_record(record["context"], len(pending))
    opened = time.perf_counter()
    try:
        while pending:
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("request_timeout")
            ready = [
                n
                for n in pending
                if all(p in values for p in record["questions"][n].get("depends_on", []))
            ]
            if serial:
                ready = ready[:1]
            if not ready:
                raise ValueError("unresolved_dependency")
            predicted = session.predict(record, ready, values.copy(), max_tokens)
            if set(predicted) != set(ready):
                raise ValueError("incomplete_wave")
            for name in ready:
                validate_value(record["questions"][name], predicted[name])
            values.update(predicted)
            if first is None:
                first = time.perf_counter() - started
            pending = [name for name in pending if name not in predicted]
        validate_object(record, values)
    finally:
        session.close()
    return values, {
        "first_usable_seconds": first,
        "session_open_seconds": opened - started,
        "lock_wait_seconds": session.metrics.get("lock_wait_seconds", 0.0),
        "output_tokens": sum(c["output_tokens"] for c in session.metrics.get("calls", [])),
        "model": session.metrics,
    }


class WholeJSON:
    def __init__(self, predictor):
        import xgrammar as xgr

        from .backends.scalar_grammar import ScalarGrammar

        self.predictor = predictor
        self.grammar = ScalarGrammar(predictor.tokenizer, predictor.model.lm_head.weight.shape[0])
        self.xgr = xgr

    def infer(self, record, max_tokens, deadline=None):
        import torch

        predictor = self.predictor
        ids = whole_prompt(record, predictor.tokenizer)
        if len(ids) + max_tokens > predictor.max_length:
            raise ValueError("overlength")
        compiled = self.grammar.compiler.compile_json_schema(
            object_schema(record), any_whitespace=False
        )
        matcher = self.xgr.GrammarMatcher(compiled, terminate_without_stop_token=False)
        mask = self.grammar.allocate(1, predictor.device)
        waited = time.perf_counter()
        with predictor.lock, torch.inference_mode():
            lock_wait = time.perf_counter() - waited
            tokens = torch.tensor([ids], device=predictor.device)
            cache, generated = None, []
            for _ in range(max_tokens):
                if deadline is not None and time.perf_counter() >= deadline:
                    raise TimeoutError("request_timeout")
                output = predictor.backbone(input_ids=tokens, past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                logits = predictor.model.lm_head(output.last_hidden_state[:, -1])
                self.grammar.mask(logits, [matcher], mask)
                token = int(logits.argmax(-1).item())
                if not matcher.accept_token(token):
                    raise ValueError("grammar_rejected_token")
                if token == predictor.tokenizer.eos_token_id:
                    if not matcher.is_terminated():
                        raise ValueError("premature_eos")
                    break
                generated.append(token)
                tokens = torch.tensor([[token]], device=predictor.device)
            else:
                raise ValueError("truncated_output")
            predictor.synchronize()
        text = predictor.tokenizer.decode(generated)
        result = strict_json(text)
        validate_object(record, result)
        return result, {
            "output_tokens": len(generated) + 1,
            "input_tokens": len(ids),
            "lock_wait_seconds": lock_wait,
        }


class VLLMJSON:
    def __init__(self, tokenizer, url="http://127.0.0.1:8000"):
        import httpx

        self.tokenizer = tokenizer
        self.client = httpx.Client(
            base_url=url, timeout=180, limits=httpx.Limits(max_connections=256)
        )

    def infer(self, record, max_tokens, deadline=None):
        ids = whole_prompt(record, self.tokenizer)
        payload = {
            "model": "jsonllm",
            "prompt": ids,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
            "cache_salt": uuid.uuid4().hex,
            "stop_token_ids": [self.tokenizer.eos_token_id],
            "structured_outputs": {"json": object_schema(record)},
        }
        remaining = 180 if deadline is None else max(0.001, deadline - time.perf_counter())
        response = self.client.post("/v1/completions", json=payload, timeout=remaining)
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        if choice["finish_reason"] != "stop":
            raise ValueError("truncated_output")
        result = strict_json(choice["text"])
        validate_object(record, result)
        return result, {
            "output_tokens": body["usage"]["completion_tokens"],
            "input_tokens": body["usage"]["prompt_tokens"],
            "lock_wait_seconds": None,
        }

    def close(self):
        self.client.close()
