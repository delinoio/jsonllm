"""Dependency-safe field waves with a scalar-only output contract."""

import math
import time

from jsonschema.exceptions import ValidationError

from .metrics import grade, strict_json
from .prompts import PROMPT_VERSION, SHARED_VERSION, VALUE_VERSION, field_example, prompt_ids
from .schema import ancestors, dependency_order, validate_questions, validate_value


def decode_result(example, raw):
    try:
        if example["choices"]:
            value = example["choices"][raw.strip()]
        else:
            decoded = strict_json(example["prefill"] + raw)
            if example["prompt_version"] in (VALUE_VERSION, SHARED_VERSION):
                value = decoded
            else:
                if not isinstance(decoded, dict) or set(decoded) != {example["field"]}:
                    raise ValueError("Wrong fields")
                value = decoded[example["field"]]
        validate_value(example["schema"], value)
        return {"valid": True, "value": value}
    except (ValueError, KeyError, TypeError, ValidationError):
        # Schema failures are counted rather than repaired or retried.
        return {"valid": False, "value": None, "error": "invalid_output"}


def run_typed(
    record,
    predictor,
    *,
    prompt_version=PROMPT_VERSION,
    batch_fields=True,
    max_length=2048,
    max_tokens=256,
    evaluate=True,
):
    """Infer a record; set evaluate=False when reference answers are unavailable."""
    if not evaluate:
        for name in ("id", "context"):
            if not isinstance(record.get(name), str) or not record[name].strip():
                raise ValueError(f"Record requires non-empty {name}")
        validate_questions(record.get("questions"))
        record = {key: value for key, value in record.items() if key != "answers"}
    predictor.synchronize()
    started = time.perf_counter()
    if hasattr(predictor, "begin_record"):
        predictor.begin_record()
    pending = dependency_order(record["questions"])
    answers, failures, calls, raws = {}, {}, [], {}
    timing = {"format_tokenize": 0.0, "inference_waves": 0.0, "validate": 0.0}
    while pending:
        blocked = [
            name
            for name in pending
            if any(p in failures for p in ancestors(record["questions"], name))
        ]
        for name in blocked:
            failures[name] = "failed_dependency"
            pending.remove(name)
        ready = [
            name
            for name in pending
            if all(p in answers for p in record["questions"][name].get("depends_on", []))
        ]
        if not batch_fields:
            ready = ready[:1]
        examples, inputs, names = [], [], []
        phase = time.perf_counter()
        for name in ready:
            example = field_example(
                record, name, dependency_answers=answers, prompt_version=prompt_version
            )
            ids = prompt_ids(example, predictor.tokenizer)
            if len(ids) + (1 if example["choices"] else max_tokens) > max_length:
                failures[name] = "overlength"
            else:
                names.append(name)
                examples.append(example)
                inputs.append(ids)
            pending.remove(name)
        timing["format_tokenize"] += time.perf_counter() - phase
        if examples:
            phase = time.perf_counter()
            results = predictor.infer_many(examples, inputs, max_tokens)
            timing["inference_waves"] += time.perf_counter() - phase
            phase = time.perf_counter()
            if len(results) != len(examples):
                raise ValueError("Incomplete batch response")
            for name, example, result in zip(names, examples, results, strict=True):
                calls.append(result["metrics"])
                raws[name] = result["text"]
                response = decode_result(example, result["text"])
                if result["metrics"].get("finish_reason") == "length" and not example["choices"]:
                    response = {"valid": False, "error": "truncated_output"}
                if response["valid"]:
                    answers[name] = response["value"]
                else:
                    failures[name] = response["error"]
            timing["validate"] += time.perf_counter() - phase
        if pending and not ready and not blocked:
            raise ValueError("Unresolvable dependency graph")
    predictor.synchronize()
    phase = time.perf_counter()
    if evaluate:
        graded = grade(record, answers, parsed=not failures, errors=failures)
    else:
        graded = {
            "schema_valid": not failures,
            "fields": [
                {
                    "field": name,
                    "value": answers.get(name),
                    "present": name in answers,
                    "valid": name in answers,
                    "error": failures.get(name),
                }
                for name in record["questions"]
            ],
        }
    timing["validate"] += time.perf_counter() - phase
    elapsed = time.perf_counter() - started
    timing["schedule_other"] = max(0, elapsed - sum(timing.values()))
    return {
        "record_id": record["id"],
        "mode": "typed",
        "context": record["context"],
        "questions": record["questions"],
        "context_chars": len(record["context"]),
        "source": record.get("metadata", {}).get("source", "synthetic"),
        "recipe": record.get("metadata", {}).get("recipe", "unspecified"),
        "domain": record.get("metadata", {}).get("domain", "legacy"),
        "has_dependencies": any(f.get("depends_on") for f in record["questions"].values()),
        "errors": failures,
        "raw": raws,
        **graded,
        "latency_seconds": elapsed,
        "timing_seconds": timing,
        "calls": calls,
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "selected_tokens": sum(c["selected_tokens"] for c in calls),
    }


class TransformersAdapter:
    """One-model sequential reference; concurrent CUDA forwards are deliberately disallowed."""

    def __init__(self, predictor):
        self.predictor = predictor
        self.tokenizer = predictor.tokenizer

    def synchronize(self):
        self.predictor.synchronize()

    def infer_many(self, examples, inputs, max_tokens):
        results = []
        for example, ids in zip(examples, inputs, strict=True):
            self.synchronize()
            start = time.perf_counter()
            choices = example["choices"]
            if choices:
                labels = list(choices)
                tokens = [
                    self.tokenizer.encode(label, add_special_tokens=False)[0] for label in labels
                ]
                scores = self.predictor.scores(ids, tokens)
                if len(scores) != len(labels) or not all(math.isfinite(s) for s in scores):
                    raise ValueError("Invalid candidate scores")
                weights = [math.exp(s - max(scores)) for s in scores]
                detail = {
                    "text": labels[max(range(len(scores)), key=scores.__getitem__)],
                    "output_tokens": 0,
                    "finish_reason": "selection",
                    "ttft_seconds": None,
                    "probabilities": [w / sum(weights) for w in weights],
                }
            else:
                detail = self.predictor.generate_details(ids, max_tokens)
            self.synchronize()
            elapsed = time.perf_counter() - start
            results.append(
                {
                    "text": detail["text"],
                    "metrics": {
                        "kind": "selection" if choices else "generation",
                        "input_tokens": len(ids),
                        "output_tokens": detail["output_tokens"],
                        "selected_tokens": int(bool(choices)),
                        "seconds": elapsed,
                        "generation_seconds": elapsed,
                        "ttft_seconds": detail["ttft_seconds"],
                        "finish_reason": detail["finish_reason"],
                        "probabilities": detail.get("probabilities"),
                    },
                }
            )
        return results
