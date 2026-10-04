"""One-field prediction used by training contract checks."""

import json
import math

from jsonschema.exceptions import ValidationError

from .prompts import VALUE_VERSION, prompt_ids
from .schema import validate_value


def predict(predictor, example, *, max_length, max_tokens):
    ids = prompt_ids(example, predictor.tokenizer)
    if len(ids) + (1 if example["choices"] else max_tokens) > max_length:
        return {"valid": False, "error": "overlength", "value": None}
    probabilities = None
    if example["choices"]:
        labels = list(example["choices"])
        token_ids = [
            predictor.tokenizer.encode(label, add_special_tokens=False)[0] for label in labels
        ]
        scores = predictor.scores(ids, token_ids)
        if len(scores) != len(labels) or not all(math.isfinite(x) for x in scores):
            raise ValueError("Non-finite or incomplete choice logits")
        weights = [math.exp(x - max(scores)) for x in scores]
        probabilities = [
            {"value": example["choices"][label], "probability": weight / sum(weights)}
            for label, weight in zip(labels, weights, strict=True)
        ]
        label = labels[max(range(len(scores)), key=scores.__getitem__)]
        raw = label
        value = example["choices"][label]
    else:
        raw = predictor.generate(ids, max_tokens)
        try:
            decoded = json.loads(example["prefill"] + raw)
            if example["prompt_version"] == VALUE_VERSION:
                value = decoded
            else:
                if not isinstance(decoded, dict) or set(decoded) != {example["field"]}:
                    raise ValueError("Response must contain exactly the requested field")
                value = decoded[example["field"]]
            validate_value(example["schema"], value)
        except (ValueError, TypeError, ValidationError):
            return {"valid": False, "error": "invalid_output", "raw": raw, "value": None}
    return {"valid": True, "value": value, "raw": raw, "probabilities": probabilities}
