"""Strict JSON parsing and deterministic scoring helpers."""

import json
import math

from jsonschema.exceptions import ValidationError

from .io import dumps
from .schema import base_type, validate_value


def correct(field, prediction, expected):
    if (
        base_type(field) in {"integer", "number"}
        and prediction is not None
        and expected is not None
    ):
        return math.isclose(prediction, expected, rel_tol=1e-6, abs_tol=1e-6)
    return type(prediction) is type(expected) and prediction == expected


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Non-finite number: {value}")

    value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    dumps(value)  # Also reject numeric overflow such as 1e999.
    return value


def grade(record, value, *, parsed, errors):
    fields = []
    obj = value if isinstance(value, dict) else {}
    expected_keys = set(record["questions"])
    for name, field in record["questions"].items():
        present = name in obj
        valid = present
        error = errors.get(name)
        if present:
            try:
                validate_value(field, obj[name])
            except (ValueError, TypeError, ValidationError):
                valid, error = False, "schema_violation"
        else:
            error = error or "missing_field"
        expected = record["answers"][name]
        prediction = obj.get(name)
        fields.append(
            {
                "field": name,
                "type": "enum" if "enum" in field else base_type(field),
                "expected": expected,
                "value": prediction,
                "present": present,
                "valid": valid,
                "correct": valid and correct(field, prediction, expected),
                "error": error,
                "absolute_error": abs(prediction - expected)
                if valid
                and base_type(field) in {"number", "integer"}
                and prediction is not None
                and expected is not None
                else None,
            }
        )
    keys_valid = parsed and isinstance(value, dict) and set(obj) == expected_keys
    schema_valid = keys_valid and all(f["valid"] for f in fields)
    return {
        "json_valid": parsed,
        "keys_valid": keys_valid,
        "schema_valid": schema_valid,
        "record_correct": schema_valid and all(f["correct"] for f in fields),
        "extra_keys": sorted(set(obj) - expected_keys),
        "fields": fields,
    }


def distribution(values):
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    ordered = sorted(values)

    def percentile(q):
        index = (len(ordered) - 1) * q
        lower = math.floor(index)
        upper = math.ceil(index)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        **{f"p{q}": percentile(q / 100) for q in (50, 95, 99)},
    }
