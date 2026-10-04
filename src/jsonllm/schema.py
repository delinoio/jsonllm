"""The deliberately small TypeLLM-compatible scalar schema subset."""

import math

from jsonschema import Draft202012Validator

from .io import dumps

TYPES = {"string", "integer", "number", "boolean"}
KEYS = {
    "type",
    "enum",
    "instructions",
    "description",
    "minimum",
    "maximum",
    "maxLength",
    "depends_on",
}


def base_type(field):
    types = field["type"]
    return next(t for t in types if t != "null") if isinstance(types, list) else types


def nullable(field):
    return isinstance(field["type"], list) and "null" in field["type"]


def value_schema(field):
    return {
        k: v for k, v in field.items() if k in {"type", "enum", "minimum", "maximum", "maxLength"}
    }


def validate_value(field, value):
    # jsonschema accepts NaN/Infinity as numbers; JSON and our target format do not.
    dumps(value)
    Draft202012Validator(value_schema(field)).validate(value)
    if base_type(field) == "integer" and value is not None and type(value) is not int:
        raise ValueError("Integer fields require a JSON integer, not 1.0 or a boolean")


def validate_questions(questions):
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object")
    for name, field in questions.items():
        if not isinstance(name, str) or not name or not isinstance(field, dict):
            raise ValueError("Each field requires a non-empty name and object schema")
        if unknown := set(field) - KEYS:
            raise ValueError(f"{name}: unsupported schema keywords: {sorted(unknown)}")
        types = field.get("type")
        if isinstance(types, list):
            if len(types) != 2 or types.count("null") != 1:
                raise ValueError(f"{name}: only [scalar, null] unions are supported")
            types = next(t for t in types if t != "null")
        if not isinstance(types, str) or types not in TYPES:
            raise ValueError(f"{name}: unsupported scalar type")
        instructions = field.get("instructions", field.get("description"))
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError(f"{name}: instructions or description is required")
        for key in ("minimum", "maximum"):
            if key in field:
                if types not in {"integer", "number"} or type(field[key]) not in {int, float}:
                    raise ValueError(f"{name}: {key} requires a number on a numeric field")
                if not math.isfinite(field[key]):
                    raise ValueError(f"{name}: non-finite bound")
        if field.get("minimum", -math.inf) > field.get("maximum", math.inf):
            raise ValueError(f"{name}: minimum exceeds maximum")
        if "maxLength" in field and (
            types != "string" or type(field["maxLength"]) is not int or field["maxLength"] < 0
        ):
            raise ValueError(f"{name}: invalid maxLength")
        if "enum" in field:
            values = field["enum"]
            if types == "boolean" or not isinstance(values, list) or not 1 <= len(values) <= 24:
                raise ValueError(f"{name}: enum requires 1–24 string/integer/number values")
            if any(a == b for i, a in enumerate(values) for b in values[i + 1 :]):
                raise ValueError(f"{name}: duplicate enum value")
            for value in values:
                validate_value(field, value)
        deps = field.get("depends_on", [])
        if (
            not isinstance(deps, list)
            or any(not isinstance(d, str) for d in deps)
            or len(set(deps)) != len(deps)
        ):
            raise ValueError(f"{name}: depends_on must contain distinct field names")
        if any(d not in questions for d in deps):
            raise ValueError(f"{name}: unknown dependency")
    dependency_order(questions)


def dependency_order(questions):
    order, active = [], set()

    def visit(name):
        if name in active:
            raise ValueError("Cyclic depends_on graph")
        if name in order:
            return
        active.add(name)
        for parent in questions[name].get("depends_on", []):
            visit(parent)
        active.remove(name)
        order.append(name)

    for name in questions:
        visit(name)
    return order


def ancestors(questions, name):
    visible = set()

    def visit(current):
        for parent in questions[current].get("depends_on", []):
            if parent not in visible:
                visible.add(parent)
                visit(parent)

    visit(name)
    return [key for key in dependency_order(questions) if key in visible]


def validate_record(record):
    if not isinstance(record, dict):
        raise ValueError("Each record must be an object")
    if not isinstance(record.get("id"), str) or not record["id"]:
        raise ValueError("Record requires a non-empty string id")
    if not isinstance(record.get("context"), str) or not record["context"].strip():
        raise ValueError("Record requires non-empty context")
    validate_questions(record.get("questions"))
    if not isinstance(record.get("answers"), dict):
        raise ValueError("answers must be an object")
    if set(record["answers"]) != set(record["questions"]):
        raise ValueError("answers must contain exactly the question fields")
    for name, field in record["questions"].items():
        validate_value(field, record["answers"][name])


def candidates(field):
    if "enum" in field:
        return field["enum"]
    if base_type(field) == "boolean":
        return [False, True] + ([None] if nullable(field) else [])
    return None


def object_schema(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
