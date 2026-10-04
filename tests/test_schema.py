import pytest
from jsonschema.exceptions import ValidationError

from jsonllm.schema import ancestors, dependency_order, validate_questions, validate_value


@pytest.mark.parametrize(
    "field,value",
    [
        ({"type": "integer"}, True),
        ({"type": "integer"}, 1.0),
        ({"type": "number"}, float("nan")),
        ({"type": "number"}, float("inf")),
        ({"type": "number", "minimum": 1}, 0),
        ({"type": "string", "maxLength": 2}, "long"),
        ({"type": "string", "enum": ["yes"]}, "no"),
        ({"type": "boolean"}, None),
    ],
)
def test_reject_invalid_values(field, value):
    with pytest.raises((ValueError, ValidationError)):
        validate_value(field, value)


def test_nullable_and_transitive_dependencies(record):
    validate_value({"type": ["integer", "null"]}, None)
    record["questions"]["last"] = {
        "type": "boolean",
        "instructions": "Continue.",
        "depends_on": ["allowed"],
    }
    validate_questions(record["questions"])
    assert ancestors(record["questions"], "last") == ["weight", "allowed"]
    assert dependency_order(record["questions"]).index("weight") < dependency_order(
        record["questions"]
    ).index("allowed")


@pytest.mark.parametrize(
    "mutation",
    [
        {"depends_on": ["absent"]},
        {"depends_on": ["weight"]},
        {"type": "array"},
        {"enum": [1, 1]},
        {"minimum": 2, "maximum": 1},
        {"pattern": "unsupported"},
        {"type": ["number", "string", "null"]},
    ],
)
def test_bad_schemas(record, mutation):
    record["questions"]["weight"].update(mutation)
    with pytest.raises((ValueError, ValidationError)):
        validate_questions(record["questions"])
