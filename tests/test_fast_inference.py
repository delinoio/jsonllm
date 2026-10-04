import pytest

from jsonllm.fast_inference import decode_result, run_typed
from jsonllm.prompts import PROMPT_VERSION, VALUE_VERSION, field_example


class WavePredictor:
    def __init__(self, tokenizer, broken=False):
        self.tokenizer = tokenizer
        self.broken = broken
        self.waves = []

    def synchronize(self):
        pass

    def infer_many(self, examples, inputs, max_tokens):
        self.waves.append(examples)
        results = []
        for example, ids in zip(examples, inputs, strict=True):
            field = example["field"]
            text = {"weight": "8", "unrelated": '"package"', "allowed": "B"}[field]
            if self.broken and field == "weight":
                text = "NaN"
            if example["prompt_version"] == PROMPT_VERSION and not example["choices"]:
                text += "}"
            results.append(
                {
                    "text": text,
                    "metrics": {
                        "input_tokens": len(ids),
                        "output_tokens": 1,
                        "selected_tokens": 0,
                        "finish_reason": "stop",
                    },
                }
            )
        return results


def test_independent_wave_and_predicted_parent(record, tokenizer):
    p = WavePredictor(tokenizer)
    row = run_typed(record, p, prompt_version=VALUE_VERSION, max_length=4096)
    assert [[e["field"] for e in wave] for wave in p.waves] == [
        ["weight", "unrelated"],
        ["allowed"],
    ]
    assert '"weight":8' in p.waves[-1][0]["messages"][1]["content"]
    assert "unrelated" not in p.waves[-1][0]["messages"][1]["content"]
    assert row["schema_valid"] and not row["record_correct"]
    sequential = run_typed(
        record,
        WavePredictor(tokenizer),
        prompt_version=VALUE_VERSION,
        batch_fields=False,
        max_length=4096,
    )
    assert row["fields"] == sequential["fields"]


def test_failed_parent_blocks_descendant(record, tokenizer):
    p = WavePredictor(tokenizer, broken=True)
    row = run_typed(record, p, prompt_version=VALUE_VERSION, max_length=4096)
    assert row["errors"] == {"weight": "invalid_output", "allowed": "failed_dependency"}
    assert len(p.waves) == 1


@pytest.mark.parametrize("raw", ["NaN", '"31"', "31 trailing", "31 2", '{"weight":31}'])
def test_invalid_scalar_never_repaired(record, raw):
    assert not decode_result(field_example(record, "weight", prompt_version=VALUE_VERSION), raw)[
        "valid"
    ]


def test_complete_number_not_first_valid_prefix(record):
    example = field_example(record, "weight", prompt_version=VALUE_VERSION)
    assert decode_result(example, "31")["value"] == 31
    assert decode_result(example, "3")["value"] == 3


def test_legacy_object_is_still_supported(record):
    assert decode_result(field_example(record, "weight"), " 31}")["value"] == 31


@pytest.mark.parametrize("version", [PROMPT_VERSION, VALUE_VERSION])
@pytest.mark.parametrize("broken", [False, True])
def test_unlabeled_inference_preserves_dependency_and_failure_semantics(
    record, tokenizer, version, broken
):
    unlabeled = {key: value for key, value in record.items() if key != "answers"}
    predictor = WavePredictor(tokenizer, broken=broken)
    result = run_typed(
        unlabeled, predictor, prompt_version=version, max_length=4096, evaluate=False
    )
    assert "record_correct" not in result
    assert all("expected" not in field and "correct" not in field for field in result["fields"])
    assert result["schema_valid"] is not broken
    fields = {f["field"]: f for f in result["fields"]}
    if broken:
        assert not fields["weight"]["present"] and not fields["allowed"]["present"]
        assert result["errors"]["allowed"] == "failed_dependency"
    else:
        assert fields["weight"]["value"] == 8
        assert '"weight":8' in predictor.waves[-1][0]["messages"][1]["content"]
        assert "unrelated" not in predictor.waves[-1][0]["messages"][1]["content"]


def test_unlabeled_request_requires_supported_nonempty_schema(record, tokenizer):
    predictor = WavePredictor(tokenizer)
    with pytest.raises(ValueError, match="non-empty object"):
        run_typed(record | {"questions": {}}, predictor, evaluate=False)
    assert not predictor.waves


def test_inference_only_ignores_labels_and_preserves_real_null(record, tokenizer):
    class NullPredictor(WavePredictor):
        def infer_many(self, examples, inputs, max_tokens):
            results = super().infer_many(examples, inputs, max_tokens)
            for example, result in zip(examples, results, strict=True):
                if example["field"] == "unrelated":
                    result["text"] = "null"
            return results

    record["questions"]["unrelated"]["type"] = ["string", "null"]
    record["answers"]["allowed"] = "invalid reference label must be ignored"
    row = run_typed(
        record,
        NullPredictor(tokenizer),
        prompt_version=VALUE_VERSION,
        max_length=4096,
        evaluate=False,
    )
    field = next(f for f in row["fields"] if f["field"] == "unrelated")
    assert field["present"] and field["valid"] and field["value"] is None
    assert row["schema_valid"]
