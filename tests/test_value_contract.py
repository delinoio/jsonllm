import json

import pytest

from jsonllm.config import TrainConfig
from jsonllm.evaluation import predict
from jsonllm.prompts import VALUE_VERSION, encode_example, field_example
from jsonllm.training import encoded_dataset


@pytest.mark.parametrize("value", [None, 0, 31, -2.75, 'quote " and\nUnicode café'])
def test_scalar_only_roundtrip_and_eos(record, tokenizer, value):
    name = "unrelated" if isinstance(value, str) else "weight"
    record["questions"][name]["type"] = ["string" if isinstance(value, str) else "number", "null"]
    record["answers"][name] = value
    example = field_example(record, name, prompt_version=VALUE_VERSION)
    assert example["prefill"] == ""
    assert json.loads(example["completion"]) == value
    encoded = encode_example(example, tokenizer)
    assert encoded["labels"][-1] == tokenizer.eos_token_id
    assert tokenizer.decode([x for x in encoded["labels"] if x != -100]) == example["completion"]

    class Predictor:
        def generate(self, ids, max_tokens):
            return example["completion"]

    predictor = Predictor()
    predictor.tokenizer = tokenizer
    result = predict(predictor, example, max_length=4096, max_tokens=256)
    assert result["valid"] and result["value"] == value


def test_choices_one_token_and_dependencies_isolated(record, tokenizer):
    example = field_example(record, "allowed", prompt_version=VALUE_VERSION)
    assert example["completion"] == "B"
    assert "unrelated" not in example["messages"][1]["content"]
    assert encode_example(example, tokenizer)["labels"][-2:] == tokenizer.encode("B") + [1]
    with pytest.raises(ValueError, match="version"):
        encoded_dataset([example], tokenizer, 4096)
    assert len(encoded_dataset([example], tokenizer, 4096, VALUE_VERSION)[0]) == 1


def test_default_contract_unchanged(record):
    assert TrainConfig().prompt_version == "field-v1"
    assert field_example(record, "weight")["completion"] == " 5}"
    with pytest.raises(ValueError, match="prompt_version"):
        TrainConfig(prompt_version="unknown")
