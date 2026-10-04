import json

from jsonllm.prompts import encode_example, field_example, padded_batch
from jsonllm.training import encoded_dataset


def test_only_ancestors_visible(record):
    example = field_example(record, "allowed")
    prompt = example["messages"][1]["content"]
    assert '"weight":5' in prompt
    assert "unrelated" not in prompt
    assert "Dependency results" not in field_example(record, "weight")["messages"][1]["content"]


def test_permutation_remaps_label(record):
    one = field_example(record, "allowed", [False, True])
    two = field_example(record, "allowed", [True, False])
    assert one["completion"] == 'B"}'
    assert two["completion"] == 'A"}'


def test_mask_includes_completion_eos_excludes_prompt_and_padding(record, tokenizer):
    example = field_example(record, "allowed")
    encoded = encode_example(example, tokenizer)
    supervised = [label for label in encoded["labels"] if label != -100]
    assert supervised == tokenizer.encode('B"}') + [tokenizer.eos_token_id]
    shorter = {"input_ids": [4, 5, 1], "labels": [-100, 5, 1]}
    batch = padded_batch([encoded, shorter], tokenizer.pad_token_id)
    assert batch["labels"][1][3:] == [-100] * (len(encoded["input_ids"]) - 3)
    assert batch["attention_mask"][1][:4] == [1, 1, 1, 0]


def test_numbers_no_exponent_and_strings_escaped(record, tokenizer):
    record["answers"]["weight"] = 1e-8
    example = field_example(record, "weight")
    assert example["completion"] == " 0.00000001}"
    record["answers"]["unrelated"] = 'quote " and\nnewline'
    example = field_example(record, "unrelated")
    assert (
        json.loads(example["prefill"] + example["completion"])["unrelated"]
        == record["answers"]["unrelated"]
    )


def test_overlength_excluded_not_truncated(record, tokenizer):
    row = field_example(record, "weight") | {"id": "long"}
    retained, excluded = encoded_dataset([row], tokenizer, max_length=4)
    assert retained == []
    assert excluded[0]["id"] == "long" and excluded[0]["tokens"] > 4
