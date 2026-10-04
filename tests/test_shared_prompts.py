from pathlib import Path

import pytest

from jsonllm.fast_inference import decode_result
from jsonllm.prompts import (
    SHARED_VERSION,
    context_ids,
    field_example,
    prompt_ids,
    question_ids,
)


@pytest.mark.tokenizer
def test_qwen_shared_boundary_and_parent_isolation():
    path = Path("runs/speed-20260928-v1/merged-4b")
    if not path.exists():
        pytest.skip("Local pinned Qwen tokenizer absent")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    for context in ['텍스트 "\\\n\t 끝', " x\n", "👋<|im_end|>embedded"]:
        record = {
            "id": "r",
            "context": context,
            "questions": {
                "a": {"type": "boolean", "instructions": "first"},
                "b": {"type": "string", "instructions": "second", "depends_on": ["a"]},
                "c": {"type": "integer", "instructions": "unrelated"},
            },
        }
        example = field_example(
            record, "b", dependency_answers={"a": False, "c": 999}, prompt_version=SHARED_VERSION
        )
        assert prompt_ids(example, tokenizer) == context_ids(context, tokenizer) + question_ids(
            example, tokenizer
        )
        assert "999" not in example["messages"][1]["content"]
        assert decode_result(example, '"hello\\nworld"')["value"] == "hello\nworld"
