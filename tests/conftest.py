import pytest


class CharacterTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def encode(self, text, **kwargs):
        return [ord(c) + 2 for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i - 2) for i in ids if i > 1)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and not enable_thinking
        return "".join(f"<{m['role']}>\n{m['content']}\n" for m in messages) + "<assistant>\n"


@pytest.fixture
def tokenizer():
    return CharacterTokenizer()


@pytest.fixture
def record():
    return {
        "id": "case",
        "context": "A package weighs 5 kg. Maximum allowed weight is 10 kg.",
        "questions": {
            "weight": {"type": "number", "instructions": "Extract weight in kg."},
            "allowed": {
                "type": "boolean",
                "instructions": "Is the package allowed?",
                "depends_on": ["weight"],
            },
            "unrelated": {"type": "string", "instructions": "Return package."},
        },
        "answers": {"weight": 5, "allowed": True, "unrelated": "package"},
    }
