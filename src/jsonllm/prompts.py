"""One prompt/continuation contract for preparing, training, and evaluating."""

import json
from decimal import Decimal
from string import ascii_uppercase

from .io import dumps
from .schema import ancestors, base_type, candidates, nullable

PROMPT_VERSION = "field-v1"
VALUE_VERSION = "typed-value-v2"
SHARED_VERSION = "shared-context-v3"
PROMPT_VERSIONS = (PROMPT_VERSION, VALUE_VERSION, SHARED_VERSION)


def scalar_json(value):
    if type(value) is float:
        return format(Decimal(str(value)), "f")
    return dumps(value)


def field_example(
    record, name, values=None, dependency_answers=None, *, prompt_version=PROMPT_VERSION
):
    if prompt_version not in PROMPT_VERSIONS:
        raise ValueError("Unknown prompt version")
    field = record["questions"][name]
    values = candidates(field) if values is None else values
    kind = base_type(field)
    type_text = ("boolean" if kind == "boolean" else "choice") if values else kind
    if nullable(field) and not values:
        type_text += " or null"
    for key in ("minimum", "maximum"):
        if key in field:
            type_text += f", {key} {scalar_json(field[key])}"
    if "maxLength" in field:
        type_text += f", at most {field['maxLength']} characters"
    lines = [
        f"Field: {dumps(name)}",
        f"Type: {type_text}",
        f"Instructions: {field.get('instructions', field.get('description'))}",
    ]
    mapping = dict(zip(ascii_uppercase, values, strict=False)) if values else {}
    if mapping:
        lines.append(f"Choices: {json.dumps(mapping, ensure_ascii=False, allow_nan=False)}")
        lines.append(f'Answer as {{{dumps(name)}: "<label>"}}.')
        prefill = "{" + dumps(name) + ': "'
    else:
        placeholder = kind + (" or null" if nullable(field) else "")
        lines.append(f"Answer as {{{dumps(name)}: <{placeholder}>}}.")
        if kind == "number":
            lines.append("Do not use exponent notation.")
        if nullable(field):
            lines.append("Return null only if there is no value.")
        prefill = "{" + dumps(name) + ":"
    answers = record.get("answers", {}) if dependency_answers is None else dependency_answers
    visible = {parent: answers[parent] for parent in ancestors(record["questions"], name)}
    question = "\n".join(lines)
    if visible:
        question = "Dependency results (JSON):\n" + dumps(visible) + "\n\n" + question
    target = None
    if name in record.get("answers", {}):
        value = record["answers"][name]
        if mapping:
            label = next(key for key, candidate in mapping.items() if candidate == value)
            target = label + '"}'
        else:
            target = " " + scalar_json(value) + "}"
    if prompt_version in (VALUE_VERSION, SHARED_VERSION):
        # Keep the state before the question so independent fields share a token prefix.
        lines = [line for line in lines if not line.startswith("Answer as ")]
        lines.append(
            "Return only the choice label."
            if mapping
            else "Return only the JSON scalar value, then end the answer."
        )
        question = "\n".join(lines)
        if visible:
            question = "Dependency results (JSON):\n" + dumps(visible) + "\n\n" + question
        prefill = ""
        if target is not None:
            target = label if mapping else scalar_json(value)
    return {
        "record_id": record["id"],
        "field": name,
        "schema": field,
        "messages": [
            {"role": "user", "content": record["context"].rstrip()},
            {"role": "user", "content": question},
        ],
        "prefill": prefill,
        "completion": target,
        "choices": mapping,
        "prompt_version": prompt_version,
    }


def prompt_ids(example, tokenizer):
    if example.get("prompt_version", PROMPT_VERSION) == SHARED_VERSION:
        common = context_ids(example["messages"][0]["content"], tokenizer)
        suffix = question_ids(example, tokenizer)
        rendered = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        if common + suffix != tokenizer.encode(rendered, add_special_tokens=False):
            raise ValueError("Shared prompt boundary is not token-equivalent")
        return common + suffix
    rendered = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    # Tokenize the entire prefix once: never concatenate separately tokenized chat fragments.
    prefix = tokenizer.encode(rendered + example["prefill"], add_special_tokens=False)
    for label in example["choices"]:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != label:
            raise ValueError(f"Label {label!r} is not a standalone single token")
        if (
            tokenizer.encode(rendered + example["prefill"] + label, add_special_tokens=False)
            != prefix + encoded
        ):
            raise ValueError("Tokenizer merges the label into the answer prefill")
    return prefix


def context_ids(context, tokenizer):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": context.rstrip()}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return tokenizer.encode(rendered, add_special_tokens=False)


def question_ids(example, tokenizer):
    rendered = tokenizer.apply_chat_template(
        example["messages"][1:],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    for label in example["choices"]:
        tokens = tokenizer.encode(label, add_special_tokens=False)
        if len(tokens) != 1 or tokenizer.decode(tokens) != label:
            raise ValueError("Choice label must be one token")
        if tokenizer.encode(rendered + label, add_special_tokens=False) != ids + tokens:
            raise ValueError("Choice token merges across the answer boundary")
    return ids


def encode_example(example, tokenizer):
    prefix = prompt_ids(example, tokenizer)
    if example["completion"] is None:
        raise ValueError("Training examples require a completion")
    # Continuations start at the inference boundary, even when ordinary BPE would merge it.
    completion = tokenizer.encode(example["completion"], add_special_tokens=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer requires an EOS token")
    tokens = prefix + completion + [tokenizer.eos_token_id]
    return {"input_ids": tokens, "labels": [-100] * len(prefix) + tokens[len(prefix) :]}


def padded_batch(examples, pad_id):
    width = max(len(row["input_ids"]) for row in examples)
    return {
        "input_ids": [r["input_ids"] + [pad_id] * (width - len(r["input_ids"])) for r in examples],
        "labels": [r["labels"] + [-100] * (width - len(r["labels"])) for r in examples],
        "attention_mask": [
            [1] * len(r["input_ids"]) + [0] * (width - len(r["input_ids"])) for r in examples
        ],
    }
