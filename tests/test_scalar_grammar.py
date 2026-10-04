import os
from pathlib import Path

import pytest

from jsonllm.backends.scalar_grammar import ScalarGrammar
from jsonllm.training import load_tokenizer

torch = pytest.importorskip("torch")
pytest.importorskip("xgrammar")


@pytest.fixture(scope="module")
def tokenizer():
    path = Path(os.environ.get("TYPELLM_TEST_TOKENIZER", "runs/speed-20260928-v1/merged-4b"))
    if not path.exists():
        pytest.skip("Requires local pinned Qwen tokenizer")
    return load_tokenizer(str(path), "main")


@pytest.fixture(scope="module")
def grammar(tokenizer):
    return ScalarGrammar(tokenizer, 248320)


@pytest.mark.parametrize(
    "schema,text",
    [
        ({"type": "string", "maxLength": 30}, '"한글\\n\\"quoted\\""'),
        ({"type": ["string", "null"]}, "null"),
        ({"type": "number"}, "120.5"),
        ({"type": "number"}, "0"),
        ({"type": "boolean"}, "false"),
    ],
)
def test_scalar_grammar_accepts_valid_scalar_and_only_terminates_at_eos(
    grammar, tokenizer, schema, text
):
    matcher = grammar.matcher(schema)
    mask = grammar.allocate(1)
    for token in tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]:
        logits = torch.zeros((1, grammar.vocab_size))
        grammar.mask(logits, [matcher], mask)
        assert torch.isfinite(logits[0, token])
        assert matcher.accept_token(token)
        assert matcher.is_terminated() == (token == tokenizer.eos_token_id)


def test_numeric_prefix_keeps_digits_available(grammar, tokenizer):
    matcher = grammar.matcher({"type": "number"})
    assert matcher.accept_string("12")
    assert not matcher.is_terminated()
    logits = torch.zeros((1, grammar.vocab_size))
    grammar.mask(logits, [matcher], grammar.allocate(1))
    assert torch.isfinite(logits[0, tokenizer.encode("3", add_special_tokens=False)[0]])
    assert torch.isfinite(logits[0, tokenizer.eos_token_id])


def legacy_matcher(grammar, schema):
    # Freeze the original grammar independently of the optimized implementation.
    nullable = isinstance(schema["type"], list)
    minimum, maximum = schema.get("minLength", 0), schema["maxLength"]
    null = ' | "null"' if nullable else ""
    ebnf = (
        f'root ::= "\\"" json_char{{{minimum},{maximum}}} "\\""{null}\n'
        r'json_char ::= [^"\\\x00-\x1f] | "\\" (["\\/bfnrt] | "u" [a-fA-F0-9]{4})' + "\n"
    )
    compiled = grammar.compiler.compile_grammar(grammar.xgr.Grammar.from_ebnf(ebnf))
    return grammar.xgr.GrammarMatcher(compiled, terminate_without_stop_token=False)


@pytest.mark.parametrize(
    "minimum,maximum,text,accepted",
    [
        (0, 0, '""', True),
        (1, 1, '""', False),
        (3, 3, '"abc"', True),
        (3, 3, '"ab"', False),
        (3, 3, '"abcd"', False),
        (0, 180, '"' + "a" * 180 + '"', True),
        (0, 180, '"' + "a" * 181 + '"', False),
        (0, 180, r'"\"\\\/\b\f\n\r\t\u0041"', True),
        (0, 180, '"한글 😀 café é"', True),
        (1, 1, '"😀"', True),
        (1, 1, r'"\ud83d\ude00"', False),
        (2, 2, r'"\ud83d\ude00"', True),
        (1, 1, '"é"', True),
        (2, 2, '"é"', False),
        (0, 180, "null", True),
        (0, 180, r'"\x"', False),
        (0, 180, '"raw\nnewline"', False),
    ],
)
def test_full_token_masks_match_legacy(grammar, tokenizer, minimum, maximum, text, accepted):
    schema = {"type": ["string", "null"], "minLength": minimum, "maxLength": maximum}
    old, new = legacy_matcher(grammar, schema), grammar.matcher(schema)
    masks = [grammar.allocate(1).host, grammar.allocate(1).host]
    completed = True
    for token in tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]:
        for matcher, mask in zip([old, new], masks, strict=True):
            matcher.fill_next_token_bitmask(mask)
        assert torch.equal(*masks)
        outcomes = [m.accept_token(token) for m in (old, new)]
        assert outcomes[0] == outcomes[1]
        if not outcomes[0]:
            completed = False
            break
    assert completed == accepted
    assert old.is_terminated() == new.is_terminated() == accepted


def test_seeded_partial_prefix_masks_match_legacy(grammar):
    import random

    rng = random.Random(20261003)
    atoms = ["a", "한", "😀", "é", r"\n", r"\"", r"\u0041", r"\\"]
    for _ in range(24):
        maximum = rng.choice([1, 4, 12, 180])
        schema = {"type": "string", "minLength": rng.randrange(maximum + 1), "maxLength": maximum}
        # Includes partial escapes and Unicode sequences, not just token boundaries.
        text = '"' + "".join(rng.choices(atoms, k=rng.randrange(1, 8)))
        prefix = text[: rng.randrange(1, len(text) + 1)]
        old, new = legacy_matcher(grammar, schema), grammar.matcher(schema)
        assert old.accept_string(prefix) == new.accept_string(prefix)
        masks = [grammar.allocate(1).host, grammar.allocate(1).host]
        old.fill_next_token_bitmask(masks[0])
        new.fill_next_token_bitmask(masks[1])
        assert torch.equal(*masks)


def test_tokenizer_normalization_and_raw_combining_character_lengths(grammar, tokenizer):
    text = '"é"'
    # Qwen normalizes the input to NFC when encoding; raw matcher input does not.
    assert tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)) == '"é"'
    for length, accepted in [(1, False), (2, True)]:
        schema = {"type": "string", "minLength": length, "maxLength": length}
        for matcher in (legacy_matcher(grammar, schema), grammar.matcher(schema)):
            assert matcher.accept_string(text) == accepted


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string", "enum": ["x", "y"], "maxLength": 1},
        {"type": "string"},
        {"type": "number", "minimum": 1, "maximum": 9},
        {"type": "boolean"},
    ],
)
def test_other_schema_paths_keep_the_same_compilation(grammar, schema):
    reference = grammar.xgr.GrammarMatcher(
        grammar.compiler.compile_json_schema(schema, any_whitespace=False),
        terminate_without_stop_token=False,
    )
    masks = [grammar.allocate(1).host, grammar.allocate(1).host]
    reference.fill_next_token_bitmask(masks[0])
    grammar.matcher(schema).fill_next_token_bitmask(masks[1])
    assert torch.equal(*masks)


@pytest.mark.parametrize("separate_target", [False, True])
def test_mask_workspace_reuses_storage_and_compacts_matcher_rows(grammar, separate_target):
    from jsonllm.backends.scalar_grammar import MaskWorkspace

    workspace = grammar.allocate(3)
    assert workspace.host is workspace.device
    if separate_target:
        # Exercise the same copy path locally; CUDA coverage remains a separate gate.
        workspace = MaskWorkspace(workspace.host, torch.empty_like(workspace.host))
    pointers = workspace.host.data_ptr(), workspace.device.data_ptr()
    fields = [{"type": "number"}, {"type": "boolean"}, {"type": "string", "maxLength": 4}]
    matchers = [grammar.matcher(field) for field in fields]
    for active, prefixes in [([0, 1, 2], ["1", "f", '"']), ([0, 2], ["2", "a"]), ([2], ["b"])]:
        current = [matchers[i] for i in active]
        logits = torch.zeros((len(current), grammar.vocab_size))
        grammar.mask(logits, current, workspace)
        for row, matcher in enumerate(current):
            reference = grammar.allocate(1)
            expected = torch.zeros((1, grammar.vocab_size))
            grammar.mask(expected, [matcher], reference)
            assert torch.equal(expected[0], logits[row])
        assert (workspace.host.data_ptr(), workspace.device.data_ptr()) == pointers
        for matcher, prefix in zip(current, prefixes, strict=True):
            assert matcher.accept_string(prefix)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA mask transfer requires GPU")
def test_cuda_mask_workspace_matches_cpu(grammar):
    fields = [{"type": "number"}, {"type": "boolean"}]
    matchers = [grammar.matcher(field) for field in fields]
    workspace = grammar.allocate(2, "cuda")
    pointer = workspace.device.data_ptr()
    for current in [matchers, matchers[1:]]:
        actual = torch.zeros((len(current), grammar.vocab_size), device="cuda")
        expected = torch.zeros_like(actual, device="cpu")
        grammar.mask(actual, current, workspace)
        grammar.mask(expected, current, grammar.allocate(len(current)))
        assert torch.equal(actual.cpu(), expected)
        assert workspace.device.data_ptr() == pointer


def test_profiled_mask_has_same_output_and_accounts_active_transfer_bytes(grammar):
    from jsonllm.backends.scalar_grammar import MaskWorkspace
    from jsonllm.backends.stage_profile import StageProfiler

    workspace = grammar.allocate(3)
    workspace = MaskWorkspace(workspace.host, torch.empty_like(workspace.host))
    metrics = {}
    profile = StageProfiler(True, torch.device("cpu"), metrics, [])
    matcher = grammar.matcher({"type": "number"})
    for count in [3, 1]:
        actual = torch.zeros((count, grammar.vocab_size))
        expected = actual.clone()
        grammar.mask(actual, [matcher] * count, workspace, profile=profile)
        grammar.mask(expected, [matcher] * count, grammar.allocate(count))
        assert torch.equal(actual, expected)
    assert set(metrics["stage_host_seconds"]) == {"grammar_fill", "mask_transfer", "mask_apply"}
    assert metrics["mask_transfer_bytes"] == workspace.host[0].numel() * 4 * 4
