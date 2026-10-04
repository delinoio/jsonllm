"""Per-field JSON grammars; each generation owns its mutable matcher state."""

from contextlib import nullcontext
from dataclasses import dataclass

JSON_CHAR = r'json_char ::= [^"\\\x00-\x1f] | "\\" (["\\/bfnrt] | "u" [a-fA-F0-9]{4})'


@dataclass
class MaskWorkspace:
    host: object
    device: object


def bounded_string_grammar(minimum, maximum, *, nullable=False):
    """Right-factored length states keep XGrammar's per-token frontier small.

    Count the same json_char units as the old bounded repetition, including one
    unit per Unicode escape (not one per decoded surrogate pair).
    """
    if minimum < 0 or maximum < minimum:
        raise ValueError("Invalid string length bounds")
    rules = [r'root ::= "\"" rest_0' + (' | "null"' if nullable else "")]
    for length in range(maximum + 1):
        alternatives = []
        if length < maximum:
            alternatives.append(f"json_char rest_{length + 1}")
        if length >= minimum:
            alternatives.append(r'"\""')
        rules.append(f"rest_{length} ::= " + " | ".join(alternatives))
    return "\n".join([*rules, JSON_CHAR]) + "\n"


class ScalarGrammar:
    def __init__(self, tokenizer, vocab_size):
        import xgrammar as xgr

        self.xgr = xgr
        self.vocab_size = vocab_size
        info = xgr.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=vocab_size, stop_token_ids=[tokenizer.eos_token_id]
        )
        self.compiler = xgr.GrammarCompiler(info, max_threads=1)

    def matcher(self, field):
        schema = {
            k: field[k]
            for k in ("type", "enum", "minimum", "maximum", "minLength", "maxLength")
            if k in field
        }
        kind = schema["type"]
        if (
            "enum" not in schema
            and "maxLength" in schema
            and (kind == "string" or isinstance(kind, list) and set(kind) == {"string", "null"})
        ):
            # Keep the existing JSON character language, including escape counting.
            ebnf = bounded_string_grammar(
                schema.get("minLength", 0), schema["maxLength"], nullable=isinstance(kind, list)
            )
            compiled = self.compiler.compile_grammar(self.xgr.Grammar.from_ebnf(ebnf))
        else:
            compiled = self.compiler.compile_json_schema(schema, any_whitespace=False)
        # A valid numeric prefix is not a completed answer until the model selects EOS.
        return self.xgr.GrammarMatcher(compiled, terminate_without_stop_token=False)

    def allocate(self, count, device="cpu"):
        import torch

        host = self.xgr.allocate_token_bitmask(count, self.vocab_size)
        target = (
            host if torch.device(device).type == "cpu" else torch.empty_like(host, device=device)
        )
        return MaskWorkspace(host, target)

    def mask(self, logits, matchers, workspace, *, profile=None):
        count = len(matchers)
        with profile("grammar_fill") if profile is not None else nullcontext():
            for row, matcher in enumerate(matchers):
                matcher.fill_next_token_bitmask(workspace.host, row)
        mask = workspace.device[:count]
        if workspace.device is not workspace.host:
            # Synchronous by design: the matcher can overwrite host storage next step.
            with profile("mask_transfer", gpu=True) if profile is not None else nullcontext():
                mask.copy_(workspace.host[:count])
            if profile is not None and profile.enabled:
                profile.add("mask_transfer_bytes", mask.numel() * mask.element_size())
        with profile("mask_apply", gpu=True) if profile is not None else nullcontext():
            self.xgr.apply_token_bitmask_inplace(logits, mask)
