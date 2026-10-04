"""Explicit Qwen3.5 serialization aliases, without changing tensor values."""


def text_checkpoint_keys(expected, saved, *, tied_embeddings):
    expected, saved = set(expected), set(saved)
    mapping = {}
    for key in saved:
        normalized = (
            key.replace("model.language_model.", "model.", 1)
            if key.startswith("model.language_model.")
            else key
        )
        if normalized in mapping:
            raise ValueError("Duplicate normalized checkpoint key")
        mapping[normalized] = key
    if tied_embeddings and "lm_head.weight" in expected and "lm_head.weight" not in mapping:
        if "model.embed_tokens.weight" not in mapping:
            raise ValueError("Missing tied embedding tensor")
        mapping["lm_head.weight"] = mapping["model.embed_tokens.weight"]
    if set(mapping) != expected:
        raise ValueError("Checkpoint tensor names do not match after explicit aliases")
    return mapping
