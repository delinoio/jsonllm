import pytest

from jsonllm.checkpoint_identity import text_checkpoint_keys


def test_known_prefix_and_tied_embedding_aliases_only():
    expected = {"model.embed_tokens.weight", "lm_head.weight", "model.layers.0.weight"}
    saved = {"model.language_model.embed_tokens.weight", "model.language_model.layers.0.weight"}
    mapping = text_checkpoint_keys(expected, saved, tied_embeddings=True)
    assert mapping["lm_head.weight"] == mapping["model.embed_tokens.weight"]
    with pytest.raises(ValueError):
        text_checkpoint_keys(expected, saved, tied_embeddings=False)
    with pytest.raises(ValueError):
        text_checkpoint_keys(expected, saved | {"unexpected.weight"}, tied_embeddings=True)
    with pytest.raises(ValueError, match="Duplicate"):
        text_checkpoint_keys(expected, saved | {"model.embed_tokens.weight"}, tied_embeddings=True)
