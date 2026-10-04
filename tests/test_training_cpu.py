"""Original CPU text-extraction and supervised-mask regression test."""

import inspect

import pytest


def tiny_text_config(vocab_size=64):
    return {
        "model_type": "qwen3_5_text",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "vocab_size": vocab_size,
        "linear_num_value_heads": 2,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 2,
        "rope_parameters": {
            "type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 2],
        },
    }


def test_torch_text_extraction_and_completion_mask(tmp_path, record, tokenizer, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5

    from jsonllm.backends.cuda import Collator, load_model
    from jsonllm.prompts import encode_example, field_example, padded_batch

    values = tiny_text_config()
    values.pop("model_type")
    values.pop("full_attention_interval")
    values["rope_parameters"]["rope_type"] = values["rope_parameters"].pop("type")
    text_config = Qwen3_5TextConfig(**values, layer_types=["linear_attention", "full_attention"])
    full_config = Qwen3_5Config(
        text_config=text_config.to_dict(),
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
        },
    )
    full_model = Qwen3_5ForConditionalGeneration(full_config)
    full_model.save_pretrained(tmp_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        # CUDA extras may be installed on a CPU-only test runner. Exercise the
        # reference path there; on GPU this test uses the installed FLA kernels.
        for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
            monkeypatch.setattr(
                modeling_qwen3_5, name, inspect.unwrap(getattr(modeling_qwen3_5, name))
            )
    text_model = load_model(str(tmp_path), "main", device=device)
    assert torch.equal(
        text_model.model.embed_tokens.weight.cpu(),
        full_model.model.language_model.embed_tokens.weight.to(torch.bfloat16),
    )
    targets = [
        n
        for n, m in text_model.named_modules()
        if n.startswith("model.layers.") and isinstance(m, torch.nn.Linear)
    ]
    model = get_peft_model(
        text_model, LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8, target_modules=targets)
    )
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    assert names and all("lora_" in name for name in names)
    assert not any("embed_tokens" in name or "lm_head" in name for name in names)
    rows = [
        encode_example(field_example(record, name), tokenizer) for name in ("weight", "allowed")
    ]
    actual = Collator(tokenizer)(rows)
    expected = padded_batch(rows, tokenizer.pad_token_id)
    assert {key: value.tolist() for key, value in actual.items()} == expected
    small = {
        "input_ids": torch.tensor([[4, 5, 6, 7, 8, 1]]),
        "labels": torch.tensor([[-100, -100, -100, 7, 8, 1]]),
    }
    loss = model(**{key: value.to(device) for key, value in small.items()}).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad)
        for n, p in model.named_parameters()
        if "lora_B" in n
    )
