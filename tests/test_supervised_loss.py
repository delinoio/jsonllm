import copy

import pytest

from jsonllm.backends.supervised_loss import supervised_loss

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.mark.parametrize("normalizer", [None, 17])
@pytest.mark.parametrize("adapter", [False, True])
def test_sparse_projection_matches_full_loss_and_every_gradient(normalizer, adapter, monkeypatch):
    # This is the CPU reference-kernel test even on hosts with CUDA FLA installed.
    # Actual 4B BF16 CUDA loss/gradients are gated by verify_shared_training.py.
    from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation

    for name in (
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    ):
        function = getattr(implementation, name)
        while hasattr(function, "__wrapped__"):
            function = function.__wrapped__
        monkeypatch.setattr(implementation, name, function)
    torch.manual_seed(3)
    config = transformers.Qwen3_5TextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        vocab_size=64,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 2],
            "mrope_interleaved": True,
        },
    )
    reference = transformers.Qwen3_5ForCausalLM(config)
    if adapter:
        peft = pytest.importorskip("peft")
        reference = peft.get_peft_model(
            reference,
            peft.LoraConfig(
                task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]
            ),
        )
    sparse = copy.deepcopy(reference)
    inputs = {
        "input_ids": torch.randint(0, 64, (2, 9)),
        "labels": torch.tensor([[-100] * 6 + [11, 12, 13], [-100] * 4 + [2, 3, 4, 5, 6]]),
    }
    reference.train()
    sparse.train()
    wanted = reference(**inputs, use_cache=False, num_items_in_batch=normalizer).loss
    widths = []
    base = sparse.get_base_model() if adapter else sparse
    hook = base.lm_head.register_forward_pre_hook(lambda _, args: widths.append(args[0].shape))
    actual, _ = supervised_loss(sparse, inputs, num_items_in_batch=normalizer)
    hook.remove()
    assert widths == [torch.Size([8, 32])]
    torch.testing.assert_close(actual, wanted, atol=1e-6, rtol=1e-6)
    wanted.backward()
    actual.backward()
    for (name, before), (_, after) in zip(
        reference.named_parameters(), sparse.named_parameters(), strict=True
    ):
        if before.grad is None:
            assert after.grad is None, name
        else:
            torch.testing.assert_close(after.grad, before.grad, atol=2e-6, rtol=2e-5, msg=name)
