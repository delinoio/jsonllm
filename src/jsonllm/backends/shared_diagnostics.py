"""Bounded, failure-only Qwen prefill diagnostics; never an accepted runtime path."""

from contextlib import contextmanager


@contextmanager
def model_precision(model, dtype):
    """Restore exact original tensors, not a lossy conversion back to their dtype."""
    parameters = [(parameter, parameter.data) for parameter in model.parameters()]
    buffers = [(module, module._buffers.copy()) for module in model.modules()]
    try:
        model.to(dtype=dtype)
        yield
    finally:
        for parameter, data in parameters:
            parameter.data = data
        for module, original in buffers:
            module._buffers.clear()
            module._buffers.update(original)


def tensor_delta(actual, expected):
    import torch

    actual, expected = actual.detach().float(), expected.detach().float()
    difference = actual - expected
    finite = torch.isfinite(actual).all() & torch.isfinite(expected).all()
    return {
        "shape": list(actual.shape),
        "finite": finite.item(),
        "unequal_elements": (actual != expected).sum().item(),
        "max_absolute_delta": difference.abs().max().item() if finite else None,
        "rms_delta": difference.square().mean().sqrt().item() if finite else None,
    }


@contextmanager
def replace_chunk_kernel(module, function):
    original = module.torch_chunk_gated_delta_rule
    try:
        module.torch_chunk_gated_delta_rule = function
        yield
    finally:
        module.torch_chunk_gated_delta_rule = original


def prefill_trace(model, device, prefix, suffix, module):
    """Find the first differing module and capture first-layer kernel inputs for replay."""
    import torch

    traces, kernels, handles = {}, {}, []
    phase = "full"
    original = module.torch_chunk_gated_delta_rule

    def kernel(query, key, value, **kwargs):
        output, state = original(query, key, value, **kwargs)
        if phase not in kernels:
            kernels[phase] = {
                "inputs": {
                    name: tensor.detach().clone()
                    for name, tensor in dict(
                        query=query, key=key, value=value, g=kwargs["g"], beta=kwargs["beta"]
                    ).items()
                },
                "output": output.detach().clone(),
                "state": state.detach().clone(),
            }
        return output, state

    def hook(name, unused_module, unused_inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        # Last-token vectors suffice to locate the first divergent operation on
        # the continuation. Keep prefix-end vectors separately, in execution order.
        if not isinstance(output, torch.Tensor) or output.ndim != 3 or output.shape[0] != 1:
            return
        row = {"last": output[0, -1].detach().float().cpu().clone()}
        if phase == "full":
            row["prefix_end"] = output[0, len(prefix) - 1].detach().float().cpu().clone()
        traces.setdefault(phase, {})[name] = row

    from functools import partial

    try:
        for name, layer in model.model.named_modules():
            if name.startswith("layers.") or name in ("embed_tokens", "norm"):
                handles.append(layer.register_forward_hook(partial(hook, name)))
        with replace_chunk_kernel(module, kernel):
            full = model.model(
                input_ids=torch.tensor([prefix + suffix], device=device), use_cache=True
            )
            del full
            phase = "prefix"
            common = model.model(input_ids=torch.tensor([prefix], device=device), use_cache=True)
            phase = "suffix"
            split = model.model(
                input_ids=torch.tensor([suffix], device=device),
                past_key_values=common.past_key_values,
                use_cache=True,
            )
            del common, split
    finally:
        for handle in handles:
            handle.remove()

    comparisons = [
        {
            "module": name,
            "prefix_end": tensor_delta(traces["prefix"][name]["last"], row["prefix_end"]),
            "suffix_end": tensor_delta(traces["suffix"][name]["last"], row["last"]),
        }
        for name, row in traces["full"].items()
    ]
    report = {
        "module_outputs": comparisons,
        "first_differing_suffix_module": next(
            (row["module"] for row in comparisons if row["suffix_end"]["unequal_elements"]), None
        ),
        "first_kernel_inputs": {
            name: tensor_delta(
                torch.cat(
                    [kernels["prefix"]["inputs"][name], kernels["suffix"]["inputs"][name]], dim=1
                ),
                full_input,
            )
            for name, full_input in kernels["full"]["inputs"].items()
        },
        "first_kernel_output": tensor_delta(
            torch.cat([kernels["prefix"]["output"], kernels["suffix"]["output"]], dim=1),
            kernels["full"]["output"],
        ),
        "first_kernel_final_state": tensor_delta(
            kernels["suffix"]["state"], kernels["full"]["state"]
        ),
    }
    return report, kernels["full"]["inputs"]


def isolated_kernel_probes(inputs, cuts, variants):
    """Replay identical captured Q/K/V/g/beta: exclude projection and convolution drift."""
    import torch

    report = {}
    for name, function in variants.items():

        def run(tokens, state=None, function=function):
            return function(
                tokens["query"],
                tokens["key"],
                tokens["value"],
                g=tokens["g"],
                beta=tokens["beta"],
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

        try:
            full, full_state = run(inputs)
            splits = {}
            for cut in cuts:
                before, state = run({k: v[:, :cut].contiguous() for k, v in inputs.items()})
                after, state = run({k: v[:, cut:].contiguous() for k, v in inputs.items()}, state)
                splits[str(cut)] = {
                    "output": tensor_delta(torch.cat([before, after], dim=1), full),
                    "state": tensor_delta(state, full_state),
                }
            report[name] = splits
        except Exception as exc:
            report[name] = {"error": str(exc)}
    return report


def split_prefill_diagnostics(model, device, prefix, suffix, generated, compare):
    import inspect

    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module

    # These alternatives run only after a failed gate. They neither change the
    # stock reference nor select a production kernel based on diagnostic results.
    variants = {
        "native_chunk": module.torch_chunk_gated_delta_rule,
        "native_recurrent": module.torch_recurrent_gated_delta_rule,
        "torch_fp32_chunk": inspect.unwrap(module.torch_chunk_gated_delta_rule),
    }
    cuts = sorted({len(prefix), len(prefix) // 64 * 64} - {0})
    report, inputs = prefill_trace(model, device, prefix, suffix, module)
    report["isolated_first_kernel"] = isolated_kernel_probes(inputs, cuts, variants)
    del inputs

    def replay(cut):
        ids, cache = prefix + suffix, None
        if cut:
            cache = model.model(
                input_ids=torch.tensor([ids[:cut]], device=device), use_cache=True
            ).past_key_values
        out = model.model(
            input_ids=torch.tensor([ids[cut:]], device=device),
            past_key_values=cache,
            use_cache=True,
        )
        for token in generated:
            out = model.model(
                input_ids=torch.tensor([[token]], device=device),
                past_key_values=out.past_key_values,
                use_cache=True,
            )
        return model.lm_head(out.last_hidden_state[0, -1]).float()

    native_full = replay(0)
    report["model_kernel_variants"] = {}
    for name, function in variants.items():
        try:
            with replace_chunk_kernel(module, function):
                full = replay(0)
                report["model_kernel_variants"][name] = {
                    "full_vs_unchanged_native_full": compare(full, native_full),
                    "split_vs_variant_full": {str(cut): compare(replay(cut), full) for cut in cuts},
                }
        except Exception as exc:
            report["model_kernel_variants"][name] = {"error": str(exc)}
    return report
