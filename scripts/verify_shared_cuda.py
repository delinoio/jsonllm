"""Actual-weight CUDA gate before training; abort on a state or output disagreement."""

import argparse
import gc
import importlib.metadata
import inspect
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from pathlib import Path

from jsonllm.backends.shared_cuda import SharedPredictor, candidate_logits
from jsonllm.io import write_json


def logit_comparison(actual, expected):
    import torch

    actual, expected = actual.float(), expected.float()
    close = torch.isclose(actual, expected, atol=0.0625, rtol=0.01)
    finite = torch.isfinite(actual).all() & torch.isfinite(expected).all()
    return {
        "elements": actual.numel(),
        "mismatched_elements": (~close).sum().item(),
        "max_absolute_delta": (actual - expected).abs().max().item() if finite else None,
        "finite": finite.item(),
        "argmax_equal": torch.equal(actual.argmax(-1), expected.argmax(-1)),
        "actual_argmax": actual.argmax(-1).tolist(),
        "expected_argmax": expected.argmax(-1).tolist(),
    }


class LogitGate:
    """Persist the first exact failing probe without changing the acceptance criteria."""

    def __init__(self, output=None):
        self.output = output
        self.report = {"status": "running", "comparisons": []}

    def save(self):
        if self.output is not None:
            write_json(self.output, self.report)

    def check(self, actual, expected, *, case, kind, diagnose=None):
        comparison = logit_comparison(actual, expected)
        entry = {"case": case, "kind": kind, **comparison}
        self.report["comparisons"].append(entry)
        if (
            not comparison["finite"]
            or comparison["mismatched_elements"]
            or not comparison["argmax_equal"]
        ):
            self.report["status"] = "failed"
            self.report["failure"] = entry
            self.save()
            if diagnose is not None:
                try:
                    entry["diagnostics"] = diagnose()
                except Exception as exc:
                    entry["diagnostic_error"] = str(exc)
            self.save()
            raise AssertionError(f"{kind} logits failed the fixed BF16 gate: {comparison}")


def diagnose_generation(predictor, prefix, suffix, generated, actual, expected):
    """Separate ordinary cached-decode drift from prefix-fork drift on identical tokens."""
    import torch

    model, device = predictor.model, predictor.device
    out = model.model(input_ids=torch.tensor([prefix + suffix], device=device), use_cache=True)
    for token in generated:
        out = model.model(
            input_ids=torch.tensor([[token]], device=device),
            past_key_values=out.past_key_values,
            use_cache=True,
        )
    baseline = model.lm_head(out.last_hidden_state[0, -1]).float()
    report = {
        "prefix_ids": prefix,
        "suffix_ids": suffix,
        "generated_ids": generated,
        "shared_vs_ordinary_cached": logit_comparison(actual, baseline),
        "ordinary_cached_vs_full_recompute": logit_comparison(baseline, expected),
        "cache_layers": [
            {
                name: {
                    str(k): {"shape": list(v.shape), "dtype": str(v.dtype)}
                    for k, v in getattr(layer, name, {}).items()
                    if v is not None
                }
                for name in ("conv_states", "recurrent_states")
            }
            for layer in out.past_key_values.layers
        ],
    }
    del out
    report["numerical_probes"] = numerical_probes(
        model, device, prefix, suffix, generated, actual, expected
    )
    from jsonllm.backends.shared_diagnostics import split_prefill_diagnostics

    try:
        report["split_prefill_diagnostics"] = split_prefill_diagnostics(
            model, device, prefix, suffix, generated, logit_comparison
        )
    except Exception as exc:
        report["split_prefill_diagnostic_error"] = str(exc)
    return report


@contextmanager
def numerical_settings(model, *, reduced_precision=None, cudnn=None, attention=None):
    """Temporary diagnostic settings must never leak into the accepted runtime."""
    import torch

    old_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    old_cudnn = torch.backends.cudnn.enabled
    old_attention = model.config._attn_implementation
    try:
        if reduced_precision is not None:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced_precision
        if cudnn is not None:
            torch.backends.cudnn.enabled = cudnn
        if attention is not None:
            model.set_attn_implementation(attention)
        yield
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_reduction
        torch.backends.cudnn.enabled = old_cudnn
        if attention is not None:
            model.set_attn_implementation(old_attention)


def numerical_probes(model, device, prefix, suffix, generated, actual, expected):
    """Bounded investigation of a failed case only; no automatic repair or gate bypass."""
    import torch

    def replay(split):
        cache = None
        ids = prefix + suffix
        if split:
            cache = model.model(
                input_ids=torch.tensor([prefix], device=device), use_cache=True
            ).past_key_values
            ids = suffix
        out = model.model(
            input_ids=torch.tensor([ids], device=device), past_key_values=cache, use_cache=True
        )
        for token in generated:
            out = model.model(
                input_ids=torch.tensor([[token]], device=device),
                past_key_values=out.past_key_values,
                use_cache=True,
            )
        return model.lm_head(out.last_hidden_state[0, -1]).float()

    variants = {
        "current": {},
        "full_precision_reduction": {"reduced_precision": False},
        "eager_attention": {"attention": "eager"},
        "cudnn_disabled": {"cudnn": False},
        "combined": {"reduced_precision": False, "attention": "eager", "cudnn": False},
    }
    report = {}
    for name, settings in variants.items():
        try:
            with numerical_settings(model, **settings):
                full, split = replay(False), replay(True)
                report[name] = {
                    "settings": settings,
                    "independent_split_vs_full_prompt_cache": logit_comparison(split, full),
                    "original_shared_vs_independent_split": logit_comparison(actual, split),
                    "full_prompt_vs_original_recompute": logit_comparison(full, expected),
                }
        except Exception as exc:
            report[name] = {"settings": settings, "error": str(exc)}
    return report


def generation_check(gate, actual, cached, recomputed, *, case, diagnose=None):
    """Compare equivalent decode paths, also requiring exact recomputed greedy output."""
    observation = {"case": case, **logit_comparison(cached, recomputed)}
    gate.report.setdefault("cached_vs_full_recompute", []).append(observation)
    # A native cached decode differs numerically from a new chunk prefill even with
    # sharing disabled. Keep the same logit gate against an independent normal cache;
    # never excuse a changed token or non-finite full-recompute result as rounding.
    gate.check(actual, cached, case=case, kind="generation_cached", diagnose=diagnose)
    if case["generation_step"] == 0:
        gate.check(actual, recomputed, case=case, kind="generation_prefill", diagnose=diagnose)
    elif not observation["finite"] or not logit_comparison(actual, recomputed)["argmax_equal"]:
        gate.report.update(status="failed", failure={"kind": "generation_recompute", **observation})
        gate.save()
        raise AssertionError("Greedy generation diverged from full recomputation")


def kernel_implementation(function):
    """Read the pinned Transformers fallback closure; do not silently benchmark Torch fallback."""
    values = inspect.getclosurevars(function).nonlocals
    implementation = values.get("implementation")
    if implementation is not None:
        return implementation.__module__ + "." + implementation.__name__
    wrapped = getattr(function, "__wrapped__", None)
    return kernel_implementation(wrapped) if wrapped is not None else function.__module__


def verify(predictor, gate=None, *, diagnose_failures=True):
    import torch

    device = predictor.device
    if device.type != "cuda":
        raise ValueError("This gate requires the actual CUDA model")
    gate = gate or LogitGate()
    torch.manual_seed(42)
    reports = []
    labels = [predictor.tokenizer.encode(c, add_special_tokens=False)[0] for c in "ABCDEFGHI"]
    for length in [127, 527, 528, 529, 1023]:
        for count in [1, 4, 8]:
            case = {"prefix_tokens": length, "questions": count}
            prefix = torch.randint(50, 20000, (length,), device="cpu").tolist()
            suffixes = [
                torch.randint(50, 20000, (5 + row * 7,), device="cpu").tolist()
                for row in range(count)
            ]
            session = predictor.open_record("verification only", count)
            session.prefix = prefix
            try:
                with torch.inference_mode():
                    hidden, branch = session._batch(suffixes, selection=True)
                    del branch
                    logits = candidate_logits(hidden, predictor.model.lm_head.weight, labels)
                    full = torch.stack(
                        [
                            predictor.model.model(
                                input_ids=torch.tensor([prefix + suffix], device=device),
                                use_cache=False,
                            ).last_hidden_state[0, -1]
                            for suffix in suffixes
                        ]
                    )
                    reference = candidate_logits(full, predictor.model.lm_head.weight, labels)
                    # Fixed BF16 numerical tolerance; choice equality is independently mandatory.
                    gate.check(logits, reference, case=case, kind="selection")
                    max_delta = (logits - reference).abs().max().item()
                    # A later dependency branch must start from the untouched common state.
                    dependent = suffixes[0] + [101, 102, 103]
                    later, branch = session._batch([dependent], selection=True)
                    del branch
                    ref = predictor.model.model(
                        input_ids=torch.tensor([prefix + dependent], device=device), use_cache=False
                    ).last_hidden_state[0, -1]
                    gate.check(
                        candidate_logits(later[0], predictor.model.lm_head.weight, labels),
                        candidate_logits(ref, predictor.model.lm_head.weight, labels),
                        case=case,
                        kind="dependency",
                    )
                    # Exact four-token greedy generation, comparing full vocabulary logits.
                    generated = []
                    one, cache = session._batch([suffixes[0]], selection=False)
                    # Independent full-prompt cache: no SharedSession or fork_cache calls.
                    ordinary = predictor.model.model(
                        input_ids=torch.tensor([prefix + suffixes[0]], device=device),
                        use_cache=True,
                    )
                    for step in range(4):
                        shared_logits = predictor.model.lm_head(one[0]).float()
                        ordinary_logits = predictor.model.lm_head(
                            ordinary.last_hidden_state[0, -1]
                        ).float()
                        all_ids = prefix + suffixes[0] + generated
                        ref_hidden = predictor.model.model(
                            input_ids=torch.tensor([all_ids], device=device), use_cache=False
                        ).last_hidden_state[0, -1]
                        ref_logits = predictor.model.lm_head(ref_hidden).float()
                        generation_check(
                            gate,
                            shared_logits,
                            ordinary_logits,
                            ref_logits,
                            case=case | {"generation_step": step},
                            diagnose=partial(
                                diagnose_generation,
                                predictor,
                                prefix,
                                suffixes[0],
                                generated,
                                shared_logits,
                                ref_logits,
                            )
                            if diagnose_failures
                            else None,
                        )
                        token = shared_logits.argmax().item()
                        assert token == ref_logits.argmax().item(), "Greedy generation diverged"
                        generated.append(token)
                        out = session._forward(
                            torch.tensor([[token]], device=device), cache, kind="decode"
                        )
                        cache, one = out.past_key_values, out.last_hidden_state[:, -1]
                        ordinary = predictor.model.model(
                            input_ids=torch.tensor([[token]], device=device),
                            past_key_values=ordinary.past_key_values,
                            use_cache=True,
                        )
                    del cache, out, one, hidden, full, logits, reference, later, ref, ref_hidden
                    del ordinary, ordinary_logits
                if count > 1:
                    assert session.metrics["common_prefills"] == 1
                    assert session.metrics["common_token_evaluations"] == length
                    assert session.metrics["forwards"][1]["batch"] == count
            finally:
                session.close()
            reports.append(
                {
                    "prefix_tokens": length,
                    "questions": count,
                    "max_candidate_logit_delta": max_delta,
                    "generated": generated,
                    "metrics": session.metrics,
                }
            )

    # Concurrent callers own separate cache objects; this implementation queues GPU records.
    def request(text):
        session = predictor.open_record(text, 4)
        try:
            with torch.inference_mode():
                output, cache = session._batch([[20, 21], [30, 31]], selection=True)
                result = output.float().cpu()
                del cache
                return result
        finally:
            session.close()

    expected = [request(text) for text in ["First record", "Unrelated second record"]]
    with ThreadPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(request, ["First record", "Unrelated second record"]))
    for first, second in zip(expected, actual, strict=True):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    session = predictor.open_record("cancel this record", 4)
    try:
        with torch.inference_mode():
            session._prefill()
            session.cancel()
            try:
                session._batch([[20, 21]], selection=True)
            except ValueError as exc:
                assert str(exc) == "cancelled"
            else:
                raise AssertionError("Cancellation was ignored")
    finally:
        session.close()
    gc.collect()
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    assert after <= before + 1024 * 1024, "Request memory did not return"
    return {
        "status": "passed",
        "probes": reports,
        "concurrent_records_isolated": True,
        "cancel_memory_before": before,
        "cancel_memory_after": after,
        "bf16_tolerance": {"atol": 0.0625, "rtol": 0.01},
        "generation_logit_reference": "independent full-prompt ordinary cached decode",
        "full_recompute_greedy_equality_required": True,
        "full_recompute_logit_scope": "selection, dependencies and initial generation prefill",
        "choices_and_generation_equal": True,
        "inference_dtype": predictor.inference_dtype,
    }


def precision_diagnostics(predictor):
    """Run the complete gate at higher mantissa precision, without accepting the failed run."""
    import torch

    from jsonllm.backends.shared_diagnostics import model_precision

    results = {}
    for name, dtype in (("float16", torch.float16), ("float32", torch.float32)):
        probe_gate = LogitGate()
        try:
            with model_precision(predictor.model, dtype):
                invalid = [
                    name
                    for name, parameter in predictor.model.named_parameters()
                    if not torch.isfinite(parameter).all()
                ]
                if invalid:
                    raise ValueError(f"Precision conversion produced nonfinite weights: {invalid}")
                # A fresh predictor owns its own lock and cache; no state is reused
                # from the failed session or from another precision experiment.
                probe = SharedPredictor(predictor.model, predictor.tokenizer)
                probe_gate.report.update(verify(probe, probe_gate, diagnose_failures=False))
        except Exception as exc:
            probe_gate.report.update(status="failed", error=str(exc))
        results[name] = probe_gate.report
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate = LogitGate(args.output)
    gate.report["versions"] = {
        p: importlib.metadata.version(p)
        for p in ["torch", "transformers", "flash-linear-attention", "xgrammar"]
    }
    predictor = SharedPredictor.load(args.model, args.revision)
    gate.report["inference_dtype"] = predictor.inference_dtype
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module

    names = [
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    ]
    kernels = {name: kernel_implementation(getattr(module, name)) for name in names}
    write_json(args.output.with_name("kernels.json"), kernels)
    gate.report["kernels"] = kernels
    gate.save()
    if not all("fla" in kernels[n] for n in names[:2]):
        raise ValueError("Optimized FLA kernels are not active")
    try:
        report = verify(predictor, gate)
    except Exception as exc:
        gate.report.update(status="failed", error=str(exc))
        gate.save()
        # The original failure is durable and remains fatal. These independent
        # precision runs supply evidence for a future runtime decision only.
        gate.report["precision_diagnostics"] = precision_diagnostics(predictor)
        gate.save()
        raise
    gate.report.update(report)
    gate.save()


if __name__ == "__main__":
    main()
