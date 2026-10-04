import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location(
    "verify_shared_cuda", Path(__file__).parents[1] / "scripts/verify_shared_cuda.py"
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [([1.0, 0.2], [1.0, 0.0]), ([0.01, 0.0], [0.0, 0.01]), ([float("nan")], [0.0])],
)
def test_gate_preserves_tolerance_choice_and_finiteness_failures(tmp_path, actual, expected):
    output = tmp_path / "gate.json"
    gate = verifier.LogitGate(output)

    def diagnose():
        # A crash during diagnosis must leave the original failure durable.
        assert json.loads(output.read_text())["status"] == "failed"
        raise RuntimeError("diagnosis interrupted")

    with pytest.raises(AssertionError, match="fixed BF16 gate"):
        gate.check(
            torch.tensor(actual),
            torch.tensor(expected),
            case={"prefix_tokens": 127, "generation_step": 1},
            kind="generation",
            diagnose=diagnose,
        )
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["failure"]["case"]["generation_step"] == 1
    assert report["failure"]["diagnostic_error"] == "diagnosis interrupted"


def test_gate_keeps_successful_comparisons_before_failure(tmp_path):
    gate = verifier.LogitGate(tmp_path / "gate.json")
    gate.check(torch.tensor([1.0, 0.02]), torch.tensor([1.0, 0.0]), case={}, kind="selection")
    with pytest.raises(AssertionError):
        gate.check(
            torch.tensor([1.0, 0.2]),
            torch.tensor([1.0, 0.0]),
            case={},
            kind="generation",
            diagnose=lambda: {"ordinary_cached_vs_full_recompute": "also fails"},
        )
    report = json.loads(gate.output.read_text())
    assert len(report["comparisons"]) == 2
    assert report["failure"]["diagnostics"]["ordinary_cached_vs_full_recompute"] == "also fails"


def test_decode_oracle_separates_native_drift_without_accepting_shared_corruption(tmp_path):
    gate = verifier.LogitGate(tmp_path / "gate.json")
    # Mirrors H100 evidence: sharing off == native cache, while a fresh chunk
    # recompute has a 0.1875 non-winning-logit difference and the same winner.
    cached, recomputed = torch.tensor([2.0, 0.1875]), torch.tensor([2.0, 0.0])
    verifier.generation_check(gate, cached.clone(), cached, recomputed, case={"generation_step": 1})
    assert gate.report["cached_vs_full_recompute"][0]["mismatched_elements"] == 1
    with pytest.raises(AssertionError, match="fixed BF16 gate"):
        verifier.generation_check(
            gate,
            cached + torch.tensor([0.0, 0.2]),
            cached,
            recomputed,
            case={"generation_step": 1},
        )


@pytest.mark.parametrize("step", [0, 1])
def test_generation_still_rejects_changed_full_recompute_winner(tmp_path, step):
    gate = verifier.LogitGate(tmp_path / "gate.json")
    cached, recomputed = torch.tensor([2.0, 1.0]), torch.tensor([1.0, 2.0])
    with pytest.raises(AssertionError):
        verifier.generation_check(
            gate, cached, cached.clone(), recomputed, case={"generation_step": step}
        )
    assert json.loads(gate.output.read_text())["status"] == "failed"


def test_initial_generation_retains_full_recompute_logit_gate(tmp_path):
    gate = verifier.LogitGate(tmp_path / "gate.json")
    with pytest.raises(AssertionError, match="fixed BF16 gate"):
        verifier.generation_check(
            gate,
            torch.tensor([2.0, 0.1875]),
            torch.tensor([2.0, 0.1875]),
            torch.tensor([2.0, 0.0]),
            case={"generation_step": 0},
        )


def test_diagnostic_settings_restore_after_failed_probe(monkeypatch):
    matmul = SimpleNamespace(allow_bf16_reduced_precision_reduction=True)
    cudnn = SimpleNamespace(enabled=True)
    monkeypatch.setattr(torch.backends.cuda, "matmul", matmul)
    monkeypatch.setattr(torch.backends, "cudnn", cudnn)
    config = SimpleNamespace(_attn_implementation="sdpa")
    model = SimpleNamespace(
        config=config,
        set_attn_implementation=lambda name: setattr(config, "_attn_implementation", name),
    )
    with pytest.raises(RuntimeError, match="probe failed"):
        with verifier.numerical_settings(
            model, reduced_precision=False, cudnn=False, attention="eager"
        ):
            assert not matmul.allow_bf16_reduced_precision_reduction
            assert not cudnn.enabled
            assert config._attn_implementation == "eager"
            raise RuntimeError("probe failed")
    assert matmul.allow_bf16_reduced_precision_reduction
    assert cudnn.enabled
    assert config._attn_implementation == "sdpa"


def test_diagnostic_success_cannot_override_failed_gate(tmp_path):
    gate = verifier.LogitGate(tmp_path / "gate.json")
    with pytest.raises(AssertionError, match="fixed BF16 gate"):
        gate.check(
            torch.tensor([2.0, 0.2]),
            torch.tensor([2.0, 0.0]),
            case={},
            kind="generation",
            diagnose=lambda: {"numerical_probes": {"combined": {"mismatched_elements": 0}}},
        )
    report = json.loads(gate.output.read_text())
    assert report["status"] == "failed"
    assert report["failure"]["mismatched_elements"] == 1


def test_precision_probes_run_full_gate_without_recursive_diagnostics(monkeypatch):
    model = torch.nn.Linear(2, 2).to(torch.bfloat16)
    model.model = torch.nn.Identity()
    predictor = SimpleNamespace(model=model, tokenizer=object())
    original = model.weight.detach().clone()
    dtypes = []

    def verify(probe, gate, *, diagnose_failures):
        assert not diagnose_failures
        assert probe.model is model
        dtypes.append(model.weight.dtype)
        if model.weight.dtype == torch.float16:
            raise AssertionError("precision probe failed")
        return {"status": "passed", "probes": list(range(15))}

    monkeypatch.setattr(verifier, "verify", verify)
    report = verifier.precision_diagnostics(predictor)
    assert dtypes == [torch.float16, torch.float32]
    assert report["float16"]["status"] == "failed"
    assert report["float32"]["status"] == "passed"
    assert len(report["float32"]["probes"]) == 15
    assert model.weight.dtype == torch.bfloat16
    assert torch.equal(model.weight, original)
