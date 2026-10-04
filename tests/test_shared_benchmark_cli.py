import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def script():
    spec = importlib.util.spec_from_file_location(
        "benchmark_shared_ui", Path(__file__).parents[1] / "scripts/benchmark_shared_ui.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("profile", [False, True])
def test_cli_passes_profile_flag_and_labels_noncomparable_run(tmp_path, monkeypatch, profile):
    torch = pytest.importorskip("torch")
    from jsonllm.backends.shared_cuda import SharedPredictor

    module = script()
    output = tmp_path / "trial"
    args = [
        "benchmark_shared_ui",
        "--model",
        "local",
        "--data",
        "unused",
        "--output",
        str(output),
    ] + (["--profile-stages"] if profile else [])
    monkeypatch.setattr(sys, "argv", args)
    received = {}

    def load(*args, **kwargs):
        received.update(kwargs)
        return SimpleNamespace(inference_dtype="float16")

    monkeypatch.setattr(SharedPredictor, "load", load)
    monkeypatch.setattr(module, "read_jsonl", lambda *a: [])
    monkeypatch.setattr(module, "GpuTelemetry", lambda *a: SimpleNamespace(close=lambda: {}))
    monkeypatch.setattr(module, "evaluate", lambda *a, **kw: ([], {"wall_seconds": 1}))
    monkeypatch.setattr(module, "summary", lambda *a, **kw: {})
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    module.main()
    result = json.loads((output / "summary.json").read_text())
    assert received["profile_stages"] == profile
    assert result["profile_stages"] == profile
    assert result["latency_comparable"] == (not profile)
    assert result["measurement_kind"] == ("stage_profile" if profile else "latency")


def test_vllm_rejects_unsupported_stage_profile_before_loading(tmp_path, monkeypatch):
    output = tmp_path / "trial"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_shared_ui",
            "--model",
            "unused",
            "--data",
            "unused",
            "--output",
            str(output),
            "--engine",
            "vllm",
            "--profile-stages",
        ],
    )
    with pytest.raises(SystemExit) as failure:
        script().main()
    assert failure.value.code == 2 and not output.exists()
