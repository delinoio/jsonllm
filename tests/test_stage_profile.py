from types import SimpleNamespace

import pytest

from jsonllm.backends.stage_profile import StageProfiler


def test_disabled_profile_has_no_timing_or_metric_overhead(monkeypatch):
    monkeypatch.setattr(
        "jsonllm.backends.stage_profile.time.perf_counter",
        lambda: pytest.fail("Disabled profile read clock"),
    )
    metrics, events = {}, []
    profile = StageProfiler(False, SimpleNamespace(type="cuda"), metrics, events)
    with profile("lm_head", gpu=True):
        pass
    profile.add("mask_transfer_bytes", 12)
    assert not metrics and not events


def test_cuda_profile_defers_sync_and_retains_failed_stage(monkeypatch):
    torch = pytest.importorskip("torch")
    records = []

    class Event:
        def __init__(self, **kwargs):
            assert kwargs == {"enable_timing": True}

        def record(self):
            records.append(self)

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: pytest.fail("Per-stage sync"))
    metrics, events = {}, []
    profile = StageProfiler(True, SimpleNamespace(type="cuda"), metrics, events)
    with pytest.raises(ValueError, match="failure"), profile("lm_head", gpu=True):
        raise ValueError("failure")
    assert events == [("lm_head", *records)]
    assert metrics["stage_host_seconds"]["lm_head"] >= 0
    with profile("grammar_fill"):
        profile.add("mask_transfer_bytes", 8)
    assert len(records) == 2
    assert metrics["mask_transfer_bytes"] == 8
    assert set(metrics["stage_host_seconds"]) == {"lm_head", "grammar_fill"}
