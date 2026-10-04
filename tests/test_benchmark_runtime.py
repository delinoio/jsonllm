import pytest

from jsonllm.benchmark_runtime import configure_eager_runtime


def test_runtime_rejects_late_or_conflicting_configuration(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(ValueError, match="Conflicting"):
        configure_eager_runtime()
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(ValueError, match="before CUDA"):
        configure_eager_runtime()


def test_runtime_enables_fixed_settings_before_loading(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    original = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    try:
        report = configure_eager_runtime()
        assert report["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        assert configure_eager_runtime() == report
    finally:
        torch.use_deterministic_algorithms(original[0])
        torch.backends.cudnn.deterministic = original[1]
        torch.backends.cudnn.benchmark = original[2]
