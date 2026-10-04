"""Fixed numerical settings for the design study, without changing public inference."""

import os

CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def configure_eager_runtime():
    """Set the accepted FP16 environment before a benchmark creates a CUDA context."""
    import torch

    previous = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if previous not in (None, CUBLAS_WORKSPACE_CONFIG):
        raise ValueError("Conflicting CUBLAS_WORKSPACE_CONFIG for the sealed benchmark")
    if torch.cuda.is_initialized() and previous is None:
        raise ValueError("Configure the benchmark environment before CUDA initialization")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG,
        "torch_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "scope": "eager PyTorch operations; third-party kernels retain their own behavior",
    }
