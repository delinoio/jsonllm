"""Opt-in host elapsed times and deferred CUDA events for shared inference."""

import time
from contextlib import contextmanager, nullcontext


class StageProfiler:
    def __init__(self, enabled, device, metrics, gpu_events):
        self.enabled, self.device = enabled, device
        self.metrics, self.gpu_events = metrics, gpu_events
        if enabled:
            metrics["profile_stages"] = True
            metrics["stage_host_seconds"] = {}
            for name in ("cache_reorder_count", "cache_reorder_bytes", "mask_transfer_bytes"):
                metrics[name] = 0

    def __call__(self, name, *, gpu=False):
        if not self.enabled:
            return nullcontext()
        return self._measure(name, gpu)

    def add(self, name, amount):
        if self.enabled:
            self.metrics[name] += amount

    @contextmanager
    def _measure(self, name, gpu):
        started = time.perf_counter()
        start, end = None, None
        if gpu and self.device.type == "cuda":
            import torch

            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        try:
            yield
        finally:
            if end is not None:
                # Resolving event times is deferred to the session's existing close sync.
                end.record()
                self.gpu_events.append((name, start, end))
            elapsed = time.perf_counter() - started
            stages = self.metrics["stage_host_seconds"]
            stages[name] = stages.get(name, 0.0) + elapsed
