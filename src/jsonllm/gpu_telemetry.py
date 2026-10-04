"""Device-wide sampled memory, including allocations in a separate vLLM process."""

import csv
import subprocess


class GpuTelemetry:
    def __init__(self, output):
        self.summary = None
        self.path = output / "gpu-memory.csv"
        self.stream = self.path.open("w")
        self.errors = (output / "gpu-memory-errors.txt").open("w")
        self.process = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
                "--loop-ms=200",
            ],
            stdout=self.stream,
            stderr=self.errors,
        )

    def close(self):
        if self.summary is not None:
            return self.summary
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.stream.close()
        self.errors.close()
        self.summary = memory_summary(self.path)
        return self.summary


def memory_summary(path):
    samples = []
    with path.open() as stream:
        for row in csv.reader(stream):
            if len(row) != 6:
                continue
            try:
                samples.append(float(row[3]) * 1024**2)
            except ValueError:
                continue
    return {
        "sampled_device_peak_memory_bytes": int(max(samples)) if samples else None,
        "device_memory_samples": len(samples),
        "device_memory_interval_ms": 200,
        "device_memory_scope": "Resident model, warmup/compilation and measured trial; "
        "whole device including the vLLM server. Sampled peak, not allocator-exact.",
    }
