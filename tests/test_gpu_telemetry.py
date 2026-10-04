from jsonllm.gpu_telemetry import memory_summary


def test_device_peak_includes_server_memory_and_identifies_sampling(tmp_path):
    path = tmp_path / "gpu-memory.csv"
    path.write_text(
        "2026/09/30 01:01:01.002, 0, NVIDIA H100, 32768, 81920, 10\n"
        "2026/09/30 01:01:01.202, 0, NVIDIA H100, 65536, 81920, 90\n"
        "2026/09/30 01:01:01.402, 0, NVIDIA H100, [Not Supported], 81920, 10\n"
    )
    result = memory_summary(path)
    assert result["sampled_device_peak_memory_bytes"] == 64 * 1024**3
    assert result["device_memory_samples"] == 2
    assert result["device_memory_interval_ms"] == 200
