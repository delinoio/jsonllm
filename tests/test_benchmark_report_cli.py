import gzip
import json
import subprocess
import sys
from pathlib import Path

from jsonllm.artifacts import verify_artifacts
from jsonllm.benchmark_measure import summarize
from jsonllm.io import write_json, write_jsonl


def test_report_cli_retains_failures_and_verifies_archive(tmp_path):
    data, run, output = [tmp_path / name for name in ("data", "run", "report")]
    data.mkdir()
    write_json(data / "manifest.json", {"fixture": True})
    write_jsonl(
        data / "core.jsonl",
        [
            {
                "id": "one",
                "tokens": {
                    "context": 100,
                    "json_prompt": 150,
                    "json_reserved": 128,
                    "field_max_reserved": 200,
                    "answer_value_tokens": [3, 5],
                },
            }
        ],
    )
    run.mkdir()
    (run / "environment-cuda.txt").write_text(
        "torch==2.14.0\n-e file:///private/local/source\npackage @ https://private.invalid\n"
    )
    write_json(run / "base-gate.json", {"status": "passed"})
    write_json(run / "active-server.json", {"pid": 12345})
    write_json(run / "private-key.json", {"secret": "must-not-export"})
    (run / "gpu-environment.txt").write_text(
        "Driver Version : fixture-driver\nGPU UUID : private-id\nProduct Name : fixture-gpu\n"
    )
    invalid = run / "invalidated" / "fixture"
    invalid.mkdir(parents=True)
    write_json(invalid / "reason.json", {"reason": "fixture environment failure"})
    (invalid / "private.log").write_text("private original evidence")
    for method in ("whole_json", "shared_fields"):
        trial = run / "trials" / method
        trial.mkdir(parents=True)
        row = {
            "id": "one",
            "kind": "choice",
            "language": "en",
            "correct": False,
            "schema_valid": False,
            "graph_valid": None,
            "error": {"type": "ValueError", "message": "fixture"},
            "latency_seconds": 1,
            "first_usable_seconds": None,
            "dispatch_queue_seconds": 0,
            "metrics": {},
        }
        write_jsonl(trial / "raw.jsonl", [row])
        write_json(
            trial / "job.json",
            {
                "stage": "core",
                "group": "core",
                "model": "base",
                "method": method,
                "concurrency": 1,
                "repeat": 0,
            },
        )
        write_json(
            trial / "summary.json",
            summarize([row], 1)
            | {
                "model": "/private/local/path",
                "pid": 42,
                "device_after": "GPU-private-id, fixture-gpu, 8000 MiB, 0 %",
            },
        )
    script = Path(__file__).parents[1] / "scripts/report_design_benchmark.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--run",
            str(run),
            "--data",
            str(data),
            "--output",
            str(output),
            "--no-plots",
        ],
        check=True,
    )
    report = json.loads((output / "report.json").read_text())
    assert not report["paired_comparisons"][0]["point_quality_gate"]
    assert report["groups"][0]["first_usable_missing"] == 1
    assert report["groups"][0]["input_tokens"]["context"]["median"] == 100
    assert report["groups"][0]["reference_value_tokens"]["median"] == 4
    assert report["groups"][0]["valid_completion_fraction"] == 0
    assert "Same-model mechanism contrasts" in (output / "README.md").read_text()
    with gzip.open(output / "raw/whole_json.jsonl.gz", "rt") as stream:
        assert json.loads(stream.readline())["error"]["message"] == "fixture"
    public_summary = (output / "raw/whole_json.summary.json").read_text()
    assert "/private/local/path" not in public_summary and '"pid"' not in public_summary
    assert "GPU-private-id" not in public_summary and "fixture-gpu" in public_summary
    assert (output / "evidence/environment-cuda.txt").read_text() == "torch==2.14.0\n"
    assert json.loads((output / "evidence/base-gate.json").read_text())["status"] == "passed"
    invalidated = json.loads((output / "evidence/invalidated.json").read_text())
    assert invalidated[0]["retained_local_files"]["private.log"]["bytes"] > 0
    assert not list(output.rglob("private-key.json"))
    assert not list(output.rglob("active-server.json"))
    assert not list(output.rglob("private.log"))
    hardware = (output / "evidence/hardware.json").read_text()
    assert "fixture-driver" in hardware and "fixture-gpu" in hardware
    assert "private-id" not in hardware
    verify_artifacts(output, json.loads((output / "artifacts.json").read_text()))
