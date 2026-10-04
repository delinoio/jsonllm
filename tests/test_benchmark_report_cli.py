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
            summarize([row], 1) | {"model": "/private/local/path", "pid": 42},
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
    with gzip.open(output / "raw/whole_json.jsonl.gz", "rt") as stream:
        assert json.loads(stream.readline())["error"]["message"] == "fixture"
    public_summary = (output / "raw/whole_json.summary.json").read_text()
    assert "/private/local/path" not in public_summary and '"pid"' not in public_summary
    verify_artifacts(output, json.loads((output / "artifacts.json").read_text()))
