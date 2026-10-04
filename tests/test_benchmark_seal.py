import importlib.util
from pathlib import Path

import pytest

from jsonllm.artifacts import file_hash, verify_artifacts
from jsonllm.benchmark_engine import METHODS
from jsonllm.io import write_json, write_jsonl


def test_seal_binds_development_data_and_evidence(tmp_path):
    path = Path(__file__).parents[1] / "scripts/seal_design_benchmark.py"
    spec = importlib.util.spec_from_file_location("seal", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run, data = tmp_path / "run", tmp_path / "data"
    run.mkdir()
    data.mkdir()
    write_jsonl(data / "dev.jsonl", [{"id": "fixture"}])
    models = {name: {"revision": name + "-revision"} for name in ("base", "jsonllm")}
    groups = {"dev": {"sha256": file_hash(data / "dev.jsonl"), "included": 1}}
    for name in ("core", "tree", "workflow", "application"):
        (data / (name + ".jsonl")).write_text((data / "dev.jsonl").read_text())
        groups[name] = dict(groups["dev"])
    write_json(data / "manifest.json", {"models": models, "groups": groups})
    for model in models:
        write_json(run / (model + "-gate.json"), {"status": "passed"})
        write_json(run / (model + "-model-integrity.json"), {"revision": models[model]["revision"]})
        for method in METHODS:
            trial = run / "trials" / f"dev-{model}-{method}"
            trial.mkdir(parents=True)
            write_json(
                trial / "summary.json",
                {
                    "revision": models[model]["revision"],
                    "data_sha256": groups["dev"]["sha256"],
                    "status": "complete",
                    "records": 1,
                    "wall_seconds": 1,
                },
            )
            write_jsonl(trial / "raw.jsonl", [{"schema_valid": True, "correct": False}])
            write_jsonl(trial / "warmup.jsonl", [])
            write_json(trial / "job.json", {"method": method})
    for name in ("environment-cuda.txt", "environment-vllm.txt", "kernels.json"):
        (run / name).write_text("fixture")
    seal = module.build_seal(run, data, "source-fixture")
    assert seal["quality_does_not_select_schedule"]
    assert len(seal["planned_jobs"]) == 108
    verify_artifacts(run, seal["development_evidence_sha256"])
    write_json(run / "base-gate.json", {"status": "failed"})
    with pytest.raises((ValueError, AssertionError)):
        verify_artifacts(run, seal["development_evidence_sha256"])
    with pytest.raises(ValueError, match="correctness gate"):
        module.build_seal(run, data, "source-fixture")
