"""Freeze development evidence, inputs, source, and the full benchmark schedule."""

import argparse
import json
import subprocess
import time
from pathlib import Path

from jsonllm.artifacts import file_hash, verify_artifacts
from jsonllm.benchmark_engine import METHODS
from jsonllm.benchmark_schedule import development_timing, estimated_seconds, schedule
from jsonllm.io import read_jsonl, write_json


def build_seal(run, data, source_commit):
    manifest = json.loads((data / "manifest.json").read_text())
    verify_artifacts(data, {k + ".jsonl": v["sha256"] for k, v in manifest["groups"].items()})
    evidence, timings = {}, {}
    for model in ("base", "jsonllm"):
        gate = run / (model + "-gate.json")
        if json.loads(gate.read_text())["status"] != "passed":
            raise ValueError(f"CUDA correctness gate did not pass: {model}")
        evidence[gate.name] = file_hash(gate)
        for method in METHODS:
            trial = run / "trials" / f"dev-{model}-{method}"
            summary = json.loads((trial / "summary.json").read_text())
            if summary["revision"] != manifest["models"][model]["revision"]:
                raise ValueError("Development model revision mismatch")
            if summary["data_sha256"] != manifest["groups"]["dev"]["sha256"]:
                raise ValueError("Development data changed")
            rows = read_jsonl(trial / "raw.jsonl")
            if summary["status"] != "complete" or not any(row["schema_valid"] for row in rows):
                raise ValueError(f"No usable compatibility evidence: {model}/{method}")
            for name in ("job.json", "summary.json", "raw.jsonl", "warmup.jsonl"):
                path = trial / name
                evidence[str(path.relative_to(run))] = file_hash(path)
            server = run / f"dev-{model}-{method}-server.json"
            startup = 0
            if server.exists():
                evidence[server.name] = file_hash(server)
                startup = json.loads(server.read_text())["startup_seconds"]
            timings[(model, method)] = development_timing(summary, startup)
    jobs = schedule(manifest)
    estimates = [
        {
            "job": j["name"],
            "seconds": estimated_seconds(j, timings, manifest["groups"][j["group"]]["included"]),
        }
        for j in jobs
    ]
    for name in (
        "base-model-integrity.json",
        "jsonllm-model-integrity.json",
        "environment-cuda.txt",
        "environment-vllm.txt",
        "kernels.json",
    ):
        evidence[name] = file_hash(run / name)
    return {
        "created_at": time.time(),
        "source_commit": source_commit,
        "data_manifest_sha256": file_hash(data / "manifest.json"),
        "development_evidence_sha256": evidence,
        "models": manifest["models"],
        "planned_jobs": jobs,
        "estimated_jobs": estimates,
        "estimated_total_seconds": sum(e["seconds"] for e in estimates),
        "estimation_limits": "Development mean times with setup and safety margins; sweep "
        "costs can differ. A fixed external deadline and whole-block admission bound spending.",
        "quality_does_not_select_schedule": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    target = args.run / "seal.json"
    if target.exists():
        raise ValueError("Seal already exists; preserve it and audit any new campaign")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Commit all intended source changes before sealing")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    seal = build_seal(args.run, args.data, commit)
    write_json(target, seal)
    print(
        json.dumps(
            {
                "jobs": len(seal["planned_jobs"]),
                "estimated_hours": seal["estimated_total_seconds"] / 3600,
                "source_commit": commit,
                "seal_sha256": file_hash(target),
            }
        )
    )


if __name__ == "__main__":
    main()
