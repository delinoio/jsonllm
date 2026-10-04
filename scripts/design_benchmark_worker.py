"""Sequential benchmark orchestration with a fixed deadline and durable results."""

import argparse
import fcntl
import itertools
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from jsonllm.artifacts import file_hash, verify_artifacts
from jsonllm.benchmark_engine import METHODS
from jsonllm.benchmark_schedule import estimated_seconds, group_key, schedule
from jsonllm.io import write_json


class Worker:
    def __init__(self, args):
        self.args = args
        self.root = args.run_root
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / ".worker.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.manifest = json.loads((args.data / "manifest.json").read_text())
        verify_artifacts(
            args.data,
            {name + ".jsonl": spec["sha256"] for name, spec in self.manifest["groups"].items()},
        )
        self.server = None
        self.server_log = None
        self.server_model = None
        write_json(self.root / "worker-process.json", {"pid": os.getpid(), "pgid": os.getpgrp()})

    def state(self, status, **detail):
        write_json(
            self.root / "worker.json",
            {"status": status, "updated_at": time.time(), "pid": os.getpid(), **detail},
        )

    def model(self, name):
        return str(self.args.models / name), self.manifest["models"][name]["revision"]

    def stop_server(self):
        if self.server is not None:
            if self.server.poll() is None:
                os.killpg(self.server.pid, signal.SIGTERM)
                try:
                    self.server.wait(30)
                except subprocess.TimeoutExpired:
                    os.killpg(self.server.pid, signal.SIGKILL)
                    self.server.wait()
            self.server_log.close()
            self.server = self.server_model = None

    def ensure_server(self, model, name):
        if self.server_model == model and self.server is not None and self.server.poll() is None:
            return
        self.stop_server()
        path, revision = self.model(model)
        command = [
            str(self.args.vllm_python),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            path,
            "--revision",
            revision,
            "--served-model-name",
            "jsonllm",
            "--dtype",
            "float16",
            "--max-model-len",
            "2048",
            "--gpu-memory-utilization",
            "0.75",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--generation-config",
            "vllm",
            "--seed",
            "42",
        ]
        self.server_log = (self.root / (name + "-server.log")).open("w")
        started = time.time()
        self.server = subprocess.Popen(
            command, stdout=self.server_log, stderr=subprocess.STDOUT, start_new_session=True
        )
        write_json(self.root / "active-server.json", {"pid": self.server.pid})
        while time.time() < min(started + 900, self.args.deadline):
            if self.server.poll() is not None:
                raise RuntimeError("vLLM startup failed; inspect preserved server log")
            try:
                response = httpx.get("http://127.0.0.1:8000/health", timeout=3)
                if response.status_code == 200:
                    self.server_model = model
                    write_json(
                        self.root / (name + "-server.json"),
                        {"startup_seconds": time.time() - started, "command": command},
                    )
                    return
            except httpx.HTTPError:
                pass
            time.sleep(3)
        raise TimeoutError("vLLM startup deadline")

    def trial(self, job):
        self.state("running", stage=job["stage"], job=job)
        path, revision = self.model(job["model"])
        if job["method"] == "vllm_json":
            self.ensure_server(job["model"], job["name"])
        else:
            self.stop_server()
        output = self.root / "trials" / job["name"]
        command = [
            sys.executable,
            "scripts/run_design_benchmark.py",
            "--model",
            path,
            "--revision",
            revision,
            "--method",
            job["method"],
            "--data",
            str(self.args.data / (job["group"] + ".jsonl")),
            "--warmup-data",
            str(self.args.data / "dev.jsonl"),
            "--output",
            str(output),
            "--concurrency",
            str(job["concurrency"]),
        ]
        for key in ("limit", "arrival_rate"):
            if key in job:
                command += ["--" + key.replace("_", "-"), str(job[key])]
        with (self.root / (job["name"] + ".log")).open("w") as log:
            subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=max(1, self.args.deadline - time.time()),
            )
        summary = json.loads((output / "summary.json").read_text())
        write_json(output / "job.json", job)
        return summary

    def run(self):
        if self.args.development_only:
            for model in ("base", "jsonllm"):
                self.stop_server()
                path, revision = self.model(model)
                self.state("running", stage="cuda_gate", model=model)
                with (self.root / (model + "-gate.log")).open("w") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "scripts/verify_shared_cuda.py",
                            "--model",
                            path,
                            "--revision",
                            revision,
                            "--output",
                            str(self.root / (model + "-gate.json")),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=min(1800, max(1, self.args.deadline - time.time())),
                    )
                for method in METHODS:
                    self.trial(
                        {
                            "stage": "development",
                            "group": "dev",
                            "repeat": 0,
                            "model": model,
                            "method": method,
                            "concurrency": 1,
                            "limit": 16,
                            "name": f"dev-{model}-{method}",
                        }
                    )
            self.stop_server()
            self.state("awaiting_seal", stage="development_complete")
            return
        seal = json.loads((self.root / "seal.json").read_text())
        if file_hash(self.args.data / "manifest.json") != seal["data_manifest_sha256"]:
            raise ValueError("Sealed data changed")
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if actual != seal["source_commit"]:
            raise ValueError("Sealed source changed")
        jobs = schedule(self.manifest)
        write_json(self.root / "schedule.json", jobs)
        timings = {}
        for model in ("base", "jsonllm"):
            for method in METHODS:
                s = json.loads(
                    (self.root / "trials" / f"dev-{model}-{method}" / "summary.json").read_text()
                )
                timings[(model, method)] = s["wall_seconds"] / s["records"]
        omitted = []
        for _, grouped in itertools.groupby(jobs, key=group_key):
            group = list(grouped)
            estimate = sum(
                estimated_seconds(j, timings, self.manifest["groups"][j["group"]]["included"])
                for j in group
            )
            if time.time() + estimate >= self.args.deadline:
                omitted.extend(
                    {**j, "reason": "insufficient_budget_for_complete_comparison"} for j in group
                )
                continue
            for job in group:
                output = self.root / "trials" / job["name"]
                if output.exists():
                    raise ValueError(
                        "Trial output already exists; explicit audited recovery required"
                    )
                self.trial(job)
        self.stop_server()
        write_json(self.root / "omitted.json", omitted)
        self.state("complete", stage="measurements_complete", omitted=len(omitted))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--models", required=True, type=Path)
    p.add_argument("--deadline", required=True, type=float)
    p.add_argument("--vllm-python", default=".venv-vllm/bin/python", type=Path)
    p.add_argument("--development-only", action="store_true")
    worker = Worker(p.parse_args())
    try:
        worker.run()
    except Exception as exc:
        worker.state("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        worker.stop_server()


if __name__ == "__main__":
    main()
