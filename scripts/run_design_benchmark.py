"""Run one immutable design benchmark trial; no training or paid API calls."""

import argparse
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path

from jsonllm.artifacts import file_hash
from jsonllm.benchmark_engine import METHODS, VLLMJSON, WholeJSON
from jsonllm.benchmark_measure import evaluate, measure
from jsonllm.benchmark_prepare import prepare
from jsonllm.gpu_telemetry import GpuTelemetry
from jsonllm.io import read_jsonl, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--warmup-data", type=Path)
    parser.add_argument("--method", choices=METHODS, default="shared_fields")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--arrival-rate", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    from jsonllm.training import load_tokenizer

    if args.prepare:
        prepare(args.output, load_tokenizer(args.model, args.revision))
        return
    if args.data is None or args.warmup_data is None:
        parser.error("--data and --warmup-data required for a trial")
    if args.concurrency not in (1, 2, 4, 8, 128):
        parser.error("Unsupported concurrency")
    args.output.mkdir(parents=True, exist_ok=False)
    cases = read_jsonl(args.data)
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        write_json(
            args.output / "summary.json", {"status": "excluded", "reason": "no eligible inputs"}
        )
        return
    loaded = time.perf_counter()
    telemetry = GpuTelemetry(args.output)
    if args.method == "vllm_json":
        backend = VLLMJSON(load_tokenizer(args.model, args.revision), args.url)
    else:
        from jsonllm.backends.shared_cuda import SharedPredictor

        predictor = SharedPredictor.load(
            args.model, args.revision, share=args.method == "shared_fields", bucket_choices=True
        )
        backend = WholeJSON(predictor) if args.method == "whole_json" else predictor
    load_seconds = time.perf_counter() - loaded
    warm = time.perf_counter()
    warmup = [measure(case, args.method, backend) for case in read_jsonl(args.warmup_data)[:8]]
    warmup_seconds = time.perf_counter() - warm
    write_jsonl(args.output / "warmup.jsonl", warmup)
    if args.method != "vllm_json":
        import torch

        torch.cuda.reset_peak_memory_stats()
    rows, summary = evaluate(
        cases, args.method, backend, concurrency=args.concurrency, arrival_rate=args.arrival_rate
    )
    write_jsonl(args.output / "raw.jsonl", rows)
    summary.update(telemetry.close())
    summary.update(
        status="complete",
        model=args.model,
        revision=args.revision,
        method=args.method,
        data_sha256=file_hash(args.data),
        concurrency=args.concurrency,
        arrival_rate=args.arrival_rate,
        load_seconds=load_seconds,
        warmup_seconds=warmup_seconds,
        inference_dtype="float16",
        pid=os.getpid(),
        precision_note="Stored weights BF16; inference weights and buffers FP16",
    )
    if args.method != "vllm_json":
        summary["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        summary["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    else:
        backend.close()
    device = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    summary["device_after"] = device.stdout.strip()
    summary["versions"] = {}
    for package in ("torch", "transformers", "xgrammar", "vllm", "flash-linear-attention"):
        try:
            summary["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    write_json(args.output / "summary.json", summary)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("status", "method", "records", "accuracy", "wall_seconds", "errors")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
