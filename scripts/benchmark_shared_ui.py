"""Run one immutable model/engine trial; the coordinator interleaves five model orders."""

import argparse
import gc
import time
from pathlib import Path

from jsonllm.backends.shared_cuda import inference_dtype
from jsonllm.gpu_telemetry import GpuTelemetry
from jsonllm.io import read_jsonl, write_json, write_jsonl
from jsonllm.shared_benchmark import evaluate, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--engine", choices=["shared", "unshared", "vllm"], default="shared")
    parser.add_argument("--prompt-version", default="typed-value-v2")
    parser.add_argument(
        "--compile-mode", choices=["eager", "compile", "cudagraph"], default="eager"
    )
    parser.add_argument("--bucket-choices", action="store_true")
    parser.add_argument(
        "--profile-stages",
        action="store_true",
        help="Collect detailed stage timings in a separate, non-comparable profile run",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--rollout", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.profile_stages and args.engine == "vllm":
        parser.error("--profile-stages requires --engine shared or unshared")
    if args.output.exists():
        raise ValueError("Use a fresh trial directory")
    args.output.mkdir(parents=True)
    rows = read_jsonl(args.data)
    started = time.perf_counter()
    if args.engine == "vllm":
        from jsonllm.backends.ui_vllm import UIFieldPredictor
        from jsonllm.backends.vllm import Predictor

        client = Predictor(args.model, args.revision)
        predictor = UIFieldPredictor(client, args.prompt_version)
    else:
        import torch

        from jsonllm.backends.shared_cuda import SharedPredictor

        predictor = SharedPredictor.load(
            args.model,
            args.revision,
            share=args.engine == "shared",
            bucket_choices=args.bucket_choices,
            profile_stages=args.profile_stages,
        )
        if args.compile_mode != "eager":
            predictor.backbone = torch.compile(
                predictor.model.model,
                dynamic=True,
                mode="reduce-overhead" if args.compile_mode == "cudagraph" else "default",
            )
        torch.cuda.reset_peak_memory_stats()
    load_seconds = time.perf_counter() - started
    telemetry = GpuTelemetry(args.output)
    try:
        raw, timing = evaluate(
            rows,
            predictor,
            concurrency=args.concurrency,
            rollout=args.rollout,
            max_tokens=args.max_tokens,
        )
        write_jsonl(args.output / "raw.jsonl", raw)
        report = summary(raw, regression_passed=False, wall_seconds=timing["wall_seconds"])
        report.update(
            timing,
            load_seconds=load_seconds,
            engine=args.engine,
            model=args.model,
            compile_mode=args.compile_mode,
            profile_stages=args.profile_stages,
            measurement_kind="stage_profile" if args.profile_stages else "latency",
            latency_comparable=not args.profile_stages,
            inference_dtype=(
                inference_dtype(args.engine) if args.engine == "vllm" else predictor.inference_dtype
            ),
        )
        if args.engine != "vllm":
            report["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
        write_json(args.output / "summary.json", report)
        profile = {"requested_mode": args.compile_mode, "cuda_graph_launches": 0}
        if args.compile_mode == "cudagraph":
            # Profiling is outside every latency sample. A requested mode alone proves nothing.
            import json

            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                evaluate(rows[:10], predictor, warmup=False, max_tokens=args.max_tokens)
            trace = args.output / "cuda-profile.json"
            prof.export_chrome_trace(str(trace))
            events = json.loads(trace.read_text())["traceEvents"]
            profile["cuda_graph_launches"] = sum(
                "cudaGraphLaunch" in e.get("name", "") for e in events
            )
        write_json(args.output / "optimization.json", profile)
    finally:
        write_json(args.output / "device-memory.json", telemetry.close())
        if args.engine == "vllm":
            client.close()
        del predictor
        gc.collect()


if __name__ == "__main__":
    main()
