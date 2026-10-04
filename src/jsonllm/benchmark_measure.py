"""Application-boundary timings and exact scoring, including failed requests."""

import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from .benchmark_data import assemble_topology, inference_record
from .benchmark_engine import field_inference, validate_object
from .io import dumps
from .metrics import distribution


def measure(case, method, backend, *, scheduled=None, timeout=120):
    entered = time.perf_counter()
    scheduled = entered if scheduled is None else scheduled
    metrics, output, error = {}, None, None
    graph_valid = None
    try:
        if entered - scheduled >= timeout:
            raise TimeoutError("expired_in_queue")
        record = inference_record(case)
        if method in {"whole_json", "vllm_json"}:
            output, metrics = backend.infer(record, case["json_max_tokens"], scheduled + timeout)
        else:
            output, metrics = field_inference(
                record,
                backend,
                serial=method == "serial_fields",
                deadline=scheduled + timeout,
            )
        validate_object(record, output)
        dumps(output)
        if case["kind"] in {"tree", "workflow"}:
            try:
                graph = assemble_topology(output, case["kind"])
                graph_valid = True
                metrics["assembled"] = graph
            except ValueError:
                graph_valid = False
        if time.perf_counter() - scheduled > timeout:
            raise TimeoutError("request_timeout")
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[:500]}
    ended = time.perf_counter()
    valid = output is not None and error is None
    correct = valid and output == case["answers"] and graph_valid is not False
    latency = ended - scheduled
    first = metrics.get("first_usable_seconds")
    return {
        "id": case["id"],
        "kind": case["kind"],
        "language": case["language"],
        "method": method,
        "latency_seconds": latency,
        "service_seconds": ended - entered,
        "dispatch_queue_seconds": entered - scheduled,
        "first_usable_seconds": (first + entered - scheduled)
        if first is not None
        else (latency if valid else None),
        "schema_valid": valid,
        "correct": correct,
        "graph_valid": graph_valid,
        "output": output,
        "error": error,
        "metrics": metrics,
        "condition": case["condition"],
    }


def evaluate(cases, method, backend, *, concurrency=1, arrival_rate=None, timeout=120):
    started = time.perf_counter()
    # Finite offered load: every scheduled request is retained, including expired queued work.
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        for i, case in enumerate(cases):
            target = started + i / arrival_rate if arrival_rate else None
            if target is not None:
                time.sleep(max(0, target - time.perf_counter()))
            futures.append(
                pool.submit(measure, case, method, backend, scheduled=target, timeout=timeout)
            )
        rows = [f.result() for f in futures]
    wall = time.perf_counter() - started
    return rows, summarize(rows, wall)


def wilson(successes, count):
    if not count:
        return [None, None]
    z, p = 1.959963984540054, successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0, center - radius), min(1, center + radius)]


def summarize(rows, wall):
    if not rows or wall <= 0:
        raise ValueError("Nonempty measurements and positive wall time required")
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    return {
        "records": n,
        "unique_records": len({r["id"] for r in rows}),
        "wall_seconds": wall,
        "accuracy": correct / n,
        "accuracy_wilson_95": wilson(correct, n),
        "schema_accuracy": sum(r["schema_valid"] for r in rows) / n,
        "errors": sum(r["error"] is not None for r in rows),
        "timeouts": sum(
            r["error"] is not None
            and ("timeout" in r["error"]["message"] or "queue" in r["error"]["message"])
            for r in rows
        ),
        "latency_ms": distribution([r["latency_seconds"] * 1000 for r in rows]),
        "first_usable_ms": distribution(
            [
                r["first_usable_seconds"] * 1000
                for r in rows
                if r["first_usable_seconds"] is not None
            ]
        ),
        "first_usable_missing": sum(r["first_usable_seconds"] is None for r in rows),
        "completed_records_per_second": n / wall,
        "correct_records_per_second": correct / wall,
        "output_tokens": sum(r["metrics"].get("output_tokens", 0) for r in rows),
        "dispatch_queue_ms_mean": statistics.mean(r["dispatch_queue_seconds"] for r in rows) * 1000,
        "p99_exploratory": n < 1000,
    }
