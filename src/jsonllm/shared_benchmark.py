"""Timed UI execution, independent/rollout scoring, and explicit pending factual reviews."""

import copy
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from .diagnostics import exception_details
from .io import dumps
from .metrics import distribution
from .ui import compile_ui, run_ui


def score(item, output):
    wanted = item["expected"]
    if output is None:
        return {
            "component_correct": False,
            "copy_correct": False,
            "state_correct": False,
            "structure_correct": False,
            "faithful": False,
        }

    def props(value):
        return {k: v for k, v in value["props"].items() if k != "description"}

    component = output["component"] == wanted["component"]
    state = output["state"] == wanted["state"]
    copied = output["props"]["entity"] == wanted["props"]["entity"]
    return {
        "component_correct": component,
        "copy_correct": copied,
        "state_correct": state,
        "structure_correct": component and state and props(output) == props(wanted),
        "faithful": True
        if item["kind"] != "generation"
        else (True if output["props"]["description"] == wanted["props"]["description"] else None),
    }


def measure(item, compiled, predictor, *, state=None, max_tokens=128):
    started = time.perf_counter()
    try:
        result = run_ui(
            compiled,
            item["context"],
            item["state"] if state is None else state,
            item["event"],
            item["facts"],
            predictor=predictor,
            max_tokens=max_tokens,
        )
        dumps(result["output"])
    except Exception as exc:
        result = {
            "output": None,
            "diagnostics": {
                "errors": {"exception": type(exc).__name__},
                "exception": exception_details(exc),
            },
        }
    elapsed = time.perf_counter() - started
    scored = score(item, result["output"])
    review = None
    if scored["faithful"] is None:
        review = {
            "sentence": result["output"]["props"]["description"],
            "intended_entry": item["answers"]["entity_choice"],
            "entries": item["facts"]["entries"],
            "language": item["language"],
        }
    return {
        "id": item["id"],
        "group": item["group"],
        "kind": item["kind"],
        "seconds": elapsed,
        "schema_valid": result["output"] is not None,
        **scored,
        **result,
        "factual_review": review,
    }


def evaluate(rows, predictor, *, concurrency=1, rollout=False, warmup=True, max_tokens=128):
    compiled = {row["id"]: compile_ui(row["spec"]) for row in rows}
    warming = time.perf_counter()
    if warmup:
        # Model loading, tokenizer/grammar compilation and warmup are outside timed requests.
        for row in rows[:40]:
            measure(row, compiled[row["id"]], predictor, max_tokens=max_tokens)
    warmup_seconds = time.perf_counter() - warming
    started = time.perf_counter()
    if rollout:
        groups = defaultdict(list)
        for row in rows:
            groups[row["group"]].append(row)

        def trajectory(items):
            state, results = copy.deepcopy(items[0]["state"]), []
            for item in sorted(items, key=lambda r: r["step"]):
                result = measure(
                    item, compiled[item["id"]], predictor, state=state, max_tokens=max_tokens
                )
                # A failed transition retains the last available prediction; never substitute gold.
                if result["output"] is not None:
                    state = result["output"]["state"]
                results.append(result)
            return results

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            measured = [r for group in pool.map(trajectory, groups.values()) for r in group]
    else:

        def one(item):
            return measure(item, compiled[item["id"]], predictor, max_tokens=max_tokens)

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            measured = list(pool.map(one, rows))
    wall = time.perf_counter() - started
    return measured, {
        "wall_seconds": wall,
        "warmup_and_compile_seconds": warmup_seconds,
        "concurrency": concurrency,
        "rollout": rollout,
        "records": len(rows),
        "completed_records_per_second": len(rows) / wall,
    }


def summary(rows, *, regression_passed, wall_seconds=None):
    modeled = [r for r in rows if r["kind"] != "code"]
    code = [r for r in rows if r["kind"] == "code"]
    free = [r for r in rows if r["kind"] == "generation"]

    def correct(r):
        return r["structure_correct"] and r["faithful"] is True

    latency = distribution([r["seconds"] * 1000 for r in modeled])
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    metrics = [r.get("diagnostics", {}).get("model", {}) for r in modeled]

    def stage_totals(key, defaults=()):
        keys = set(defaults).union(*(m.get(key, {}) for m in metrics))
        return {k: sum(m.get(key, {}).get(k, 0) for m in metrics) for k in sorted(keys)}

    profiled = [m for m in metrics if m.get("profile_stages")]
    return {
        "records": len(rows),
        "model_records": len(modeled),
        "unique_model_records": len({r.get("id", i) for i, r in enumerate(modeled)}),
        "code_records": len(code),
        "judgments_complete": all(r["faithful"] is not None for r in rows),
        "pending_factual_reviews": sum(r["faithful"] is None for r in rows),
        "schema_accuracy": sum(r["schema_valid"] for r in rows) / len(rows),
        "code_accuracy": sum(correct(r) for r in code) / len(code) if code else 1,
        "model_accuracy": sum(correct(r) for r in modeled) / len(modeled),
        "component_accuracy": sum(r["component_correct"] for r in modeled) / len(modeled),
        "copy_accuracy": sum(r["copy_correct"] for r in modeled) / len(modeled),
        "state_accuracy": sum(r["state_correct"] for r in modeled) / len(modeled),
        "factual_accuracy": sum(r["faithful"] is True for r in free) / len(free) if free else None,
        "trajectory_accuracy": sum(all(correct(r) for r in g) for g in groups.values())
        / len(groups),
        "regression_passed": regression_passed,
        "latency_ms": latency,
        "p50_ms": latency["p50"],
        "p95_ms": latency["p95"],
        "p99_ms": latency["p99"],
        "code_latency_ms": distribution([r["seconds"] * 1000 for r in code]),
        "correct_records_per_second": sum(correct(r) for r in rows) / wall_seconds
        if wall_seconds
        else None,
        "model_correct_records_per_second": sum(correct(r) for r in modeled) / wall_seconds
        if wall_seconds
        else None,
        "under_500ms_fraction": sum(r["seconds"] <= 0.5 for r in modeled) / len(modeled),
        "failed_records": sum(not r["schema_valid"] for r in rows),
        "output_tokens": sum(
            c.get("output_tokens", 0) for m in metrics for c in m.get("calls", [])
        ),
        "decision_calls": sum(len(m.get("calls", [])) for m in metrics),
        "backbone_forwards": sum(len(m.get("forwards", [])) for m in metrics),
        "common_prefills": sum(m.get("common_prefills", 0) for m in metrics),
        "common_token_evaluations": sum(m.get("common_token_evaluations", 0) for m in metrics),
        "context_tokenizations": sum(m.get("context_tokenizations", 0) for m in metrics),
        "cache_copy_bytes": sum(m.get("cache_copy_bytes", 0) for m in metrics),
        "gpu_seconds": stage_totals("gpu_seconds", ("common", "questions", "decode", "cache_copy")),
        "stage_profile": {
            "profiled_records": len(profiled),
            "host_seconds": stage_totals("stage_host_seconds"),
            **{
                k: sum(m.get(k, 0) for m in profiled)
                for k in ("cache_reorder_count", "cache_reorder_bytes", "mask_transfer_bytes")
            },
        },
    }
