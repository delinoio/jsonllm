"""Aggregate repeated trials without treating repeated records as new samples."""

import random
import statistics
from collections import defaultdict

from .benchmark_measure import wilson
from .metrics import distribution


def span(values):
    values = [v for v in values if v is not None]
    return (
        {"median": statistics.median(values), "min": min(values), "max": max(values)}
        if values
        else None
    )


def record_clusters(rows):
    clusters = defaultdict(list)
    for row in rows:
        clusters[row["id"]].append(row)
    return dict(clusters)


def quality(rows):
    clusters = record_clusters(rows)
    stable = sum(all(r["correct"] for r in group) for group in clusters.values())
    return {
        "executions": len(rows),
        "unique_records": len(clusters),
        "execution_accuracy": statistics.mean(r["correct"] for r in rows),
        "correct_in_every_repeat": stable,
        "stable_accuracy_wilson_95": wilson(stable, len(clusters)),
        "schema_accuracy": statistics.mean(r["schema_valid"] for r in rows),
        "errors": sum(r["error"] is not None for r in rows),
        "graph_invalid": sum(r["graph_valid"] is False for r in rows),
    }


def paired_comparison(candidate, baseline, *, draws=2000, seed=20261005):
    """Resample record IDs; keep all repetitions inside each record cluster."""
    left, right = record_clusters(candidate), record_clusters(baseline)
    if set(left) != set(right) or not left:
        raise ValueError("Paired comparisons require exactly the same nonempty record IDs")
    ids = sorted(left)
    delta, speed_left, speed_right = [], [], []
    potential_losses = 0
    for key in ids:
        a, b = left[key], right[key]
        delta.append(
            statistics.mean(r["correct"] for r in a) - statistics.mean(r["correct"] for r in b)
        )
        speed_left.append(statistics.median(r["latency_seconds"] for r in a))
        speed_right.append(statistics.median(r["latency_seconds"] for r in b))
        potential_losses += any(r["correct"] for r in b) and not all(r["correct"] for r in a)
    rng, differences, ratios = random.Random(seed), [], []
    for _ in range(draws):
        sample = rng.choices(range(len(ids)), k=len(ids))
        differences.append(statistics.mean(delta[i] for i in sample))
        ratios.append(
            statistics.median(speed_right[i] for i in sample)
            / statistics.median(speed_left[i] for i in sample)
        )

    def interval(values):
        values.sort()
        return [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]

    point = statistics.mean(delta)
    schema_pass = all(r["schema_valid"] and r["graph_valid"] is not False for r in candidate)
    loss_upper = wilson(potential_losses, len(ids))[1]
    return {
        "unique_paired_records": len(ids),
        "accuracy_delta": point,
        "accuracy_delta_cluster_bootstrap_95": interval(differences),
        "median_record_latency_speedup": statistics.median(speed_right)
        / statistics.median(speed_left),
        "speedup_cluster_bootstrap_95": interval(ratios),
        "potential_loss_fraction_wilson_upper_95": loss_upper,
        "point_quality_gate": schema_pass and point >= -0.02,
        "conservative_quality_gate": schema_pass and loss_upper <= 0.02,
        "uncertainty_note": (
            "Bootstrap resamples record IDs and can be degenerate for perfect scores. "
            "The Wilson upper bound conservatively counts any candidate instability "
            "against any baseline success, without offsetting losses by gains. "
            "Neither interval proves generalization outside these tasks."
        ),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def aggregate_trials(trials):
    """Each trial contains job, summary, and raw rows; only completed trials enter."""
    grouped = defaultdict(list)
    for trial in trials:
        job = trial["job"]
        key = (
            job["stage"],
            job["group"],
            job["model"],
            job["method"],
            job["concurrency"],
            job.get("arrival_rate"),
        )
        grouped[key].append(trial)
    report, rows_by_key = [], {}
    for key, items in grouped.items():
        rows = [r for item in items for r in item["rows"]]
        rows_by_key[key] = rows
        summaries = [item["summary"] for item in items]
        result = dict(
            zip(
                ("stage", "group", "model", "method", "concurrency", "arrival_rate"),
                key,
                strict=True,
            )
        )
        result.update(repeats=len(items), quality=quality(rows), p99_exploratory=True)
        result["trial_metrics"] = {
            name: span(s.get(name) for s in summaries)
            for name in (
                "accuracy",
                "schema_accuracy",
                "completed_records_per_second",
                "correct_records_per_second",
                "output_tokens",
                "wall_seconds",
                "load_seconds",
                "warmup_seconds",
                "sampled_device_peak_memory_bytes",
                "peak_allocated_bytes",
            )
        }
        result["latency_ms"] = {
            p: span(s["latency_ms"][p] for s in summaries) for p in ("p50", "p95", "p99")
        }
        result["by_kind"] = {}
        for kind in sorted({r["kind"] for r in rows}):
            selected = [r for r in rows if r["kind"] == kind]
            result["by_kind"][kind] = {
                "quality": quality(selected),
                "latency_ms": {
                    p: span(
                        distribution(
                            [r["latency_seconds"] * 1000 for r in t["rows"] if r["kind"] == kind]
                        )[p]
                        for t in items
                    )
                    for p in ("p50", "p95", "p99")
                },
            }
        result["first_usable_ms"] = {
            p: span(s["first_usable_ms"].get(p) for s in summaries) for p in ("p50", "p95", "p99")
        }
        result["first_usable_missing"] = sum(s["first_usable_missing"] for s in summaries)
        result["known_output_tokens"] = sum(s["output_tokens"] for s in summaries)
        result["output_token_counts_may_omit_failed_work"] = any(
            r["error"] is not None for r in rows
        )
        result["dispatch_queue_ms"] = distribution(
            [1000 * r["dispatch_queue_seconds"] for r in rows]
        )
        result["lock_queue_ms"] = distribution(
            [
                1000 * r["metrics"]["lock_wait_seconds"]
                for r in rows
                if r["metrics"].get("lock_wait_seconds") is not None
            ]
        )
        result["server_queue_ms"] = distribution(
            [
                r["metrics"]["server_timing"]["queue_time_ms"]
                for r in rows
                if (r["metrics"].get("server_timing") or {}).get("queue_time_ms") is not None
            ]
        )
        result["common_token_evaluations_per_record"] = span(
            r["metrics"]["model"]["common_token_evaluations"]
            for r in rows
            if "model" in r["metrics"]
        )
        result["cache_copy_bytes_per_record"] = span(
            r["metrics"]["model"]["cache_copy_bytes"] for r in rows if "model" in r["metrics"]
        )
        report.append(result)
    comparisons = []
    for key, rows in rows_by_key.items():
        stage, group, model, method, concurrency, rate = key
        if method != "shared_fields" or stage == "development":
            continue
        for baseline_method in ("whole_json", "serial_fields", "batch_fields", "vllm_json"):
            baseline_key = (stage, group, model, baseline_method, concurrency, rate)
            if baseline_key not in rows_by_key:
                continue
            comparisons.append(
                {
                    "stage": stage,
                    "group": group,
                    "model": model,
                    "candidate": method,
                    "baseline": baseline_method,
                    "concurrency": concurrency,
                    "arrival_rate": rate,
                    **paired_comparison(rows, rows_by_key[baseline_key]),
                }
            )
    return {
        "groups": report,
        "paired_comparisons": comparisons,
        "quality_denominator": "unique record IDs; repetitions stay in the same cluster",
    }
