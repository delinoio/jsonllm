import pytest

from jsonllm.benchmark_measure import summarize
from jsonllm.benchmark_report import aggregate_trials, paired_comparison, quality


def row(name, *, correct=True, latency=1):
    return {
        "id": name,
        "correct": correct,
        "latency_seconds": latency,
        "schema_valid": True,
        "graph_valid": None,
        "error": None,
    }


def test_repetitions_do_not_inflate_quality_sample_size():
    rows = [row("one"), row("two", correct=False)]
    single, repeated = quality(rows), quality(rows * 5)
    assert repeated["executions"] == 10
    assert repeated["unique_records"] == single["unique_records"] == 2
    assert repeated["stable_accuracy_wilson_95"] == single["stable_accuracy_wilson_95"]


def test_paired_bootstrap_retains_uncertainty_for_perfect_small_samples():
    candidate = [row(str(i)) for i in range(8)]
    baseline = [row(str(i), latency=2) for i in range(8)]
    result = paired_comparison(candidate * 5, baseline * 5, draws=100)
    assert result["unique_paired_records"] == 8
    assert result["speedup_cluster_bootstrap_95"] == [2, 2]
    assert result["point_quality_gate"]
    assert not result["conservative_quality_gate"]
    assert result["potential_loss_fraction_wilson_upper_95"] > 0.02


def test_incorrect_or_missing_records_cannot_pass_comparison():
    candidate, baseline = [row("a", correct=False)], [row("a")]
    assert not paired_comparison(candidate, baseline, draws=100)["point_quality_gate"]
    with pytest.raises(ValueError, match="same nonempty record IDs"):
        paired_comparison(candidate, [row("b")])


def test_repeat_instability_is_preserved_in_conservative_loss_bound():
    candidate = [row("a"), row("a", correct=False)]
    result = paired_comparison(candidate, [row("a")] * 2, draws=100)
    assert result["accuracy_delta"] == -0.5
    assert not result["point_quality_gate"]


def test_adjacent_contrasts_and_terminal_valid_correct_throughput_are_distinct():
    trials = []
    for method in ("whole_json", "serial_fields", "batch_fields", "shared_fields", "vllm_json"):
        rows = []
        for i, language in enumerate(("en", "ko")):
            r = row(str(i), correct=i == 0)
            r.update(
                kind="choice",
                language=language,
                first_usable_seconds=0.5,
                dispatch_queue_seconds=0,
                metrics={},
            )
            if method == "whole_json" and i == 1:
                r.update(
                    schema_valid=False,
                    error={"type": "ReadTimeout", "message": "", "timeout": True},
                )
            rows.append(r)
        trials.append(
            {
                "job": {
                    "stage": "core",
                    "group": "core",
                    "model": "base",
                    "method": method,
                    "concurrency": 1,
                },
                "summary": summarize(rows, 2),
                "rows": rows,
            }
        )
    result = aggregate_trials(trials)
    contrasts = {(c["candidate"], c["baseline"]) for c in result["paired_comparisons"]}
    assert ("serial_fields", "whole_json") in contrasts
    assert ("batch_fields", "serial_fields") in contrasts
    assert ("shared_fields", "batch_fields") in contrasts
    groups = {g["method"]: g for g in result["groups"]}
    whole = groups["whole_json"]
    assert whole["quality"]["timeouts"] == 1
    assert whole["valid_completion_fraction"] == 0.5
    assert whole["trial_metrics"]["completed_records_per_second"]["median"] == 1
    assert whole["valid_records_per_second"]["median"] == 0.5
    serial = groups["serial_fields"]
    assert serial["valid_records_per_second"]["median"] == 1
    assert serial["trial_metrics"]["correct_records_per_second"]["median"] == 0.5
    assert serial["by_language"]["ko"]["quality"]["unique_records"] == 1
