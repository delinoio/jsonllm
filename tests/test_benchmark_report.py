import pytest

from jsonllm.benchmark_report import paired_comparison, quality


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
