import copy

from jsonllm.shared_benchmark import score, summary


def item():
    return {
        "kind": "generation",
        "expected": {
            "component": "Card",
            "state": {"x": 0},
            "props": {"entity": "entry", "description": "Entry is ready", "urgent": False},
        },
    }


def test_paraphrases_remain_unjudged_and_failures_count_in_denominator():
    example = item()
    output = copy.deepcopy(example["expected"])
    output["props"]["description"] = "Entry: ready"
    scored = score(example, output)
    assert scored["structure_correct"] and scored["faithful"] is None
    rows = [
        {"group": 0, "kind": "generation", "seconds": 0.1, "schema_valid": True, **scored},
        {
            "group": 1,
            "kind": "semantic",
            "seconds": 0.2,
            "schema_valid": False,
            **score(example, None),
        },
    ]
    result = summary(rows, regression_passed=True, wall_seconds=0.3)
    assert not result["judgments_complete"] and result["model_accuracy"] == 0
    assert result["schema_accuracy"] == 0.5 and result["failed_records"] == 1
    assert result["latency_ms"]["count"] == 2
    assert result["copy_accuracy"] == 0.5


def test_code_only_latency_and_accuracy_are_separate_from_model_results():
    base = {
        "group": 0,
        "schema_valid": True,
        "structure_correct": True,
        "faithful": True,
        "component_correct": True,
        "copy_correct": True,
        "state_correct": True,
    }
    rows = [base | {"kind": "code", "seconds": 0.0001}, base | {"kind": "semantic", "seconds": 1}]
    result = summary(rows, regression_passed=True, wall_seconds=1.0001)
    assert result["p50_ms"] == 1000 and result["code_latency_ms"]["p50"] == 0.1
    assert result["model_records"] == 1 and result["under_500ms_fraction"] == 0


def test_uncaught_errors_keep_message_and_traceback_without_locals(monkeypatch):
    import jsonllm.shared_benchmark as benchmark

    def fail(*args, **kwargs):
        sensitive_local = "no-local-variable-dumps"
        assert sensitive_local
        raise RuntimeError("kernel failure details")

    monkeypatch.setattr(benchmark, "run_ui", fail)
    example = item() | {"id": "x", "group": 0, "context": "", "state": {}, "event": {}, "facts": {}}
    result = benchmark.measure(example, object(), object())
    assert result["output"] is None and not result["schema_valid"]
    diag = result["diagnostics"]
    assert diag["errors"]["exception"] == "RuntimeError"
    assert diag["exception"]["message"] == "kernel failure details"
    assert "RuntimeError: kernel failure details" in diag["exception"]["traceback"]
    assert "no-local-variable-dumps" not in diag["exception"]["traceback"]


def test_summary_includes_new_profile_stages_and_accepts_old_metrics():
    base = {
        "group": 0,
        "kind": "semantic",
        "seconds": 0.1,
        "schema_valid": True,
        "structure_correct": True,
        "faithful": True,
        "component_correct": True,
        "copy_correct": True,
        "state_correct": True,
    }
    old = base | {"diagnostics": {"model": {"gpu_seconds": {"decode": 0.2}}}}
    new = base | {
        "diagnostics": {
            "model": {
                "profile_stages": True,
                "gpu_seconds": {"decode": 0.3, "lm_head": 0.1, "cache_reorder": 0.02},
                "stage_host_seconds": {"lock_wait": 0.4, "grammar_fill": 0.2},
                "cache_reorder_count": 2,
                "cache_reorder_bytes": 100,
                "mask_transfer_bytes": 16,
            }
        }
    }
    report = summary([old, new], regression_passed=True)
    assert report["gpu_seconds"]["decode"] == 0.5
    assert report["gpu_seconds"]["lm_head"] == 0.1
    assert report["gpu_seconds"]["cache_copy"] == 0
    assert report["stage_profile"] == {
        "profiled_records": 1,
        "host_seconds": {"lock_wait": 0.4, "grammar_fill": 0.2},
        "cache_reorder_count": 2,
        "cache_reorder_bytes": 100,
        "mask_transfer_bytes": 16,
    }
