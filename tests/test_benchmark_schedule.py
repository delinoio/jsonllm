from jsonllm.benchmark_schedule import (
    admit_group,
    development_timing,
    estimated_seconds,
    group_key,
    schedule,
)


def test_fixed_order_repeats_and_complete_core():
    jobs = schedule({"groups": {"scale-width-1": {"included": 32}}})
    core = [j for j in jobs if j["stage"] == "core"]
    assert len(core) == 50
    assert len({j["name"] for j in jobs}) == len(jobs)
    assert {j["repeat"] for j in core} == set(range(5))
    assert core[0]["method"] != core[10]["method"]
    assert all(j["concurrency"] != 32 for j in jobs)
    assert group_key(core[0]) == group_key(core[-1])


def test_time_estimate_is_independent_of_quality():
    job = {"model": "base", "method": "whole_json", "limit": 4, "arrival_rate": 0.5}
    assert estimated_seconds(job, {("base", "whole_json"): 1}, 256) == 132


def test_budget_exhaustion_cannot_select_later_cheap_conditions():
    group = [{"model": "base", "method": "whole_json", "group": "core"}]
    manifest = {"groups": {"core": {"included": 256}}}
    timings = {("base", "whole_json"): 1}
    assert admit_group(group, timings, manifest, 100)[0] is False
    assert admit_group(group, timings, manifest, 1000)[0] is True
    assert admit_group(group, timings, manifest, 1000, exhausted=True)[0] is False


def test_expensive_server_setup_is_included_in_every_budget_admission():
    timing = development_timing(
        {
            "records": 16,
            "wall_seconds": 16,
            "load_seconds": 80,
            "warmup_seconds": 20,
            "schema_preparation": {"seconds": 4},
        },
        server_startup=300,
    )
    assert timing["setup_seconds"] > 500
    job = {"model": "base", "method": "vllm_json"}
    assert estimated_seconds(job, {("base", "vllm_json"): timing}, 32) > 550
