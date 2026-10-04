from jsonllm.benchmark_schedule import estimated_seconds, group_key, schedule


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
