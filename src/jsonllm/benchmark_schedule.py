"""Fixed experiment ordering and conservative budget admission."""

from .benchmark_engine import METHODS


def schedule(manifest):
    jobs = []
    for repeat in range(5):
        methods = METHODS[repeat:] + METHODS[:repeat]
        for model in ("base", "jsonllm") if repeat % 2 == 0 else ("jsonllm", "base"):
            for method in methods:
                jobs.append(
                    {
                        "stage": "core",
                        "group": "core",
                        "repeat": repeat,
                        "model": model,
                        "method": method,
                        "concurrency": 1,
                    }
                )
    # vLLM is the practical baseline in core/load; sweeps isolate the four eager mechanisms.
    for group in manifest["groups"]:
        if not group.startswith("scale-") or not manifest["groups"][group]["included"]:
            continue
        for repeat in range(3):
            methods = METHODS[:4]
            methods = methods[repeat:] + methods[:repeat]
            for model in ("base", "jsonllm"):
                for method in methods:
                    jobs.append(
                        {
                            "stage": "scale",
                            "group": group,
                            "repeat": repeat,
                            "model": model,
                            "method": method,
                            "concurrency": 1,
                        }
                    )
    for model in ("base", "jsonllm"):
        for method in ("shared_fields", "vllm_json"):
            for concurrency in (1, 2, 4, 8):
                jobs.append(
                    {
                        "stage": "load",
                        "group": "core",
                        "repeat": 0,
                        "model": model,
                        "method": method,
                        "concurrency": concurrency,
                        "limit": 64,
                    }
                )
            for rate in (0.5, 1, 2, 4):
                jobs.append(
                    {
                        "stage": "load",
                        "group": "core",
                        "repeat": 0,
                        "model": model,
                        "method": method,
                        "concurrency": 128,
                        "arrival_rate": rate,
                        "limit": 64,
                    }
                )
    for group in ("tree", "workflow"):
        for model in ("base", "jsonllm"):
            for method in METHODS:
                jobs.append(
                    {
                        "stage": "topology",
                        "group": group,
                        "repeat": 0,
                        "model": model,
                        "method": method,
                        "concurrency": 1,
                    }
                )
    for model in ("base", "jsonllm"):
        for method in ("whole_json", "shared_fields", "vllm_json"):
            jobs.append(
                {
                    "stage": "application",
                    "group": "application",
                    "repeat": 0,
                    "model": model,
                    "method": method,
                    "concurrency": 1,
                }
            )
    for index, job in enumerate(jobs):
        job["name"] = f"{index:04}-{job['stage']}-{job['group']}-{job['model']}-{job['method']}"
    return jobs


def group_key(job):
    """Do not start a comparison condition unless all its methods can plausibly finish."""
    return (job["stage"], job["group"]) if job["stage"] != "load" else ("load",)


def admit_group(group, timings, manifest, remaining, *, exhausted=False):
    """Stop at the first unaffordable block; never backfill with cheaper later blocks."""
    estimate = sum(
        estimated_seconds(j, timings, manifest["groups"][j["group"]]["included"]) for j in group
    )
    return not exhausted and estimate < remaining, estimate


def estimated_seconds(job, timings, count):
    timing = timings[(job["model"], job["method"])]
    if isinstance(timing, dict):
        per_record = timing["seconds_per_record"]
        setup = max(120, timing["setup_seconds"])
    else:
        per_record, setup = timing, 120
    records = min(count, job.get("limit", count))
    arrival = records / job["arrival_rate"] if job.get("arrival_rate") else 0
    # Include process/model/server startup and extra margin. No quality-dependent admission.
    return 1.5 * max(per_record * records, arrival) + setup


def development_timing(summary, server_startup=0):
    """Use only development timings, including preparation and server startup."""
    return {
        "seconds_per_record": summary["wall_seconds"] / summary["records"],
        "setup_seconds": 1.25
        * (
            summary.get("load_seconds", 0)
            + summary.get("warmup_seconds", 0)
            + summary.get("schema_preparation", {}).get("seconds", 0)
            + server_startup
        )
        + 15,
    }
