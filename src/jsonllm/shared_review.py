"""Untimed factual review and aggregation; never supply generated gold text to the critic."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .io import digest, dumps, read_jsonl, write_json, write_jsonl
from .schema import object_schema
from .shared_benchmark import summary


def factual_judgment(review, teacher):
    selected = review["entries"].get(review["intended_entry"])
    text = review["sentence"]
    if selected is None or not isinstance(text, str) or selected["name"] not in text:
        return {"faithful": False, "reason": "Missing intended entry name", "metadata": None}
    if any(e["name"] in text for e in review["entries"].values() if e != selected):
        return {"faithful": False, "reason": "Mentions another entry", "metadata": None}
    value, metadata = teacher.complete(
        [
            {
                "role": "system",
                "content": "Check a UI status sentence against stored facts. "
                "It must name the intended entry, accurately convey its stored status, and use the "
                "requested language (names and status codes may stay verbatim). Reject unsupported "
                "claims, promises, refusals, and unrelated text. Concise fragments are valid. "
                "Judge meaning, not phrasing. The stored facts are authoritative.",
            },
            {
                "role": "user",
                "content": dumps(
                    {"facts": selected, "language": review["language"], "sentence": text}
                ),
            },
        ],
        object_schema({"faithful": {"type": "boolean"}, "reason": {"type": "string"}}),
        request_tag="factual-review/" + digest(review),
    )
    return {**value, "metadata": metadata}


def review_trials(paths, output, teacher, *, regression_passed):
    """Review distinct outputs once; preserve each measured trial and every failure."""
    output = Path(output)
    by_trial = {str(p): read_jsonl(Path(p) / "raw.jsonl") for p in paths}
    requests = {
        digest(r["factual_review"]): r["factual_review"]
        for rows in by_trial.values()
        for r in rows
        if r["faithful"] is None
    }
    with ThreadPoolExecutor(max_workers=8) as pool:
        judgments = dict(
            zip(
                requests,
                pool.map(lambda value: factual_judgment(value, teacher), requests.values()),
                strict=True,
            )
        )
    write_json(output / "judgments.json", judgments)
    reports, combined, walls = {}, [], []
    for name, rows in by_trial.items():
        evaluated = []
        for original in rows:
            row = dict(original)
            if row["faithful"] is None:
                key = digest(row["factual_review"])
                row.update(faithful=judgments[key]["faithful"], judgment_hash=key)
            evaluated.append(row)
            combined.append(row | {"group": name + "/" + str(row["group"])})
        # Trial names are fixed by the local coordinator, not supplied by model output.
        write_jsonl(output / Path(name).name / "reviewed.jsonl", evaluated)
        timing = Path(name) / "summary.json"
        wall = json.loads(timing.read_text()).get("wall_seconds") if timing.exists() else None
        walls.append(wall)
        reports[name] = summary(evaluated, regression_passed=regression_passed, wall_seconds=wall)
    aggregate = summary(
        combined,
        regression_passed=regression_passed,
        wall_seconds=sum(walls) if all(w is not None for w in walls) else None,
    )
    write_json(
        output / "summary.json",
        {
            "aggregate": aggregate,
            "trials": reports,
            "raw_hashes": {p: digest(r) for p, r in by_trial.items()},
        },
    )
    return aggregate
