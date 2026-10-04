"""Prepare all inference contracts before sealing the benchmark inputs."""

import hashlib
import math
from pathlib import Path

from .benchmark_data import suite
from .benchmark_engine import render_output, whole_prompt
from .io import dumps, write_json, write_jsonl
from .prompts import SHARED_VERSION, context_ids, field_example, prompt_ids
from .release import BASE_MODEL, BASE_REVISION, MODEL_ID, MODEL_REVISION


def prepare(root, tokenizer):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    prepared, exclusions = {}, []
    for group, rows in suite().items():
        accepted = []
        for row in rows:
            target = row["condition"].get("context_tokens")
            if target:
                # Tokenize the complete context every time; no artificial token boundary.
                while len(context_ids(row["context"], tokenizer)) < target:
                    row["context"] += "\nUnrelated note: plain neutral filler."
                row["condition"]["actual_context_tokens"] = len(
                    context_ids(row["context"], tokenizer)
                )
            json_ids = whole_prompt(row, tokenizer)
            row["json_max_tokens"] = max(
                128,
                math.ceil(
                    len(
                        tokenizer.encode(
                            dumps(render_output(row, row["answers"])), add_special_tokens=False
                        )
                    )
                    * 1.25
                )
                + 32,
            )
            lengths = []
            for name in row["questions"]:
                example = field_example(row, name, prompt_version=SHARED_VERSION)
                count = len(prompt_ids(example, tokenizer))
                lengths.append(count + (1 if example["choices"] else 128))
            json_length = len(json_ids) + row["json_max_tokens"]
            row["tokens"] = {
                "context": len(context_ids(row["context"], tokenizer)),
                "json_prompt": len(json_ids),
                "json_reserved": row["json_max_tokens"],
                "field_max_reserved": max(lengths),
                "answer_value_tokens": [
                    len(tokenizer.encode(dumps(v), add_special_tokens=False))
                    for v in row["answers"].values()
                ],
            }
            if max(*lengths, json_length) > 2048 or row["json_max_tokens"] > 1024:
                exclusions.append(
                    {
                        "id": row["id"],
                        "group": group,
                        "reason": "overlength",
                        "tokens": row["tokens"],
                    }
                )
                continue
            accepted.append(row)
        write_jsonl(root / (group + ".jsonl"), accepted)
        prepared[group] = {
            "requested": len(rows),
            "included": len(accepted),
            "sha256": hashlib.sha256((root / (group + ".jsonl")).read_bytes()).hexdigest(),
        }
    write_json(root / "exclusions.json", exclusions)
    manifest = {
        "schema_version": 1,
        "seed": 20261005,
        "groups": prepared,
        "models": {
            "base": {"id": BASE_MODEL, "revision": BASE_REVISION},
            "jsonllm": {"id": MODEL_ID, "revision": MODEL_REVISION},
        },
        "precision": "float16",
        "max_length": 2048,
        "field_max_tokens": 128,
        "core_repeats": 5,
        "scale_repeats": 3,
        "concurrencies": [1, 2, 4, 8],
        "arrival_rates": [0.5, 1, 2, 4],
        "request_timeout_seconds": 120,
        "priority": ["core", "scale", "load", "topology", "application"],
        "total_budget_usd": 100,
        "recovery_at_usd": 90,
        "output_budget_policy": (
            "1.25 times reference JSON tokens + 32, at least 128; fixed before inference"
        ),
        "factual_judge": "independent deterministic oracle; no teacher API",
        "token_limit_exclusions_apply_to_all_methods": True,
    }
    write_json(root / "manifest.json", manifest)
    return manifest
