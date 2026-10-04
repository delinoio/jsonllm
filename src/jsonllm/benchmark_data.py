"""Fresh, executable-oracle tasks for the design benchmark; no external API calls."""

import copy
import hashlib
import random

from .io import dumps
from .schema import validate_record

SEED = 20261005
KINDS = ("choice", "number", "string", "dependency")


def reference(case):
    """Compute gold from task facts, independently of prompts and model execution."""
    source = case["source"]
    kind = case["kind"]
    if kind in {"tree", "workflow"}:
        result = {}
        for i, node in enumerate(source["nodes"]):
            active = node["enabled"]
            result[f"active_{i}"] = active
            result[f"kind_{i}"] = node["kind"] if active else "none"
            result[f"parent_{i}"] = node["parent"] if active else -1
            if kind == "workflow":
                result[f"extra_{i}"] = node["extra"] if active else -1
        return result
    entries = {entry["key"]: entry for entry in source["entries"]}
    result = {}
    for i, query in enumerate(source["queries"]):
        if kind == "choice":
            pool = [entries[key] for key in query["keys"]]
            result[f"f{i}"] = max(pool, key=lambda row: row["score"])["key"]
        elif kind == "number":
            row = entries[query["key"]]
            result[f"f{i}"] = row["value"] if row["available"] else None
        elif kind == "string":
            result[f"f{i}"] = entries[query["key"]]["text"]
        elif kind == "dependency":
            key = query["key"] if query["parent"] is None else result[f"f{query['parent']}"]
            result[f"f{i}"] = entries[key]["next"]
        else:
            raise ValueError(kind)
    return result


def make_case(split, index, *, kind=None, width=8, depth=4, choices=4, text_words=8):
    kind = kind or KINDS[(index // 2) % len(KINDS)]
    seed = int.from_bytes(hashlib.sha256(f"{SEED}:{split}:{index}".encode()).digest()[:8])
    rng = random.Random(seed)
    language = "en" if index % 2 == 0 else "ko"
    keys = [f"item_{i}" for i in range(max(width, choices))]
    scores = rng.sample(range(100, 999), len(keys))
    entries = [
        {
            "key": key,
            "score": scores[i],
            "available": rng.choice([True, True, False]),
            "value": rng.randrange(-30, 70),
            "text": " ".join(
                rng.choices(["red", "blue", "green", "silver", "gold", "white"], k=text_words)
            ),
            "next": rng.choice(keys),
        }
        for i, key in enumerate(keys)
    ]
    questions, queries = {}, []
    for i in range(width):
        key = rng.choice(keys)
        if kind == "choice":
            selected = rng.sample(keys, choices)
            query = {"keys": selected}
            instructions = (
                f"Among {dumps(selected)}, return the key with the highest score."
                if language == "en"
                else f"{dumps(selected)} 중 score가 가장 높은 항목의 key를 반환하세요."
            )
            field = {"type": "string", "enum": selected}
        elif kind == "number":
            query = {"key": key}
            instructions = (
                f"For {key}, return value if available is true; otherwise return null."
                if language == "en"
                else f"{key}의 available이 true이면 value를, 아니면 null을 반환하세요."
            )
            field = {"type": ["integer", "null"], "minimum": -30, "maximum": 69}
        elif kind == "string":
            query = {"key": key}
            instructions = (
                f"Copy the exact text for {key}."
                if language == "en"
                else f"{key}의 text를 정확히 복사하세요."
            )
            field = {"type": "string", "maxLength": max(len(e["text"]) for e in entries)}
        elif kind == "dependency":
            # Fixed width, variable chain length; siblings never receive each other's answers.
            parent = i - 1 if i % depth else None
            query = {"key": key, "parent": parent}
            subject = key if parent is None else f"the key returned by f{parent}"
            instructions = (
                f"Return the next key for {subject}."
                if language == "en"
                else f"{key if parent is None else f'f{parent}의 결과'} 항목의 next를 반환하세요."
            )
            field = {"type": "string", "enum": keys}
            if parent is not None:
                field["depends_on"] = [f"f{parent}"]
        else:
            raise ValueError(kind)
        questions[f"f{i}"] = field | {"instructions": instructions}
        queries.append(query)
    needed = {
        "choice": ("key", "score"),
        "number": ("key", "available", "value"),
        "string": ("key", "text"),
        "dependency": ("key", "next"),
    }[kind]
    source = {"entries": [{k: row[k] for k in needed} for row in entries], "queries": queries}
    context = (
        "Use only the supplied facts. Each field is a separate query.\n"
        if language == "en"
        else "제공된 사실만 사용하세요. 각 필드는 별개의 질문입니다.\n"
    ) + dumps({"entries": source["entries"], "nonce": f"{seed:016x}"})
    case = {
        "id": f"design-{split}-{index:04}",
        "language": language,
        "kind": kind,
        "context": context,
        "questions": questions,
        "source": source,
        "condition": {"width": width, "depth": depth, "choices": choices},
    }
    case["answers"] = reference(case)
    validate_record(case)
    return case


def topology_case(kind, index):
    case = make_case(kind, index, width=1)
    rng = random.Random(f"{SEED}-{kind}-{index}")
    nodes, questions = [], {}
    active = []
    for i in range(8):
        enabled = i == 0 or rng.random() > 0.35
        parent = rng.choice(active) if active and enabled else -1
        extra_pool = [p for p in active if p != parent]
        extra = rng.choice(extra_pool) if extra_pool and enabled and rng.random() > 0.5 else -1
        nodes.append(
            {
                "enabled": enabled,
                "kind": rng.choice(["input", "map", "output"]),
                "parent": parent,
                "extra": extra,
            }
        )
        if enabled:
            active.append(i)
        specs = {
            "active": {"type": "boolean"},
            "kind": {"type": "string", "enum": ["none", "input", "map", "output"]},
            "parent": {"type": "integer", "enum": list(range(-1, max(1, i)))},
        }
        if kind == "workflow":
            specs["extra"] = specs["parent"].copy()
        for attr, field in specs.items():
            instructions = (
                f"For node {i}, return enabled."
                if attr == "active"
                else f"For node {i}, return {attr} if enabled; otherwise "
                + ("none." if attr == "kind" else "-1.")
            )
            if case["language"] == "ko":
                instructions = (
                    f"노드 {i}의 enabled를 반환하세요."
                    if attr == "active"
                    else f"노드 {i}의 enabled가 true이면 {attr}, 아니면 "
                    + ("none을 반환하세요." if attr == "kind" else "-1을 반환하세요.")
                )
            questions[f"{attr}_{i}"] = field | {"instructions": instructions}
    case.update(
        kind=kind,
        source={"nodes": nodes},
        questions=questions,
        context="Select the enabled nodes and their links.\n" + dumps({"nodes": nodes}),
    )
    case["answers"] = reference(case)
    validate_record(case)
    return case


def assemble_topology(values, kind):
    """Assemble predicted slots; never infer, repair, or consult source facts."""
    nodes = []
    for i in range(8):
        if values[f"active_{i}"]:
            parents = [values[f"parent_{i}"]]
            if kind == "workflow":
                parents.append(values[f"extra_{i}"])
            nodes.append(
                {"id": i, "kind": values[f"kind_{i}"], "parents": [p for p in parents if p != -1]}
            )
    ids = {n["id"] for n in nodes}
    for node in nodes:
        if node["kind"] == "none" or len(set(node["parents"])) != len(node["parents"]):
            raise ValueError("Invalid node")
        if any(p not in ids or p >= node["id"] for p in node["parents"]):
            raise ValueError("Invalid reference or non-topological edge")
    if kind == "tree" and (not nodes or sum(not n["parents"] for n in nodes) != 1):
        raise ValueError("Expected one tree root")
    return {"nodes": nodes}


def inference_record(case):
    """Allowlist the inference boundary; no source oracle or reference answers."""
    return copy.deepcopy({key: case[key] for key in ("id", "context", "questions")})


def suite():
    result = {
        "dev": [make_case("dev", i) for i in range(64)],
        "core": [make_case("core", i) for i in range(256)],
    }
    sweeps = {
        "width": [1, 4, 8, 16],
        "context_tokens": [128, 512, 1024, 1536],
        "depth": [1, 2, 4, 8],
        "choices": [2, 4, 8, 16],
        "text_words": [8, 32, 96],
    }
    for axis, levels in sweeps.items():
        for level in levels:
            name = f"scale-{axis}-{level}"
            options = {axis: level} if axis != "context_tokens" else {}
            options["kind"] = (
                "dependency"
                if axis == "depth"
                else ("string" if axis == "text_words" else "choice")
            )
            result[name] = [make_case(name, i, **options) for i in range(32)]
            for row in result[name]:
                row["condition"].update(axis=axis, level=level)
                if axis == "context_tokens":
                    row["condition"]["context_tokens"] = level
    for kind in ("tree", "workflow"):
        result[kind] = [topology_case(kind, i) for i in range(64)]
    return result
