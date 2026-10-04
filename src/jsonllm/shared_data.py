"""GenUI semantic tasks, grouped splits, and an independent deterministic reference."""

import random
import re

from .io import digest, dumps
from .schema import object_schema
from .ui import compile_ui, expression, model_context, ui_environment

VERSION = "shared-genui-v1"


def normalized_context(text):
    """Ignore synthetic entry names when rejecting repeated phrasings."""
    return " ".join(re.sub(r"(?:Entry|항목) \d+-\d+", "ENTRY", text).casefold().split())


# Entire workflow families are held out, not just entities or paraphrases.
FAMILIES = {
    "train": [
        "support tickets",
        "product discovery",
        "order tracking",
        "refund requests",
        "meeting scheduling",
        "document approval",
        "inventory management",
        "sales dashboard",
        "account settings",
        "subscription management",
        "task planning",
        "expense review",
        "course enrollment",
        "event registration",
        "recipe search",
        "media library",
        "team directory",
        "shipment dispatch",
        "survey builder",
        "project milestones",
    ],
    "validation": [
        "room reservations",
        "service appointments",
        "incident triage",
        "invoice reconciliation",
    ],
    "test": ["travel itinerary", "hotel services", "software deployments", "equipment maintenance"],
}
PURPOSES = [
    "Browse or search multiple entries; compare the available options without modifying them.",
    "Inspect the full details of one existing entry without changing it.",
    "Create a new entry or edit the information of an existing entry.",
    "Request destructive deletion of an existing entry, with confirmation before execution.",
    "Export existing entries into a downloadable file without changing the entries.",
    "View aggregated counts, trends and performance statistics rather than individual details.",
    "Change notification delivery preferences without modifying entry information.",
    "Invite another person or change their access permissions.",
]
KINDS = ["semantic"] * 3 + ["code"] + ["semantic"] * 3 + ["code"] + ["generation"] * 2


def assignment(group):
    if not 0 <= group < 1000:
        raise ValueError("Group outside frozen 10,000-record allocation")
    split = "train" if group < 800 else "validation" if group < 900 else "test"
    families = FAMILIES[split]
    return split, families[group % len(families)]


def blueprint(group, seed=20260929):
    rng = random.Random(seed + group)
    split, family = assignment(group)
    purposes = rng.sample(range(len(PURPOSES)), rng.randint(4, 8))
    ids = [f"panel_{group}_{n}" for n in rng.sample(range(10, 99), len(purposes))]
    registry = {
        key: {
            "purpose": PURPOSES[purpose],
            "title": f"{family} · "
            + [
                "Browse",
                "Details",
                "Edit",
                "Confirm",
                "Export",
                "Statistics",
                "Settings",
                "Access",
            ][purpose],
            "action": [
                "search",
                "open",
                "save",
                "confirm",
                "export",
                "analyse",
                "configure",
                "invite",
            ][purpose],
        }
        for key, purpose in zip(ids, purposes, strict=True)
    }
    entity_ids = [f"ref_{n}" for n in rng.sample(range(100, 999), 3)]
    entities = {
        key: {
            "name": f"{'항목' if group % 2 else 'Entry'} {group}-{j}",
            "count": rng.choice([0, 1, 3, 12]),
            "status": rng.choice(["pending", "ready", "blocked"]),
        }
        for j, key in enumerate(entity_ids)
    }
    cases = []
    for step, kind in enumerate(KINDS):
        selected = rng.choice(ids + [None]) if step < 8 else rng.choice(ids)
        entity = rng.choice(entity_ids + [None]) if step < 8 else entity_ids[step % 3]
        cases.append(
            {
                "step": step,
                "kind": kind,
                "component": selected,
                "entity": entity,
                "urgent": bool(rng.randrange(2)),
                "seed": rng.randrange(10**8),
            }
        )
    return {
        "group": group,
        "split": split,
        "family": family,
        "language": "ko" if group % 2 else "en",
        "registry": registry,
        "entities": entities,
        "cases": cases,
    }


def path(name):
    return {"path": name}


def op(name, *args):
    return {"op": name, "args": list(args)}


def choose(condition, yes, no):
    return {"if": condition, "then": yes, "else": no}


def lookup(table, key):
    return {"lookup": table, "key": key}


def specification(bp, kind):
    rng = random.Random(bp["group"] * 17 + KINDS.index(kind))
    component_ids, entity_ids = list(bp["registry"]) + [None], list(bp["entities"]) + [None]
    rng.shuffle(component_ids)
    rng.shuffle(entity_ids)
    user = op("eq", path("event.type"), "user")
    exact_ref = None
    for key, entry in reversed(list(bp["entities"].items())):
        exact_ref = choose(op("in", entry["name"], path("context")), key, exact_ref)
    decisions = {
        "component_choice": {
            "type": ["string", "null"],
            "enum": component_ids,
            "instructions": "Select the registered panel whose purpose matches the requested "
            "operation. Use null only when no registered purpose matches. Respect negation.",
            "when": user,
            "otherwise": path("state.selection"),
        },
        "entity_choice": {
            "type": ["string", "null"],
            "enum": entity_ids,
            "instructions": "Select the reference of the entry explicitly named in the user input; "
            "return null if no entry is specified. Return its reference, not its display name.",
            "when": op("and", user, op("eq", exact_ref, None)),
            "otherwise": choose(user, exact_ref, None),
        },
        "urgent": {
            "type": "boolean",
            "instructions": "Is the user explicitly asking for urgent/immediate "
            "attention or saying progress is blocked? A negated urgency statement is false.",
            "when": user,
            "otherwise": False,
        },
    }
    if kind == "generation":
        decisions["message"] = {
            "type": "string",
            "maxLength": 180,
            "instructions": "Write one short factual UI sentence in the user's language. Include "
            "the exact display name and status of the selected entry. Do not invent facts, "
            "promise completion, "
            "or mention any other entry. Use the dependency entity_choice to find the entry.",
            "depends_on": ["component_choice", "entity_choice"],
            "when": user,
            "otherwise": "",
        }
    selected = path("state.selection")
    no_selection = op("eq", selected, None)
    entity = path("decisions.entity_choice")
    count = choose(
        op("eq", entity, None),
        0,
        lookup({k: e["count"] for k, e in bp["entities"].items()}, entity),
    )
    return {
        "version": "genui-v1",
        "components": list(bp["registry"]) + ["EmptyState"],
        "registry": bp["registry"],
        "decisions": decisions,
        "before": [
            {
                "when": op("eq", path("event.type"), "reset"),
                "set": {"phase": "idle", "count": 0, "error": None, "selection": None},
            },
            {
                "when": op(
                    "and",
                    op("eq", path("event.type"), "response"),
                    op("eq", path("event.request_id"), path("facts.request_id")),
                ),
                "set": {"phase": "ready", "count": path("event.count"), "error": None},
            },
        ],
        "after": [
            {
                "when": user,
                "set": {
                    "selection": path("decisions.component_choice"),
                    "phase": choose(
                        op("eq", path("decisions.component_choice"), None), "unknown", "selected"
                    ),
                    "count": count,
                    "error": None,
                },
            }
        ],
        "view": {
            "component": choose(no_selection, "EmptyState", selected),
            "props": {
                "title": choose(
                    no_selection,
                    "Choose an operation",
                    lookup({k: r["title"] for k, r in bp["registry"].items()}, selected),
                ),
                "primary_action": choose(
                    no_selection,
                    "none",
                    lookup({k: r["action"] for k, r in bp["registry"].items()}, selected),
                ),
                "entity": choose(
                    op("eq", entity, None),
                    None,
                    lookup({k: e["name"] for k, e in bp["entities"].items()}, entity),
                ),
                "count": path("state.count"),
                "error": path("state.error"),
                "urgent": path("decisions.urgent"),
                "disabled": no_selection,
                "description": path("decisions.message") if kind == "generation" else "",
            },
        },
    }


def oracle(bp, before, event, answers):
    """Independent reference: ordinary branches, no runtime expressions or reducer calls."""
    state = dict(before)
    if event["type"] == "reset":
        state.update(phase="idle", count=0, error=None, selection=None)
    elif event["type"] == "response" and event["request_id"] == f"request-{bp['group']}":
        state.update(phase="ready", count=event["count"], error=None)
    elif event["type"] == "user":
        selected, entity = answers["component_choice"], answers["entity_choice"]
        state.update(
            selection=selected,
            phase="unknown" if selected is None else "selected",
            count=0 if entity is None else bp["entities"][entity]["count"],
            error=None,
        )
    selected = state["selection"]
    view = bp["registry"].get(selected)
    entity = answers.get("entity_choice") if event["type"] == "user" else None
    return {
        "component": selected if selected is not None else "EmptyState",
        "state": state,
        "props": {
            "title": view["title"] if view else "Choose an operation",
            "primary_action": view["action"] if view else "none",
            "entity": bp["entities"][entity]["name"] if entity else None,
            "count": state["count"],
            "error": state["error"],
            "urgent": answers.get("urgent", False) if event["type"] == "user" else False,
            "disabled": selected is None,
            "description": answers.get("message", ""),
        },
    }


def generation_request(bp, steps=None):
    tasks = []
    for case in semantic_cases(bp):
        if steps is not None and case["step"] not in steps:
            continue
        tasks.append(
            {
                "step": case["step"],
                "variation_seed": case["seed"],
                "writing_style": [
                    "a terse fragment",
                    "a polite indirect question",
                    "one concise sentence",
                    "a request explaining a practical reason",
                    "contrast with an unwanted action",
                    "a conversational follow-up",
                    "a short first-person request",
                ][case["seed"] % 7],
                "intended_purpose": bp["registry"]
                .get(case["component"], {})
                .get(
                    "purpose",
                    [
                        "Translate prose into another language, without editing entries.",
                        "Place a telephone call to a human specialist to discuss this matter.",
                        "Set a countdown timer to remind the user to take a short break.",
                        "Tell the user a short joke related to their work.",
                    ][case["seed"] % 4],
                ),
                "named_entry": bp["entities"].get(case["entity"], {}).get("name"),
                "entry_requirement": (
                    "The USER MESSAGE must contain this exact name once: "
                    + bp["entities"][case["entity"]]["name"]
                    if case["entity"] is not None
                    else "Do not name any specific entry in the user message."
                ),
                "urgent": case["urgent"],
                "urgency_requirement": (
                    "The user must explicitly demand urgent/immediate attention or say they "
                    "are blocked. Include that urgency in the user message."
                    if case["urgent"]
                    else "Not urgent. Avoid urgency; optionally explicitly negate it."
                ),
                "needs_ui_sentence": case["kind"] == "generation",
                "entry_facts": bp["entities"].get(case["entity"]),
            }
        )
    messages = [
        {
            "role": "system",
            "content": "Create realistic, diverse user messages for a UI assistant. "
            "Follow the intended operation, entry and urgency exactly in the requested language. "
            "Vary syntax, negation and polite indirect requests. Do not print panel IDs, "
            "reference IDs, answers, or annotations. Mention the named entry exactly once, "
            "or no entry if null. No other entry names. For urgent=false avoid implying urgency. "
            "For needs_ui_sentence include the exact entry name and status in a short UI summary; "
            "otherwise summary must be empty. The summary may state only the entry name and "
            "given status, with no claims about actions completed or available. "
            "Write text from the requesting user's perspective, not an assistant response. "
            "Follow each urgency_requirement explicitly. "
            "Follow writing_style; keep wording varied and avoid repeated stock phrases. "
            "Deletion means remove the entry, not undo/cancel a deletion request. "
            "Do not describe panel choices, task constraints, or registered operations. "
            "Every message must be specific to this workflow with varied realistic context, "
            "not generic stock phrases. The registry is exhaustive; an unrelated request must "
            "fall outside every listed purpose.",
        },
        {
            "role": "user",
            "content": dumps(
                {
                    "family": bp["family"],
                    "language": bp["language"],
                    "registered_purposes": [v["purpose"] for v in bp["registry"].values()],
                    "tasks": tasks,
                }
            ),
        },
    ]
    if len(tasks) == 1:
        messages[-1]["content"] += (
            "\nWrite ONE user request whose main requested action is: "
            + tasks[0]["intended_purpose"]
            + " Do not perform that action yourself: ask for it. For example a joke task must "
            "ask someone to tell a joke, not contain a joke or ask to edit an entry. "
            "Urgency is secondary to this explicit main action. Include concrete workflow "
            "details so the wording is distinct even after the entry name is removed."
        )
        if bp["cases"][tasks[0]["step"]]["component"] is None:
            messages[-1]["content"] += (
                " The main request is outside the registered UI. Put the unsupported action "
                "in the main clause. Urgency refers to that action, not to fixing, removing, "
                "viewing or editing the named entry. Do not say you are blocked on the entry."
                " The entry name supplies background context only. If translating prose, "
                "include a distinct short quotation or message to translate and explicitly "
                "leave the stored entry unchanged. Do not translate the entry itself or "
                "its title, description, notes or documentation: that can imply editing."
            )
    item = object_schema(
        {
            "step": {"type": "integer"},
            "text": {"type": "string", "minLength": 10, "maxLength": 500},
            "summary": {"type": "string", "maxLength": 180},
        }
    )
    schema = object_schema(
        {"cases": {"type": "array", "items": item, "minItems": len(tasks), "maxItems": len(tasks)}}
    )
    return messages, schema


def validate_generated(bp, value, steps=None):
    cases = [c for c in semantic_cases(bp) if steps is None or c["step"] in steps]
    if [c["step"] for c in value["cases"]] != [c["step"] for c in cases]:
        raise ValueError("Wrong scenario steps")
    seen = set()
    for case, generated in zip(cases, value["cases"], strict=True):
        text = generated["text"]
        if text in seen or any(key in text for key in list(bp["registry"]) + list(bp["entities"])):
            raise ValueError("Duplicate text or leaked answer ID")
        seen.add(text)
        for key, entry in bp["entities"].items():
            if text.count(entry["name"]) != int(key == case["entity"]):
                raise ValueError("Entry grounding mismatch")
        if case["kind"] == "generation":
            entry = bp["entities"][case["entity"]]
            if any(entry[key] not in generated["summary"] for key in ("name", "status")):
                raise ValueError("Ungrounded UI sentence")
        elif generated["summary"]:
            raise ValueError("Unexpected generated sentence")


def records(bp, generated):
    validate_generated(bp, generated)
    descriptions = {x["step"]: x for x in generated["cases"]}
    before = {"phase": "idle", "count": 3, "error": 'quoted "error"\nmessage', "selection": None}
    result = []
    for case in bp["cases"]:
        step, kind = case["step"], case["kind"]
        event = {"type": "user", "request_id": f"request-{bp['group']}"}
        if step == 7 and bp["group"] % 3 == 0:
            event.update(type="reset")
        elif kind == "code":
            event.update(
                type="response",
                count=bp["group"] % 13,
                request_id=f"request-{bp['group']}" if step == 7 else "stale",
            )
        answers = {
            "component_choice": case["component"],
            "entity_choice": case["entity"],
            "urgent": case["urgent"],
        }
        text = descriptions[step]["text"] if kind != "code" else "Apply this structured event."
        if kind == "generation":
            answers["message"] = descriptions[step]["summary"]
        spec = specification(bp, kind)
        facts = {"request_id": f"request-{bp['group']}", "entries": bp["entities"]}
        expected = oracle(bp, before, event, answers if kind != "code" else {})
        item = {
            "id": f"shared-{bp['group']:04}-{step}",
            "group": bp["group"],
            "step": step,
            "split": bp["split"],
            "family": bp["family"],
            "language": bp["language"],
            "kind": kind,
            "spec": spec,
            "context": text,
            "state": dict(before),
            "event": event,
            "facts": facts,
            "answers": answers if kind != "code" else {},
            "expected": expected,
        }
        result.append(item)
        before = expected["state"]
    return result


def semantic_cases(bp):
    return [c for c in bp["cases"] if c["kind"] != "code"]


def training_record(item):
    spec, env = ui_environment(
        compile_ui(item["spec"]), item["context"], item["state"], item["event"], item["facts"]
    )
    return {
        "id": item["id"],
        "context": model_context(env),
        "questions": {
            n: {k: v for k, v in f.items() if k not in {"when", "otherwise"}}
            for n, f in spec["decisions"].items()
        },
        "answers": item["answers"],
        "model_fields": [
            name
            for name, field in spec["decisions"].items()
            if expression(field.get("when", True), env)
        ],
    }


def allocation(seed=20260929, teacher="qwen/qwen3-235b-a22b-2507", budget_usd=20):
    return {
        "version": VERSION,
        "seed": seed,
        "records": 10000,
        "groups": 1000,
        "split_groups": {"train": 800, "validation": 100, "test": 100},
        "family_hash": digest(FAMILIES),
        "kind_per_group": KINDS,
        "teacher": teacher,
        "budget_usd": budget_usd,
    }
