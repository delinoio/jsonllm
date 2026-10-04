"""Frozen synthetic UI workflows with an independent deterministic state reducer.

The model predicts scalar decisions; trusted code assembles component/props/state.
These fixtures are evaluation-only and contain no training examples.
"""

import copy
import random

from .io import dumps
from .schema import validate_record

COMPONENTS = [
    "EditForm",
    "LoadingPanel",
    "ErrorAlert",
    "EmptyState",
    "ResultList",
    "DataCard",
    "ConfirmDialog",
    "SuccessNotice",
]
ACTIONS = ["search", "submit", "delete", "confirm", "retry", "upload", "next", "none"]
VIEWS = {
    "search": {
        "idle": ("EditForm", "search"),
        "loading": ("LoadingPanel", "none"),
        "empty": ("EmptyState", "search"),
        "results": ("ResultList", "search"),
        "error": ("ErrorAlert", "retry"),
    },
    "checkout": {
        "editing": ("EditForm", "submit"),
        "submitting": ("LoadingPanel", "none"),
        "error": ("ErrorAlert", "retry"),
        "complete": ("SuccessNotice", "none"),
    },
    "delete": {
        "viewing": ("DataCard", "delete"),
        "confirming": ("ConfirmDialog", "confirm"),
        "deleting": ("LoadingPanel", "none"),
        "error": ("ErrorAlert", "retry"),
        "deleted": ("SuccessNotice", "none"),
    },
    "upload": {
        "ready": ("EditForm", "upload"),
        "uploading": ("LoadingPanel", "none"),
        "error": ("ErrorAlert", "retry"),
        "done": ("SuccessNotice", "none"),
    },
    "pagination": {
        "ready": ("ResultList", "next"),
        "loading": ("LoadingPanel", "none"),
        "error": ("ErrorAlert", "retry"),
    },
}
RULES = {
    "search": "submit in idle/empty/results or retry in error starts loading, resets count to "
    "0 and clears error. received in loading sets count to event.count and phase to empty if "
    "count=0, otherwise results. failure in loading sets error phase, count=0, and event.message.",
    "checkout": "submit in editing starts submitting only when facts.form_valid=true. "
    "failure in submitting sets error phase and event.message. retry in error starts submitting "
    "and clears error. success in submitting sets complete and clears error. Preserve count.",
    "delete": "open in viewing sets confirming; cancel in confirming returns viewing; confirm "
    "in confirming sets deleting. failure in deleting sets error phase and event.message. "
    "retry in error sets deleting and clears error. success in deleting sets deleted, count=0 "
    "and clears error. Otherwise preserve count.",
    "upload": "start in ready starts uploading only when facts.file_valid=true, resetting "
    "count=0 and error=null. progress in uploading sets count to event.percent clamped to "
    "0..100. cancel in uploading returns ready with count=0 and error=null. failure in uploading "
    "sets error phase and event.message. retry in error sets uploading, count=0, error=null. "
    "success in uploading sets done, count=100, error=null.",
    "pagination": "count is the current page. next in ready when count<facts.max_page adds 1 "
    "to count and starts loading. previous in ready when count>1 subtracts 1 and starts loading. "
    "received in loading sets ready. failure in loading sets error phase and event.message. "
    "retry in error sets loading and clears error, preserving count. "
    "Valid navigation clears error.",
}
SHARED = (
    "Apply exactly one event. Any transition not listed is a no-op: preserve phase/count/error. "
    "For received/failure/success/progress, ignore the entire event if event.request_id differs "
    "from facts.request_id, even if the event otherwise looks valid. Never infer success before "
    "a valid success event. Non-error transitions clear error where specified. "
    "Use the view registry for the NEW phase; copy its title exactly. "
    "disabled=true if action=none, or checkout editing with form_valid=false, or upload ready "
    "with file_valid=false, or pagination ready at max_page. Otherwise disabled=false."
)


def reduce_state(flow, state, event, facts):
    """Gold reducer; never called to repair model outputs or supply rollout inputs."""
    out = copy.deepcopy(state)
    phase, kind = state["phase"], event["type"]
    if kind in {"received", "failure", "success", "progress"}:
        if event.get("request_id") != facts["request_id"]:
            return out
    if flow == "search":
        if (kind == "submit" and phase in {"idle", "empty", "results"}) or (
            kind == "retry" and phase == "error"
        ):
            out.update(phase="loading", count=0, error=None)
        elif kind == "received" and phase == "loading":
            count = event["count"]
            out.update(phase="results" if count else "empty", count=count, error=None)
        elif kind == "failure" and phase == "loading":
            out.update(phase="error", count=0, error=event["message"])
    elif flow == "checkout":
        if (kind == "submit" and phase == "editing" and facts["form_valid"]) or (
            kind == "retry" and phase == "error"
        ):
            out.update(phase="submitting", error=None)
        elif kind == "failure" and phase == "submitting":
            out.update(phase="error", error=event["message"])
        elif kind == "success" and phase == "submitting":
            out.update(phase="complete", error=None)
    elif flow == "delete":
        target = {
            ("viewing", "open"): "confirming",
            ("confirming", "cancel"): "viewing",
            ("confirming", "confirm"): "deleting",
            ("error", "retry"): "deleting",
        }
        if (phase, kind) in target:
            out.update(phase=target[phase, kind], error=None)
        elif phase == "deleting" and kind == "failure":
            out.update(phase="error", error=event["message"])
        elif phase == "deleting" and kind == "success":
            out.update(phase="deleted", count=0, error=None)
    elif flow == "upload":
        if (kind == "start" and phase == "ready" and facts["file_valid"]) or (
            kind == "retry" and phase == "error"
        ):
            out.update(phase="uploading", count=0, error=None)
        elif phase == "uploading":
            if kind == "progress":
                out["count"] = max(0, min(100, event["percent"]))
            elif kind == "cancel":
                out.update(phase="ready", count=0, error=None)
            elif kind == "failure":
                out.update(phase="error", error=event["message"])
            elif kind == "success":
                out.update(phase="done", count=100, error=None)
    elif flow == "pagination":
        if phase == "ready" and kind in {"next", "previous"}:
            delta = 1 if kind == "next" else -1
            page = state["count"] + delta
            if 1 <= page <= facts["max_page"]:
                out.update(phase="loading", count=page, error=None)
        elif phase == "loading" and kind == "received":
            out.update(phase="ready", error=None)
        elif phase == "loading" and kind == "failure":
            out.update(phase="error", error=event["message"])
        elif phase == "error" and kind == "retry":
            out.update(phase="loading", error=None)
    else:
        raise ValueError("Unknown flow")
    return out


def flat_view(flow, state, facts, registry):
    component, action, title = registry[state["phase"]]
    disabled = (
        action == "none"
        or (flow == "checkout" and state["phase"] == "editing" and not facts["form_valid"])
        or (flow == "upload" and state["phase"] == "ready" and not facts["file_valid"])
        or (
            flow == "pagination"
            and state["phase"] == "ready"
            and state["count"] >= facts["max_page"]
        )
    )
    return state | {
        "component": component,
        "title": title,
        "primary_action": action,
        "disabled": disabled,
    }


def assemble(values):
    """Pure shape mapping: no corrections, inferred values, defaults, or gold lookups."""
    return {
        "component": values["component"],
        "props": {k: values[k] for k in ["title", "primary_action", "disabled", "count", "error"]},
        "state": {k: values[k] for k in ["phase", "count", "error"]},
    }


def make_record(trajectory, index, state):
    step, flow = trajectory["steps"][index], trajectory["flow"]
    registry = trajectory["registry"]
    context = (
        "Choose a UI component and update its props/state for a fixed catalog.\n"
        f"Workflow: {flow}.\nRules: {RULES[flow]}\n{SHARED}\n"
        "View registry (phase: [component, primary_action, exact title]): "
        + dumps(registry)
        + "\nCurrent state: "
        + dumps(state)
        + "\nFacts: "
        + dumps(step["facts"])
        + "\nEvent: "
        + dumps(step["event"])
    )
    rng = random.Random(trajectory["seed"] + index)
    components, actions, phases = list(COMPONENTS), list(ACTIONS), list(registry)
    for options in [components, actions, phases]:
        rng.shuffle(options)
    questions = {
        "phase": {
            "type": "string",
            "enum": phases,
            "instructions": "Return the NEW state phase after applying this one event.",
        },
        "count": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "instructions": "Return the NEW numeric count/page/progress after this event.",
            "depends_on": ["phase"],
        },
        "error": {
            "type": ["string", "null"],
            "maxLength": 100,
            "instructions": "Return the NEW error string exactly, or null when no error.",
            "depends_on": ["phase"],
        },
        "component": {
            "type": "string",
            "enum": components,
            "instructions": "Select the component in the registry for the NEW phase.",
            "depends_on": ["phase"],
        },
        "title": {
            "type": "string",
            "maxLength": 100,
            "instructions": "Copy the exact title from the registry for the NEW phase.",
            "depends_on": ["phase", "component"],
        },
        "primary_action": {
            "type": "string",
            "enum": actions,
            "instructions": "Select the registry primary_action for the NEW phase.",
            "depends_on": ["phase"],
        },
        "disabled": {
            "type": "boolean",
            "instructions": "Apply the disabled rule to the NEW UI.",
            "depends_on": ["phase", "primary_action", "count"],
        },
    }
    return {"id": f"{trajectory['id']}-{index}", "context": context, "questions": questions}


def build_suite():
    trajectories = []
    for n, flow in enumerate(VIEWS):
        for variant in range(2):
            request_id = f"req-{n}-{variant}"
            message = 'Request failed: "timeout"' if variant == 0 else '요청 실패: "시간 초과"'
            facts = {
                "request_id": request_id,
                "form_valid": True,
                "file_valid": True,
                "max_page": 3,
            }

            def event(kind, _request_id=request_id, **extra):
                return {"type": kind, "request_id": _request_id, **extra}

            initial = {"phase": list(VIEWS[flow])[0], "count": 0, "error": None}
            if flow == "search":
                events = [
                    event("submit"),
                    event("received", count=0),
                    event("submit"),
                    event("failure", message=message),
                    event("retry"),
                    event("received", count=7 + variant),
                ]
            elif flow == "checkout":
                initial["count"] = 2 + variant
                events = [
                    event("submit"),
                    event("submit"),
                    event("failure", message=message),
                    event("retry"),
                    event("success"),
                    event("submit"),
                ]
            elif flow == "delete":
                initial["count"] = 1
                events = [
                    event("open"),
                    event("cancel"),
                    event("open"),
                    event("confirm"),
                    event("success") if not variant else event("failure", message=message),
                    event("success") if not variant else event("retry"),
                ]
            elif flow == "upload":
                events = [
                    event("start"),
                    event("progress", percent=140),
                    event("failure", message=message),
                    event("retry"),
                    event("success"),
                    event("progress", percent=20),
                ]
                if variant:
                    events = [
                        event("start"),
                        event("start"),
                        event("progress", percent=-5),
                        event("cancel"),
                        event("success"),
                        event("start"),
                    ]
            else:
                initial["count"] = 1 if not variant else 3
                events = [
                    event("previous"),
                    event("next"),
                    event("received"),
                    event("next"),
                    event("received", request_id="stale-response"),
                    event("failure", message=message),
                ]
                if variant:
                    events = [
                        event("next"),
                        event("previous"),
                        event("received"),
                        event("previous"),
                        event("failure", message=message),
                        event("retry"),
                    ]
            registry = {
                phase: [
                    component,
                    action,
                    f"{flow}: {phase}" if not variant else f'화면 "{flow}"\n{phase}',
                ]
                for phase, (component, action) in VIEWS[flow].items()
            }
            trajectory = {
                "id": f"{flow}-{'en' if not variant else 'ko'}",
                "flow": flow,
                "variant": variant,
                "seed": 9400 + n * 10 + variant,
                "initial_state": initial,
                "registry": registry,
                "steps": [],
            }
            state = copy.deepcopy(initial)
            for i, ev in enumerate(events):
                step_facts = facts | {
                    "form_valid": not (flow == "checkout" and i == 0),
                    "file_valid": not (flow == "upload" and variant and i == 0),
                }
                next_state = reduce_state(flow, state, ev, step_facts)
                expected = flat_view(flow, next_state, step_facts, registry)
                trajectory["steps"].append(
                    {"event": ev, "facts": step_facts, "gold_before": state, "expected": expected}
                )
                state = next_state
            trajectories.append(trajectory)
    for trajectory in trajectories:
        for i, step in enumerate(trajectory["steps"]):
            validate_record(
                make_record(trajectory, i, step["gold_before"]) | {"answers": step["expected"]}
            )
    return {
        "version": "genui-eval-v1",
        "components": COMPONENTS,
        "evaluation_only": True,
        "trajectories": trajectories,
    }
