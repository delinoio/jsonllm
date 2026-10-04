"""Example registered UI rules; no benchmark oracle is imported or called here."""


def reference(name):
    return {"path": name}


def operation(name, *args):
    return {"op": name, "args": list(args)}


def workflow_spec(flow, registry):
    phase, event = reference("state.phase"), reference("event.type")

    def equal(a, b):
        return operation("eq", a, b)

    def when(phases, kind, **updates):
        return {
            "when": operation("and", operation("in", phase, phases), equal(event, kind)),
            "set": updates,
        }

    def guard(rule, condition):
        rule["when"] = operation("and", rule["when"], condition)
        return rule

    if flow == "search":
        rules = [
            when(["idle", "empty", "results"], "submit", phase="loading", count=0, error=None),
            when(["error"], "retry", phase="loading", count=0, error=None),
            when(
                ["loading"],
                "received",
                count=reference("event.count"),
                error=None,
                phase={
                    "if": equal(reference("event.count"), 0),
                    "then": "empty",
                    "else": "results",
                },
            ),
            when(["loading"], "failure", phase="error", count=0, error=reference("event.message")),
        ]
    elif flow == "checkout":
        rules = [
            guard(
                when(["editing"], "submit", phase="submitting", error=None),
                reference("facts.form_valid"),
            ),
            when(["error"], "retry", phase="submitting", error=None),
            when(["submitting"], "failure", phase="error", error=reference("event.message")),
            when(["submitting"], "success", phase="complete", error=None),
        ]
    elif flow == "delete":
        rules = [
            when([old], kind, phase=new, error=None)
            for old, kind, new in (
                ("viewing", "open", "confirming"),
                ("confirming", "cancel", "viewing"),
                ("confirming", "confirm", "deleting"),
                ("error", "retry", "deleting"),
            )
        ]
        rules += [
            when(["deleting"], "failure", phase="error", error=reference("event.message")),
            when(["deleting"], "success", phase="deleted", count=0, error=None),
        ]
    elif flow == "upload":
        rules = [
            guard(
                when(["ready"], "start", phase="uploading", count=0, error=None),
                reference("facts.file_valid"),
            ),
            when(["error"], "retry", phase="uploading", count=0, error=None),
            when(
                ["uploading"],
                "progress",
                count=operation("max", 0, operation("min", 100, reference("event.percent"))),
            ),
            when(["uploading"], "cancel", phase="ready", count=0, error=None),
            when(["uploading"], "failure", phase="error", error=reference("event.message")),
            when(["uploading"], "success", phase="done", count=100, error=None),
        ]
    elif flow == "pagination":
        rules = [
            guard(
                when(
                    ["ready"],
                    "next",
                    phase="loading",
                    error=None,
                    count=operation("add", reference("state.count"), 1),
                ),
                operation("lt", reference("state.count"), reference("facts.max_page")),
            ),
            guard(
                when(
                    ["ready"],
                    "previous",
                    phase="loading",
                    error=None,
                    count=operation("sub", reference("state.count"), 1),
                ),
                operation("gt", reference("state.count"), 1),
            ),
            when(["loading"], "received", phase="ready", error=None),
            when(["loading"], "failure", phase="error", error=reference("event.message")),
            when(["error"], "retry", phase="loading", error=None),
        ]
    else:
        raise ValueError("Unregistered workflow")
    for rule in rules:
        # Guard precedes all event-specific attribute reads; missing/old events never execute.
        rule["when"] = operation(
            "and",
            operation(
                "or",
                operation(
                    "not", operation("in", event, ["received", "failure", "success", "progress"])
                ),
                equal(reference("event.request_id"), reference("facts.request_id")),
            ),
            rule["when"],
        )

    def value(index):
        return {"lookup": {k: v[index] for k, v in registry.items()}, "key": phase}

    extra = False
    if flow == "checkout":
        extra = operation(
            "and", equal(phase, "editing"), operation("not", reference("facts.form_valid"))
        )
    elif flow == "upload":
        extra = operation(
            "and", equal(phase, "ready"), operation("not", reference("facts.file_valid"))
        )
    elif flow == "pagination":
        extra = operation(
            "and",
            equal(phase, "ready"),
            operation("ge", reference("state.count"), reference("facts.max_page")),
        )
    return {
        "version": "genui-v1",
        "components": sorted({v[0] for v in registry.values()}),
        "before": rules,
        "view": {
            "component": value(0),
            "props": {
                "primary_action": value(1),
                "title": value(2),
                "disabled": operation("or", equal(value(1), "none"), extra),
                "count": reference("state.count"),
                "error": reference("state.error"),
            },
        },
    }
