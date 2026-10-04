"""Explicit, portable GenUI rules. Model outputs never execute code or repair state."""

import copy
import math
import time
from dataclasses import dataclass

from jsonschema.exceptions import ValidationError

from .diagnostics import exception_details
from .io import digest, dumps
from .schema import dependency_order, validate_questions, validate_value

OPS = {
    "eq": lambda a, b: type(a) is type(b) and a == b,
    "ne": lambda a, b: type(a) is not type(b) or a != b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "min": min,
    "max": max,
    "not": lambda a: not a,
    "in": lambda a, b: a in b,
}
ARITY = dict.fromkeys(OPS, 2) | {"not": 1}
ROOTS = {"context", "state", "event", "facts", "decisions", "registry"}


class _MissingInput(ValueError):
    def __init__(self, path):
        super().__init__(f"Missing UI input path: {path}")
        self.path = path


def expression(node, env):
    """Evaluate a closed JSON expression language; no eval, imports, or callbacks."""
    if isinstance(node, list):
        return [expression(x, env) for x in node]
    if not isinstance(node, dict):
        return node
    if set(node) == {"literal"}:
        return copy.deepcopy(node["literal"])
    if set(node) == {"path"}:
        value = env
        for key in node["path"].split("."):
            if not isinstance(value, dict) or key not in value:
                raise _MissingInput(node["path"])
            value = value[key]
        return copy.deepcopy(value)
    if set(node) == {"lookup", "key"}:
        table, key = expression(node["lookup"], env), expression(node["key"], env)
        if not isinstance(table, dict) or not isinstance(key, str) or key not in table:
            raise ValueError("Unknown registry key")
        return copy.deepcopy(table[key])
    if set(node) == {"if", "then", "else"}:
        return expression(node["then"] if expression(node["if"], env) else node["else"], env)
    if set(node) == {"template", "values"}:
        values = {k: expression(v, env) for k, v in node["values"].items()}
        return node["template"].format_map(values)
    if set(node) == {"op", "args"}:
        op, args = node["op"], node["args"]
        if op == "and":
            return all(expression(a, env) for a in args)
        if op == "or":
            return any(expression(a, env) for a in args)
        return OPS[op](*(expression(a, env) for a in args))
    return {k: expression(v, env) for k, v in node.items()}


def _check(node):
    if isinstance(node, float) and not math.isfinite(node):
        raise ValueError("Nonfinite UI literal")
    if isinstance(node, list):
        for value in node:
            _check(value)
    if not isinstance(node, dict):
        return
    if "literal" in node:
        if set(node) != {"literal"}:
            raise ValueError("Literal cannot have other operators")
        dumps(node["literal"])
        return
    if "path" in node:
        if set(node) != {"path"} or not isinstance(node["path"], str):
            raise ValueError("Invalid path expression")
        if node["path"].split(".")[0] not in ROOTS:
            raise ValueError("Unknown input root")
        return
    if "op" in node:
        if set(node) != {"op", "args"} or not isinstance(node["args"], list):
            raise ValueError("Invalid operator expression")
        if node["op"] not in OPS and node["op"] not in {"and", "or"}:
            raise ValueError("Unknown UI operator")
        if node["op"] in ARITY and len(node["args"]) != ARITY[node["op"]]:
            raise ValueError("Wrong operator arity")
    if "if" in node and set(node) != {"if", "then", "else"}:
        raise ValueError("Conditional requires if, then, else")
    if "lookup" in node and set(node) != {"lookup", "key"}:
        raise ValueError("Lookup requires a table and key")
    if "template" in node:
        import string

        if set(node) != {"template", "values"} or not isinstance(node["values"], dict):
            raise ValueError("Invalid template")
        for _, name, fmt, conversion in string.Formatter().parse(node["template"]):
            if name is not None and (
                not name.isidentifier() or name not in node["values"] or fmt or conversion
            ):
                raise ValueError("Templates accept only declared plain placeholders")
    for value in node.values():
        _check(value)


@dataclass(frozen=True)
class CompiledUI:
    # Store immutable source rather than exposing a caller-mutable compiled registry.
    source: str
    fingerprint: str


def compile_ui(spec):
    allowed = {"version", "components", "registry", "decisions", "before", "after", "view"}
    if not isinstance(spec, dict) or set(spec) - allowed or spec.get("version") != "genui-v1":
        raise ValueError("Expected a genui-v1 specification")
    if not isinstance(spec.get("components"), list) or not spec["components"]:
        raise ValueError("A component allowlist is required")
    if any(not isinstance(c, str) or not c for c in spec["components"]):
        raise ValueError("Component names must be strings")
    if len(set(spec["components"])) != len(spec["components"]):
        raise ValueError("Duplicate component")
    if set(spec.get("view", {})) != {"component", "props"}:
        raise ValueError("View must declare component and props")
    questions = {}
    for name, field in spec.get("decisions", {}).items():
        questions[name] = {k: v for k, v in field.items() if k not in {"when", "otherwise"}}
        if ("when" in field) != ("otherwise" in field):
            raise ValueError("Conditional decisions require an explicit otherwise value")
        for key in ("when", "otherwise"):
            if key in field:
                _check(field[key])
    if questions:
        validate_questions(questions)
    for stage in ("before", "after"):
        for transition in spec.get(stage, []):
            if set(transition) != {"when", "set"} or not isinstance(transition["set"], dict):
                raise ValueError("Transitions require when and set")
            if any(not isinstance(k, str) or not k or "." in k for k in transition["set"]):
                raise ValueError("Transitions update declared top-level state keys")
            _check(transition)
            if stage == "before" and '"decisions.' in dumps(transition):
                raise ValueError("Before transitions cannot depend on model decisions")
    _check(spec["view"])
    source = dumps(spec)
    return CompiledUI(source, digest(spec))


def _transition(transitions, env):
    for transition in transitions:
        if expression(transition["when"], env):
            # All assignments read the same pre-transition state; exactly one rule fires.
            updates = {k: expression(v, env) for k, v in transition["set"].items()}
            if any(k not in env["state"] for k in updates):
                raise ValueError("Transition writes an undeclared state key")
            env["state"].update(updates)
            break


def ui_environment(compiled, context, state, event, facts):
    import json

    spec = json.loads(compiled.source)
    if not isinstance(context, str) or any(not isinstance(x, dict) for x in (state, event, facts)):
        raise ValueError("UI input requires text context and object state/event/facts")
    env = copy.deepcopy(
        {
            "context": context,
            "state": state,
            "event": event,
            "facts": facts,
            "registry": spec.get("registry", {}),
            "decisions": {},
        }
    )
    dumps(env)
    _transition(spec.get("before", []), env)
    return spec, env


def model_context(env):
    return (
        "User input:\n"
        + env["context"]
        + "\nUI inputs (JSON):\n"
        + dumps({k: env[k] for k in ("registry", "state", "event", "facts")})
    )


def _model_decision_count(pending, fields, env):
    count = 0
    for name in pending:
        field = fields[name]
        try:
            needed = "when" not in field or bool(expression(field["when"], env))
        except _MissingInput as exc:
            if not exc.path.startswith("decisions."):
                raise
            # A guard depending on a future prediction can still require a model branch.
            needed = True
        count += needed
    return count


def run_ui(compiled, context, state, event, facts, *, predictor=None, max_tokens=64):
    """Return {output, diagnostics}; an error returns output=None, never a repaired UI."""
    started = time.perf_counter()
    try:
        spec, env = ui_environment(compiled, context, state, event, facts)
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        return {
            "output": None,
            "diagnostics": {
                "spec_hash": compiled.fingerprint,
                "model_fields": [],
                "errors": {"runtime": str(exc)},
                "exception": exception_details(exc),
                "seconds": time.perf_counter() - started,
            },
        }
    fields = spec.get("decisions", {})
    questions = {
        n: {k: v for k, v in f.items() if k not in {"when", "otherwise"}} for n, f in fields.items()
    }
    pending = dependency_order(questions) if questions else []
    diagnostics = {"spec_hash": compiled.fingerprint, "model_fields": [], "errors": {}}
    session = None
    common = None
    try:
        while pending:
            ready = [
                n
                for n in pending
                if all(p in env["decisions"] for p in questions[n].get("depends_on", []))
            ]
            if not ready:
                raise ValueError("Unresolved UI decision dependencies")
            modeled = []
            for name in ready:
                field = fields[name]
                if "when" in field and not expression(field["when"], env):
                    value = expression(field["otherwise"], env)
                    validate_value(questions[name], value)
                    env["decisions"][name] = value
                    pending.remove(name)
                else:
                    modeled.append(name)
            if modeled:
                if predictor is None:
                    raise ValueError("This UI request requires a model predictor")
                if session is None:
                    # One handle per record; later waves reuse this immutable common context.
                    common = model_context(env)
                    session = predictor.open_record(
                        common, _model_decision_count(pending, fields, env)
                    )
                record = {
                    "id": compiled.fingerprint,
                    "context": common,
                    "questions": questions,
                }
                predictions = session.predict(record, modeled, env["decisions"], max_tokens)
                if set(predictions) != set(modeled):
                    raise ValueError("Incomplete model decision batch")
                for name in modeled:
                    validate_value(questions[name], predictions[name])
                    env["decisions"][name] = predictions[name]
                    pending.remove(name)
                diagnostics["model_fields"].extend(modeled)
        _transition(spec.get("after", []), env)
        output = {
            "component": expression(spec["view"]["component"], env),
            "props": expression(spec["view"]["props"], env),
            "state": env["state"],
        }
        if output["component"] not in spec["components"] or not isinstance(output["props"], dict):
            raise ValueError("UI output violates its component/props contract")
        dumps(output)
    except (ValueError, TypeError, KeyError, ArithmeticError, ValidationError) as exc:
        diagnostics["errors"]["runtime"] = str(exc)
        diagnostics["exception"] = exception_details(exc)
        output = None
    finally:
        if session is not None:
            session.close()
            diagnostics["model"] = session.metrics
        diagnostics["seconds"] = time.perf_counter() - started
    return {"output": output, "diagnostics": diagnostics}
