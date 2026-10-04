import copy

import pytest

from jsonllm.ui import compile_ui, expression, run_ui


def spec():
    return {
        "version": "genui-v1",
        "components": ["List", "Spinner"],
        "before": [
            {
                "when": {"op": "eq", "args": [{"path": "event.type"}, "submit"]},
                "set": {"phase": "loading", "count": 0, "error": None},
            }
        ],
        "view": {
            "component": {
                "if": {"op": "eq", "args": [{"path": "state.phase"}, "loading"]},
                "then": "Spinner",
                "else": "List",
            },
            "props": {"count": {"path": "state.count"}, "error": {"path": "state.error"}},
        },
    }


def test_code_only_transition_is_atomic_and_does_not_mutate_inputs():
    source = spec()
    compiled = compile_ui(source)
    source["before"].clear()
    state = {"phase": "idle", "count": 5, "error": "failure"}
    actual = run_ui(compiled, "", state, {"type": "submit"}, {})
    assert actual["output"] == {
        "component": "Spinner",
        "props": {"count": 0, "error": None},
        "state": {"phase": "loading", "count": 0, "error": None},
    }
    assert state["count"] == 5
    assert actual["diagnostics"]["model_fields"] == []


def test_unknown_event_is_noop_and_missing_input_is_not_invented():
    state = {"phase": "idle", "count": 5, "error": None}
    assert run_ui(compile_ui(spec()), "", state, {"type": "old"}, {})["output"]["state"] == state
    assert run_ui(compile_ui(spec()), "", state, {}, {})["output"] is None


def test_model_decisions_dependencies_and_release():
    source = spec()
    source["decisions"] = {
        "x": {"type": "boolean", "instructions": "Choose."},
        "y": {"type": "string", "instructions": "Write.", "depends_on": ["x"]},
    }
    source["view"]["props"]["title"] = {"path": "decisions.y"}

    class Session:
        metrics = {}
        closed = False

        def predict(self, record, names, answers, max_tokens):
            if names == ["x"]:
                assert answers == {}
                return {"x": False}
            assert answers == {"x": False}
            return {"y": 'a "quoted" title'}

        def close(self):
            self.closed = True

    class Model:
        calls = 0
        session = Session()

        def open_record(self, context, count):
            self.calls += 1
            assert count == 2
            return self.session

    model = Model()
    result = run_ui(
        compile_ui(source),
        "hello",
        {"phase": "idle", "count": 0, "error": None},
        {"type": "other"},
        {},
        predictor=model,
    )
    assert result["output"]["props"]["title"] == 'a "quoted" title'
    assert model.calls == 1 and model.session.closed


def test_conditional_model_can_be_skipped():
    source = spec()
    source["decisions"] = {
        "x": {"type": "boolean", "instructions": "Choose.", "when": False, "otherwise": False}
    }
    result = run_ui(
        compile_ui(source), "", {"phase": "idle", "count": 0, "error": None}, {"type": "other"}, {}
    )
    assert result["output"] is not None and not result["diagnostics"]["model_fields"]


def test_dependent_code_copy_does_not_turn_one_model_decision_into_extra_prefill():
    source = spec()
    source["decisions"] = {
        "choice": {"type": "boolean", "instructions": "Choose."},
        "copy": {
            "type": "boolean",
            "instructions": "Copy.",
            "depends_on": ["choice"],
            "when": False,
            "otherwise": {"path": "decisions.choice"},
        },
    }
    source["view"]["props"]["copied"] = {"path": "decisions.copy"}

    class Predictor:
        metrics = {}
        closed = False

        def open_record(self, context, count):
            assert count == 1  # Shared CUDA must use its one-forward path.
            return self

        def predict(self, record, names, answers, max_tokens):
            assert names == ["choice"]
            return {"choice": False}

        def close(self):
            self.closed = True

    predictor = Predictor()
    result = run_ui(
        compile_ui(source),
        "",
        {"phase": "idle", "count": 0, "error": None},
        {"type": "other"},
        {},
        predictor=predictor,
    )
    assert result["output"]["props"]["copied"] is False
    assert result["diagnostics"]["model_fields"] == ["choice"] and predictor.closed


def test_three_model_dependency_waves_keep_one_session_and_common_context():
    source = spec()
    source["decisions"] = {
        "a": {"type": "boolean", "instructions": "First."},
        "b": {"type": "boolean", "instructions": "Second.", "depends_on": ["a"]},
        "c": {
            "type": "boolean",
            "instructions": "Third.",
            "depends_on": ["b"],
            "when": {"path": "decisions.b"},
            "otherwise": False,
        },
    }

    class Predictor:
        metrics = {}
        waves = []
        opened = False

        def open_record(self, context, count):
            assert count == 3 and not self.opened
            self.opened, self.context = True, context
            return self

        def predict(self, record, names, answers, max_tokens):
            assert record["context"] == self.context
            assert names == ["abc"[len(self.waves)]]
            self.waves.append(names[0])
            return {names[0]: True}

        def close(self):
            self.opened = False

    predictor = Predictor()
    result = run_ui(
        compile_ui(source),
        "",
        {"phase": "idle", "count": 0, "error": None},
        {"type": "other"},
        {},
        predictor=predictor,
    )
    assert result["output"] is not None
    assert predictor.waves == ["a", "b", "c"] and not predictor.opened


@pytest.mark.parametrize(
    "bad",
    [
        {"op": "eval", "args": ["x"]},
        {"path": "os.environ"},
        {"template": "{x.__class__}", "values": {"x": 0}},
    ],
)
def test_reject_executable_or_invalid_expressions(bad):
    source = copy.deepcopy(spec())
    source["view"]["props"]["bad"] = bad
    with pytest.raises(ValueError):
        compile_ui(source)


def test_null_false_zero_and_short_circuit():
    assert not expression({"op": "eq", "args": [False, 0]}, {})
    assert expression({"op": "and", "args": [False, {"path": "event.missing"}]}, {}) is False
    assert expression({"template": "{n}:{v}", "values": {"n": 0, "v": False}}, {}) == "0:False"


@pytest.mark.parametrize("fail_before_session", [True, False])
def test_caught_errors_retain_details_and_release_session(fail_before_session):
    class Session:
        metrics = {"test": True}
        closed = False

        def predict(self, *args):
            secret_local = "do-not-dump-frame-locals"
            assert secret_local
            raise ValueError("invalid generation details")

        def close(self):
            self.closed = True

    class Predictor:
        session = Session()

        def open_record(self, *args):
            return self.session

    source = spec()
    source["decisions"] = {"x": {"type": "boolean", "instructions": "Choose"}}
    model = Predictor()
    event = {} if fail_before_session else {"type": "other"}
    result = run_ui(
        compile_ui(source), "context", {"phase": "idle", "count": 1}, event, {}, predictor=model
    )
    assert result["output"] is None
    diag = result["diagnostics"]
    assert diag["exception"]["message"] == diag["errors"]["runtime"]
    assert diag["exception"]["type"] in {"ValueError", "_MissingInput"}
    assert "Traceback" in diag["exception"]["traceback"]
    assert "do-not-dump-frame-locals" not in diag["exception"]["traceback"]
    assert model.session.closed == (not fail_before_session)
