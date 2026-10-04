import json
from pathlib import Path

from jsonllm.ui import compile_ui, run_ui


def test_documented_ui_rule_example_handles_exact_props_and_stale_events():
    root = Path(__file__).parents[1]
    spec = compile_ui(json.loads((root / "examples/genui-order.spec.json").read_text()))
    inputs = json.loads((root / "examples/genui-order.input.json").read_text())

    class Predictor:
        metrics = {}

        def open_record(self, context, count):
            assert count == 2
            return self

        def predict(self, record, names, answers, max_tokens):
            return {"component": "EditAddress", "urgent": False}

        def close(self):
            pass

    first = run_ui(spec, **inputs, predictor=Predictor())
    assert first["output"] == {
        "component": "EditAddress",
        "props": {
            "title": "Order demo-42",
            "shipping_address": "12 Example Street",
            "total": 25.0,
            "urgent": False,
            "requires_confirmation": False,
        },
        "state": {"selection": "EditAddress", "phase": "selected", "total": 25.0},
    }
    inputs.update(state=first["output"]["state"], event={"type": "response", "request_id": "old"})
    stale = run_ui(spec, **inputs, predictor=None)
    assert stale["output"] == first["output"]
    inputs["event"] = {"type": "reset"}
    reset = run_ui(spec, **inputs, predictor=None)
    assert reset["output"]["component"] == "EmptyState"
    assert reset["output"]["state"] == {"selection": None, "phase": "idle", "total": 0}
    assert reset["diagnostics"]["model_fields"] == []
