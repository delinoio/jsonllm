# Usage

Run commands from the repository root. The basic environment needs no model or API credentials.

## A code-only event

```python
import json
from pathlib import Path

from jsonllm import compile_ui, run_ui

spec = compile_ui(json.loads(Path("examples/genui-order.spec.json").read_text()))
inputs = json.loads(Path("examples/genui-order.reset.json").read_text())
result = run_ui(spec, **inputs)
assert result["output"]["component"] == "EmptyState"
assert result["diagnostics"]["model_fields"] == []
```

The order example includes typed decisions, dependent decisions, before/after rules, exact fact copies, and arithmetic. A reset event bypasses all model decisions. Inputs are copied before state transitions, so the caller's state is not mutated.

## Decisions with the frozen model

After `uv sync --locked --extra cuda`, use the following in a CUDA environment:

```python
import json
from pathlib import Path

from jsonllm import compile_ui, run_ui
from jsonllm.backends.shared_cuda import SharedPredictor
from jsonllm.release import MODEL_ID, MODEL_REVISION

predictor = SharedPredictor.load(
    MODEL_ID,
    MODEL_REVISION,
    share=True,
    bucket_choices=True,
    compile_model=False,
    max_length=2048,
)
spec = compile_ui(json.loads(Path("examples/genui-order.spec.json").read_text()))
inputs = json.loads(Path("examples/genui-order.input.json").read_text())
result = run_ui(spec, **inputs, predictor=predictor, max_tokens=128)
if result["output"] is None:
    raise RuntimeError(result["diagnostics"]["errors"])
print(result["output"])
```

This uses `shared-context-v3`, the pinned fine-tuned revision, FP16 inference, eager execution, and choice bucketing. Do not use the upstream Qwen commit as the fine-tuned repository's revision. The two repositories have separate commit histories.

`SharedPredictor` is an experimental backend, not an HTTP service. Keep a loaded predictor for successive calls instead of loading it for each event. Loading, warmup, transport, and rendering costs must be accounted for separately in an application.

## Specification and failure contract

- Supply `version: "genui-v1"`, a nonempty `components` allowlist, and `view` with `component` and `props`.
- Inputs are a text `context` and object-valued `state`, `event`, and `facts`.
- Decisions use the existing typed schema and optional `depends_on` edges. Unknown dependencies and cycles are rejected.
- Every conditional decision needs both `when` and `otherwise`.
- Transitions can update declared top-level state keys. Before rules cannot use model decisions.
- `compile_ui` raises `ValueError` for invalid specifications. `run_ui` returns diagnostic errors for invalid execution inputs, unavailable required decisions, or rejected values. Backend resource failures can still raise exceptions.

The CLI returns a nonzero exit status for failed execution. Tests in `test_ui.py`, `test_ui_example.py`, and `test_value_contract.py` provide executable contract examples. The specification is application code: review its allowed operations and state changes before using it with real systems.
