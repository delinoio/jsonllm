import json
from pathlib import Path

from jsonllm.cli import main, parser
from jsonllm.release import MODEL_ID, MODEL_REVISION


def test_release_revision_is_separate_from_base_revision():
    args = parser().parse_args(["run", "--spec", "s", "--input", "i"])
    assert (args.model, args.revision) == (MODEL_ID, MODEL_REVISION)
    assert args.revision != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def test_code_only_run_never_loads_cuda(tmp_path, monkeypatch, capsys):
    def forbidden(*a, **kw):
        raise AssertionError("Code-only execution must not load weights")

    monkeypatch.setattr("jsonllm.cli.LazyCUDA.open_record", forbidden)
    inputs = json.loads(Path("examples/genui-order.input.json").read_text())
    inputs["event"] = {"type": "reset"}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(inputs))
    assert main(["run", "--spec", "examples/genui-order.spec.json", "--input", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["output"]["component"] == "EmptyState"
    assert result["diagnostics"]["model_fields"] == []
    assert result["diagnostics"]["model_load_seconds"] == 0


def test_invalid_spec_fails_before_loading(tmp_path, capsys):
    spec = tmp_path / "bad.json"
    spec.write_text('{"version":"unsupported"}')
    assert main(["run", "--spec", str(spec), "--input", "unused"]) == 1
    assert "genui-v1" in capsys.readouterr().err
