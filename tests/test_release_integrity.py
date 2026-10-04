import importlib.util
import json

import pytest

from jsonllm.artifacts import file_hash

spec = importlib.util.spec_from_file_location("verify_release", "scripts/verify_release.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def manifest(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    value = {"files": {name: {"sha256": file_hash(path), "bytes": path.stat().st_size}}}
    (root / "manifest.json").write_text(json.dumps(value))


def test_release_checks_bytes_and_missing_files(tmp_path):
    manifest(tmp_path, "data/train.jsonl", "frozen\n")
    assert module.verify(tmp_path) == 1
    (tmp_path / "data/train.jsonl").write_text("edited\n")
    with pytest.raises(ValueError, match="checksum"):
        module.verify(tmp_path)
    (tmp_path / "data/train.jsonl").unlink()
    with pytest.raises(ValueError, match="checksum"):
        module.verify(tmp_path)


def test_release_rejects_parent_paths(tmp_path):
    nested = tmp_path / "release"
    nested.mkdir()
    manifest(nested, "../outside", "data")
    with pytest.raises(ValueError, match="checksum"):
        module.verify(nested)


def test_release_rejects_incorrect_size(tmp_path):
    manifest(tmp_path, "model", "weights")
    path = tmp_path / "manifest.json"
    value = json.loads(path.read_text())
    value["files"]["model"]["bytes"] += 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="size"):
        module.verify(tmp_path)
