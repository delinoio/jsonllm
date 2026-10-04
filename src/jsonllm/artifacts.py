"""File integrity helpers; no infrastructure management."""

import hashlib
from pathlib import Path


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact_manifest(root):
    root = Path(root)
    return {
        str(path.relative_to(root)): file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in {"artifacts.json", "worker.json", "watchdog.json"}
        and path.suffix != ".log"
    }


def verify_artifacts(root, manifest):
    root = Path(root).resolve()
    if not manifest:
        raise ValueError("Empty artifact manifest")
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
