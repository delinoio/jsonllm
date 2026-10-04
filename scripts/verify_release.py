"""Verify a downloaded release without loading weights or contacting a service."""

import argparse
import json
from pathlib import Path

from jsonllm.artifacts import verify_artifacts


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    entries = manifest["files"]
    verify_artifacts(root, {name: value["sha256"] for name, value in entries.items()})
    for name, value in entries.items():
        if (root / name).stat().st_size != value["bytes"]:
            raise ValueError(f"Artifact size mismatch: {name}")
    return len(entries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(f"Verified {verify(args.directory)} release files.")
