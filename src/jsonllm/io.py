"""Small, strict, durable JSON helpers."""

import hashlib
import json
import os
from pathlib import Path


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def read_jsonl(path):
    rows = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line, parse_constant=_reject_constant))
                except ValueError as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc
    return rows


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(dumps(value) + "\n")
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(dumps(row) + "\n" for row in rows))
    temporary.replace(path)


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(dumps(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
