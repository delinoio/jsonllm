import re
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


def read_env_file(path):
    """Read single-line shell-style assignments without executing or expanding them."""
    result = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
        if not match:
            raise ValueError(f"Invalid env assignment on line {number}")
        try:
            values = shlex.split(match[2], comments=True)
        except ValueError:
            raise ValueError(f"Invalid env quoting on line {number}") from None
        if len(values) > 1:
            raise ValueError(f"Quote env values containing spaces on line {number}")
        result[match[1]] = values[0] if values else ""
    return result


@dataclass
class TrainConfig:
    model: str = "Qwen/Qwen3.5-4B"
    revision: str = "main"
    backend: str = "mlx"
    data: str = "data/prepared"
    output: str = "runs/qwen35-4b-mlx"
    rank: int = 16
    alpha: int = 32
    learning_rate: float = 1e-4
    max_length: int = 2048
    batch_size: int = 1
    gradient_accumulation: int = 8
    epochs: int = 1
    seed: int = 42
    save_steps: int = 100
    max_steps: int | None = None
    gradient_checkpointing: bool = True
    prompt_version: str = "field-v1"

    def __post_init__(self):
        from .prompts import PROMPT_VERSIONS

        if self.prompt_version not in PROMPT_VERSIONS:
            raise ValueError("Unknown prompt_version")
        for key in ("model", "revision", "data", "output"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"{key} must be a non-empty string")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be a boolean")
        if self.backend not in {"mlx", "cuda"}:
            raise ValueError("backend must be mlx or cuda")
        for key in (
            "rank",
            "alpha",
            "max_length",
            "batch_size",
            "gradient_accumulation",
            "epochs",
            "save_steps",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if self.max_steps is not None and (type(self.max_steps) is not int or self.max_steps <= 0):
            raise ValueError("max_steps must be a positive integer")


def load_config(path=None, overrides=None):
    values = yaml.safe_load(Path(path).read_text()) if path else {}
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ValueError("Training config must be a mapping")
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    try:
        return TrainConfig(**values)
    except TypeError as exc:
        raise ValueError(f"Invalid training config: {exc}") from exc


def save_config(path, config):
    Path(path).write_text(yaml.safe_dump(asdict(config), sort_keys=False))
