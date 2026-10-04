"""Backend-neutral token preparation and run provenance."""

import importlib.metadata
import json
import re
from dataclasses import asdict, replace
from pathlib import Path

from .config import save_config
from .io import digest, read_jsonl, write_json
from .prompts import PROMPT_VERSION, encode_example


def resolve_model(model, revision):
    if Path(model).is_dir():
        # Local models should themselves be kept immutable; record configuration fingerprint.
        path = Path(model).resolve()
        return str(path), {
            "path": str(path),
            "config_hash": digest(json.loads((path / "config.json").read_text())),
        }
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision, {"repo": model, "revision": revision}

    from huggingface_hub import model_info

    sha = model_info(model, revision=revision).sha
    return sha, {"repo": model, "revision": sha}


def load_tokenizer(model, revision):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encoded_dataset(rows, tokenizer, max_length, prompt_version=PROMPT_VERSION):
    accepted, excluded = [], []
    for row in rows:
        if row.get("prompt_version") != prompt_version:
            raise ValueError("Prepared prompt version differs; regenerate the prepared data")
        encoded = encode_example(row, tokenizer)
        if len(encoded["input_ids"]) > max_length:
            excluded.append({"id": row.get("id"), "tokens": len(encoded["input_ids"])})
        else:
            accepted.append(encoded)
    return accepted, excluded


def train_model(config, *, resume=None, warm_start=None):
    if resume and config.backend != "cuda":
        raise ValueError("--resume is CUDA-only; use --warm-start for MLX adapter continuation")
    if warm_start and config.backend != "mlx":
        raise ValueError("--warm-start is MLX-only")
    output = Path(config.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output must be empty; resume into a new run directory")
    source = Path(config.data)
    rows = {split: read_jsonl(source / f"{split}.jsonl") for split in ("train", "validation")}
    revision, model_metadata = resolve_model(config.model, config.revision)
    if "repo" in model_metadata:
        config = replace(config, revision=revision)
    tokenizer = load_tokenizer(config.model, revision)
    datasets, exclusions = {}, {}
    for split, examples in rows.items():
        datasets[split], exclusions[split] = encoded_dataset(
            examples, tokenizer, config.max_length, config.prompt_version
        )
        if not datasets[split]:
            raise ValueError(f"No {split} examples remain after length filtering")
    metadata = {
        "config": asdict(config),
        "model": model_metadata,
        "prompt_version": config.prompt_version,
        "dataset_hashes": {split: digest(examples) for split, examples in rows.items()},
        "retained": {split: len(examples) for split, examples in datasets.items()},
        "excluded_overlength": exclusions,
        "resume": resume,
        "warm_start": warm_start,
        "versions": {},
    }
    for package in (
        "jsonllm",
        "transformers",
        "mlx-lm",
        "mlx",
        "torch",
        "peft",
        "accelerate",
        "flash-linear-attention",
        "fla-core",
        "triton",
    ):
        try:
            metadata["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    if resume or warm_start:
        previous_dir = Path(resume or warm_start)
        if resume:
            previous_dir = previous_dir.parent
        previous = json.loads((previous_dir / "run.json").read_text())
        if previous["model"] != model_metadata or previous["config"]["backend"] != config.backend:
            raise ValueError("Adapter/checkpoint base model differs from this run")
        previous["config"].setdefault("prompt_version", PROMPT_VERSION)
        if resume and (
            previous["dataset_hashes"] != metadata["dataset_hashes"]
            or previous["config"] | {"output": config.output} != asdict(config)
        ):
            raise ValueError("CUDA resume requires identical data and config except output")
        if warm_start and any(
            previous["config"][key] != getattr(config, key) for key in ("rank", "alpha")
        ):
            raise ValueError("Warm start requires the original LoRA rank and alpha")
    output.mkdir(parents=True, exist_ok=True)
    save_config(output / "config.yaml", config)
    write_json(output / "run.json", {**metadata, "status": "running"})
    try:
        if config.backend == "mlx":
            from .backends.mlx import train

            metrics = train(config, revision, tokenizer, datasets, warm_start=warm_start)
        else:
            from .backends.cuda import train

            metrics = train(config, revision, tokenizer, datasets, resume=resume)
        tokenizer.save_pretrained(output / "tokenizer")
        write_json(output / "run.json", {**metadata, "status": "complete", "metrics": metrics})
        return metrics
    except Exception:
        write_json(output / "run.json", {**metadata, "status": "failed"})
        raise
