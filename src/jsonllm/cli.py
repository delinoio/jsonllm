"""Explicit local commands. Installation and help never load weights or call APIs."""

import argparse
import json
import sys
import time
from pathlib import Path

from jsonschema.exceptions import ValidationError

from .config import load_config
from .io import dumps
from .release import MODEL_ID, MODEL_REVISION


class LazyCUDA:
    def __init__(self, model, revision, share):
        self.model, self.revision, self.share = model, revision, share
        self.loaded = None
        self.load_seconds = 0

    def open_record(self, context, count):
        if self.loaded is None:
            from .backends.shared_cuda import SharedPredictor

            started = time.perf_counter()
            self.loaded = SharedPredictor.load(self.model, self.revision, share=self.share)
            self.load_seconds = time.perf_counter() - started
        return self.loaded.open_record(context, count)


def parser():
    root = argparse.ArgumentParser(description="JSONLLM typed decisions and structured execution")
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Execute a registered UI specification")
    run.add_argument("--spec", required=True)
    run.add_argument("--input", required=True)
    run.add_argument("--model", default=MODEL_ID)
    run.add_argument("--revision", default=MODEL_REVISION)
    run.add_argument("--unshared", action="store_true")
    train = commands.add_parser("train", help="Train a LoRA from prepared data")
    train.add_argument("--config", required=True)
    train.add_argument("--data")
    train.add_argument("--output")
    train.add_argument("--max-steps", type=int)
    train.add_argument("--resume", help="CUDA checkpoint; use a new output directory")
    merge = commands.add_parser("merge-adapter", help="Merge and verify a completed CUDA run")
    merge.add_argument("--adapter", required=True)
    merge.add_argument("--output", required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            from .ui import compile_ui, run_ui

            spec = compile_ui(json.loads(Path(args.spec).read_text()))
            inputs = json.loads(Path(args.input).read_text())
            predictor = LazyCUDA(args.model, args.revision, not args.unshared)
            result = run_ui(spec, **inputs, predictor=predictor, max_tokens=128)
            result["diagnostics"]["model_load_seconds"] = predictor.load_seconds
            print(dumps(result))
            return 0 if result["output"] is not None else 1
        if args.command == "train":
            from .training import train_model

            config = load_config(
                args.config,
                {key: getattr(args, key) for key in ("data", "output", "max_steps")},
            )
            result = train_model(config, resume=args.resume)
        else:
            from .merge import merge_adapter

            result = merge_adapter(args.adapter, args.output)
        print(dumps(result))
        return 0
    except (ValueError, OSError, ValidationError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
