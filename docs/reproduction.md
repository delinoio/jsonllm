# Reproduction

The release preserves one completed experiment. No training, GPU evaluation, or paid API call is performed when the package is installed or its CPU tests run.

## Get and verify the frozen artifacts

Install the Hugging Face CLI and authenticate with `hf auth login`. From the repository root:

```sh
uv sync --locked
MODEL_REVISION=$(uv run --no-sync python -c 'from jsonllm.release import MODEL_REVISION; print(MODEL_REVISION)')
hf download kdy1/JSONLLM-Qwen3.5-4B-v0.1 \
  --revision "$MODEL_REVISION" \
  --local-dir models/JSONLLM-Qwen3.5-4B-v0.1
uv run --no-sync python scripts/verify_release.py models/JSONLLM-Qwen3.5-4B-v0.1
```

The manifest contains byte-level SHA-256 digests and sizes. The dataset also has canonical JSON digests from collection. These are different hash definitions and must not be substituted for one another.

## Historical training recipe

The following commands are for a future, explicitly chosen reproduction run on CUDA hardware. They were **not rerun for repository preparation**. Use a new output directory and preserve the distributed artifacts.

```sh
uv sync --locked --extra cuda
uv run --no-sync jsonllm train --config configs/qwen35-4b-v01.yaml
uv run --no-sync jsonllm merge-adapter \
  --adapter runs/jsonllm-qwen35-4b-v01 \
  --output runs/jsonllm-qwen35-4b-v01-merged
```

The recipe uses the original Qwen revision, not the fine-tuned model as a warm start. It uses LR 5e-5, rank 16, alpha 32, batch size 1, accumulation 8, one epoch, training seed 42, length limit 2048, and gradient checkpointing. The data-generation seed is 20261004. Training retained 15,619 field examples and 1,946 validation examples; no examples were excluded for length. The run completed 1,953 optimizer steps. Hardware and kernels can affect numerical reproducibility.

The source training/merge tools preserve the existing loss and reload checks. The released adapter can also be applied to the exact upstream base with PEFT; the merged root weights avoid that extra merge step for inference. The distributed weights are already merged and must not receive the adapter a second time.

## Historical evaluation recipe

For a future single inference trial:

```sh
uv run --no-sync python scripts/benchmark_shared_ui.py \
  --model models/JSONLLM-Qwen3.5-4B-v0.1 \
  --data models/JSONLLM-Qwen3.5-4B-v0.1/data/test.jsonl \
  --output runs/reproduction-c1 \
  --engine shared --prompt-version shared-context-v3 \
  --compile-mode eager --bucket-choices --max-tokens 128 --concurrency 1
```

Use fresh output directories. The historical protocol ran each model five times at concurrency 1 and five at concurrency 8, plus one rollout per model with `--rollout`. Validation preceded candidate sealing; test results were not used to change the candidate. The original model was `Qwen/Qwen3.5-4B` at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` under the same runtime.

Inference summaries can contain pending factual judgments. They are not the final reviewed accuracy. `jsonllm.shared_review` accepts an explicitly supplied teacher client; no API credential or campaign-specific cloud coordinator is included. The released evaluation summary contains the completed historical judgments. Reproducing those judgments requires a separately authorized review procedure; it is not part of this private upload.

## Local checks

```sh
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest -m 'not mlx and not cuda and not tokenizer'
uv build
```

The core environment skips tests that need optional backend packages. Installing the CUDA extra makes more CPU tests available, but does not provide a CUDA device on a Mac. To use an already downloaded release tokenizer, set `JSONLLM_TEST_TOKENIZER=models/JSONLLM-Qwen3.5-4B-v0.1`. No model weights are loaded for scalar grammar or prompt-boundary checks. GPU checks and new performance comparisons are separate from these local checks.

`requirements-vllm.lock` is an optional separate serving environment for future comparisons. The retained vLLM client expects a local server with `--served-model-name jsonllm`; vLLM was not used for the reported shared-runtime results.
