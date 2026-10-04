---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
base_model_relation: finetune
language:
  - en
  - ko
library_name: transformers
tags:
  - jsonllm
  - structured-generation
  - lora
  - research
inference: false
---

# JSONLLM-Qwen3.5-4B-v0.1

A research POC for typed semantic decisions and dependency-aware structured execution. It is intended to help researchers and model builders investigate how a model can resolve unknown values while a runtime handles known structure and deterministic work.

The current task is restricted to registered UI components and state transitions. The model does not generate arbitrary component trees or synthesize general workflows. The repositories are currently private; this card prepares a future open release.

## Model and files

- Base: `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Root: merged text-model weights, tokenizer, chat template, model config, and generation config.
- `adapter/`: the original LoRA weights and configuration.
- `data/`: frozen 8,000 train / 1,000 validation / 1,000 test records, prepared training/validation files, split description, and SHA-256 sums.
- `docs/`: evaluation, training metadata, dataset description, and provenance.
- `manifest.json`: sizes and SHA-256 digests for all release files except the manifest itself.

The stored merged weights are **BF16**. The evaluated shared CUDA runtime casts weights **and buffers to FP16**. The model is a modified derivative of the upstream base. Do not apply the adapter again to the merged root weights. Optimizer states, intermediate checkpoints, API logs, credentials, and cloud operations are excluded.

## Training

One candidate was trained from the original base with LoRA rank 16, alpha 32, learning rate 5e-5, batch size 1, gradient accumulation 8, one epoch, training seed 42, maximum length 2048, and gradient checkpointing. The data seed was 20261004. Training used 15,619 prepared field examples and 1,946 validation examples, with zero overlength exclusions. It completed 1,953 optimizer steps in 6,142.433 seconds. Train loss was 0.0488001158 and evaluation loss was 0.0132439472; these are not accuracy values.

The synthetic teacher was `deepseek/deepseek-v4.1-flash`, accessed through OpenRouter with the official `deepseek` provider fixed, no fallback, reasoning disabled, and strict JSON. English and Korean records are equally represented in each split. Earlier provider data and weights were not reused. See the dataset and provenance documents for rights and limitations.

## Use with shared-context-v3

Use the [JSONLLM runtime](https://github.com/delinoio/jsonllm), Python 3.12, and the locked CUDA extra. The runtime is experimental and was evaluated on an H100 80 GB. Generic chat generation does not reproduce this model's typed execution contract.

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
print(result)
```

Run the example from the JSONLLM checkout. `MODEL_REVISION` pins the uploaded model commit, not the upstream Qwen revision. Keep the original `shared-context-v3` prompt/tokenization contract. The application supplies structure, authoritative facts, state, and event data. The runtime batches independent fields, reuses common context, runs dependencies in order, and constructs JSON.

## Evaluation and limitations

After one validation run per model, the candidate was frozen. Each model ran five trials at concurrency 1 and five at concurrency 8, followed by one rollout. Each trial uses the same 1,000 test records: 800 model requests and 200 deterministic code records. These are timing repetitions, not 5,000 independent quality samples.

| Result | JSONLLM | Original Qwen under same runtime |
|---|---:|---:|
| Test model-record accuracy, c1 / c8 | 100% / 100% | 75.100% / 75.075% |
| Rollout model-record accuracy | 100% | 74.875% |
| Rollout code accuracy | 100% | 93.5% |
| Test p95 median, c1 | 1,421.418 ms | 2,183.151 ms |
| Test p95 median, c8 | 6,041.267 ms | 7,954.604 ms |
| Completed records/s median, c1 | 2.624 | 2.267 |
| Completed records/s median, c8 | 1.983 | 1.735 |

The candidate's schema, code, factual-judgment, and whole-trajectory scores were 100% in test and rollout. Factual scoring uses accepted synthetic references or teacher judgments against stored facts, not independent real-world truth. Schema validity depends on runtime constraints. Deterministic code records are not model-generated programs. Throughput includes those code records.

Concurrency 8 had worse throughput than concurrency 1. A session lock serializes records, even though independent fields within a record can run together. The single validation run improved p95 but increased p50 (240.614 versus 230.363 ms). Timings exclude model loading, warmup, factual-review API calls, transport, and rendering. Peak allocated GPU memory was about 8.216 GiB at c1 and 8.435 GiB at c8; this is not a minimum hardware recommendation.

The experiment does not establish general tree/workflow synthesis, broad factual reliability, or a speed advantage over whole-JSON generation or optimized serving engines. Full per-trial ranges, type breakdowns, tokens, memory, and scoring definitions are in the evaluation report. New causal speed benchmarks are deferred. The repository also includes the dataset description, training metadata, and provenance with frozen hashes.

## License

Apache-2.0 for the derivative model, code, and included synthetic data, with upstream Qwen attribution retained in LICENSE and NOTICE. The data-use basis is [DeepSeek API Terms §4.2](https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html), subject to its terms and applicable law. Provider names document origin and do not imply endorsement.
