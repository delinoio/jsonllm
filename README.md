# JSONLLM

**Typed decisions, shared context, and dependency-aware execution for structured outputs.**

JSONLLM is a personal research proof of concept. It asks whether a model should spend its time making semantic decisions while a runtime handles structure, exact values, and deterministic work. The aim is to make this approach concrete enough for researchers and model builders to evaluate and improve.

Generating an entire UI description as a serial text stream can put unnecessary work on the critical path. Here, an application supplies a specification. The runtime asks the model only for unknown values, batches independent decisions, reuses their common context, and constructs the result in code. The current demonstration uses registered UI components and state transitions.

This is a small, budget-conscious experiment based on Qwen3.5-4B. It is an invitation to investigate an execution interface for models, rather than a claim that general GenUI is solved.

## What is implemented

- Typed choice, boolean, and constrained scalar decisions.
- Dependency waves: independent fields run together; dependent fields wait for their inputs.
- `shared-context-v3`: common context is processed once and its hybrid model cache is copied for field branches.
- Deterministic state transitions, arithmetic, exact copies, component selection, and output assembly.
- An experimental CUDA runtime, LoRA training, merge checks, and evaluation tools.
- `compile_ui`, `run_ui`, and `SharedPredictor`, with the original POC input contracts preserved.

The model does **not** invent arbitrary component trees or workflows in this version. The application defines the structure and rules. General tree generation, dynamic graph construction, and larger-model scaling are research questions. See [design](docs/design.md) and [planned benchmarks](docs/benchmark-plan.md).

## Execution design results

The completed study ran all 540 planned trials on new synthetic tasks, with frozen weights and no retraining. Batching fields reduced median latency versus serial fields by about 3.14× under both models, with no observed quality loss. Prefix sharing was slower than ordinary batching on the short-context core set, but faster at 513–1,026 common-context tokens in the small exploratory sweep.

Quality limits the broader claim. JSONLLM core exact accuracy was 78.52% for whole JSON and 65.23% for shared fields. Shared fields had lower median latency but worse p95, and failed the quality-preserving speed criterion. vLLM had the highest correct throughput, while typed tree/workflow selection scored 0% exact accuracy. All failures and exclusions are retained. See the [full design report and graphs](docs/design-benchmark.md), [raw measurements](docs/benchmarks/design-20261004/README.md), and [protocol](docs/design-benchmark-protocol.md).

## Historical training evidence

A single LoRA candidate was trained on 8,000 synthetic records, validated on 1,000, then frozen before testing on 1,000 held-out records. Both models used the same custom runtime on one H100 80 GB GPU.

| Test | JSONLLM | Original Qwen3.5-4B |
|---|---:|---:|
| Model-record accuracy, concurrency 1 | 100% | 75.100% |
| Median trial p95, concurrency 1 | 1,421 ms | 2,183 ms |
| Median completed records/s, concurrency 1 | 2.624 | 2.267 |
| Median trial p95, concurrency 8 | 6,041 ms | 7,955 ms |
| Median completed records/s, concurrency 8 | 1.983 | 1.735 |

Each condition repeats the **same** test set five times for latency measurement. Each trial has 800 model records and 200 deterministic code records; throughput includes both. Factual scores use synthetic reference facts and teacher judgments, not independently established real-world truth. Schema validity also depends on runtime constraints.

Concurrency 8 made throughput worse than concurrency 1. The runtime serializes records with a lock, despite batching fields within a record. These results compare fine-tuned and original weights; they do **not** establish a speed advantage over whole-JSON generation or an optimized serving engine. In the single validation trial, p95 improved but p50 increased. [Full evaluation and ranges](docs/evaluation.md).

## Quick start

Python 3.12 and [uv](https://docs.astral.sh/uv/) are used for the locked environment. The code and model repositories are public.

```sh
git clone https://github.com/delinoio/jsonllm.git
cd jsonllm
uv sync --locked
uv run --no-sync jsonllm --help
uv run --no-sync jsonllm run \
  --spec examples/genui-order.spec.json \
  --input examples/genui-order.reset.json
```

The reset example executes entirely in code and does not load weights. It returns an `EmptyState` component, updated state, and diagnostics.

For model decisions, use a compatible NVIDIA CUDA environment:

```sh
uv sync --locked --extra cuda
uv run --no-sync jsonllm run \
  --spec examples/genui-order.spec.json \
  --input examples/genui-order.input.json
```

The CLI uses the model ID and immutable release revision in [`release.py`](src/jsonllm/release.py). The model can be downloaded without Hugging Face authentication. Stored weights are **BF16**; the tested shared CUDA runtime converts weights and buffers to **FP16**. This custom runtime is the supported way to reproduce the POC behavior. A generic chat pipeline does not reproduce its execution contract.

See [usage](docs/usage.md), [reproduction](docs/reproduction.md), [model card](docs/model-card.md), and [dataset description](docs/dataset.md). Weights, the original adapter, and frozen data are stored together in [the model repository](https://huggingface.co/kdy1/JSONLLM-Qwen3.5-4B-v0.1); they are not checked into GitHub.

## Development

```sh
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest -m 'not mlx and not cuda and not tokenizer'
uv build
```

Optional-backend tests skip when their dependencies or hardware are unavailable. No paid service is needed for the CPU tests. [Release verification](docs/release-verification.json) records the original packaging checks; [design validation](docs/benchmarks/design-20261004/evidence/validation.json) records the completed benchmark checks.

## License and provenance

Code, this model derivative, and the included synthetic dataset are prepared under Apache-2.0. The original Qwen license and attribution are retained. See [LICENSE](LICENSE), [NOTICE](NOTICE), and [provenance](docs/provenance.md) for source revisions, modification notices, data rights, and hashes. The project is independent of the upstream model and API providers.
