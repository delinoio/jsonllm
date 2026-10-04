# Design benchmark protocol

This study tests execution mechanisms. It does not train a model or select a new
checkpoint. Both the code repository and model repository remain private until
the owner reviews the results.

## Comparisons

The eager CUDA comparison uses the same text weights, FP16 conversion, tokenizer,
hardware, greedy decoding, 2048-token limit, and exact-answer oracle:

1. `whole_json`: one schema-constrained JSON object.
2. `serial_fields`: one typed field at a time, without shared prefix state.
3. `batch_fields`: independent fields in a dependency wave, without shared state.
4. `shared_fields`: the same field prompts and waves, with shared prefix state.

`vllm_json` is a separate practical baseline with JSON Schema constraints and the
engine's default optimizations. Its loopback HTTP overhead is included. It is
not used to attribute a speed difference to prefix sharing alone. Each request
has a fresh cache salt; repeats cannot reuse a previous request's cached prefix.

The original Qwen revision and JSONLLM release revision are fixed in
`jsonllm.release`. They are reported separately. JSONLLM was trained on scalar
answers, not whole-object answers. Thus whole-object versus typed results can
reflect training-format compatibility as well as output serialization and
execution. The field-to-field ablations keep the prompt contract identical.

The primary output is the same flat decision object for every method. A selected
field uses one label token in the typed methods; whole JSON emits field names,
JSON syntax, and values. This is part of the design comparison, not an equal
output-token experiment. Output token counts make that difference visible.

## Tasks and exclusions

`benchmark_data.py` generates fresh English/Korean tasks with seed 20261005.
There are 64 development records and 256 core test records, balanced by language
and among selection, nullable numeric extraction, string copying, and dependent
lookup tasks. The oracle reads facts through a separate implementation. Only
`id`, `context`, and `questions` cross the inference boundary. No reference answer,
oracle code, or source metadata is passed to an inference backend.

These are controlled synthetic tasks. They test retrieval, simple comparisons,
dependency handling, and exact copying. They do not measure open-ended reasoning,
creative writing, or real-world UI usefulness. No external factual judge is used.

Sweeps change field count, context length, dependency depth, choice count, or
copied-string length. The requested context size is a lower bound: source facts
are never truncated to fit a nominal size. Reports include actual token counts.
String-length levels are generated from approximately single-token English
words; reports use measured answer-token lengths, not word counts as tokens.

Before GPU inference, the preparer checks every method's prompt and reserved
output length. An overlength record is excluded from all compared methods, with
its reason retained. Whole-JSON output allowance is 1.25 times the reference
JSON token count plus 32, with a minimum of 128. This allowance is fixed before
inference; the reference text is never part of the prompt. Field generation has
a 128-token allowance. This allocation policy is part of the workload definition.

The exploratory topology tasks contain eight candidate slots. The model chooses
which nodes exist, node kinds, and parent links. The trusted assembler does not
repair outputs. It checks references and ordering. This is **bounded topology
selection from supplied facts**, not unrestricted tree design or workflow
synthesis from a vague natural-language request. Workflows are never executed.
Topology selection latency includes assembly/validation after the same flat
decision object. It is not a comparison against directly serializing the final
nested tree; that additional application experiment remains unperformed.

## Timing and quality

Core test records run five times per model and method. Sweep records run three
times. Repeated timings are not additional independent quality samples. The
method and model order rotates across core repeats. Warmup uses development
records only. Loading, server startup, and warmup costs are reported separately.

Latency includes input formatting, inference, parsing, and validation. It includes
failures, not just successful outputs. `first_usable_seconds` is the internal time
when a validated field wave becomes available; whole-object methods expose only
the completed object. This is not browser rendering latency or network streaming
time. The existing public UI API still returns its final output atomically.

The report gives p50/p95/p99, five-trial medians and ranges, complete outputs per
second, **correct complete outputs per second**, exact accuracy, schema validity,
failures, timeouts, output tokens, and memory. Token totals can be incomplete for
failed requests and must be labeled accordingly. Device-wide memory is sampled
every 200 ms; CUDA allocator peaks are also recorded for eager methods. vLLM's
cache reservation makes its device-memory number different from active tensor
memory. p99 from fewer than 1000 independent records is exploratory.

Concurrent load retains JSONLLM's record-level lock. Closed-loop concurrency and
fixed arrival-rate experiments are reported separately. Open-loop requests time
out after 120 seconds, including queueing. The finite client pool has 128 workers;
that is not a model concurrency-128 performance claim. Expired queued requests
remain failures. Report offered work, achieved throughput, and queueing rather
than interpreting the configured concurrency as useful GPU parallelism.

Quality criteria remain schema/code correctness 100% and no more than a two
percentage-point loss in exact accuracy. Point estimates and uncertainty are
separate. Paired comparisons resample record IDs, not repeated executions.
Deterministic assembler tests are not mixed into model accuracy. A failed quality
criterion prevents a claim of a quality-preserving speedup.

## Reproduction and budget

Use the locked CUDA environment and a separate environment installed from
`requirements-vllm.lock`. The runner and worker provide `--help`. For example:

```sh
uv run --extra cuda python scripts/run_design_benchmark.py \
  --model kdy1/JSONLLM-Qwen3.5-4B-v0.1 \
  --revision 2a020c2cddaec6fcb86ef0d6665f1ee11f699738 \
  --data DATA/core.jsonl --warmup-data DATA/dev.jsonl \
  --method shared_fields --output runs/example-trial
```

Output directories must be new. Preserve failed or invalidated runs. A code or
environment defect requires a documented invalidation and a fresh complete
affected comparison, not replacement of only unfavorable records.

The extra campaign budget is $100, including setup, unsuccessful attempts,
storage, and recovery. Stop experiments before $90 and reserve $10 for recovery
and cleanup. Admit complete comparison blocks using development timing only.
Work proceeds in the fixed order core, scaling, load, then topology. Unperformed
blocks and exclusions remain in the report. One H100 80GB Secure GPU is used;
there is no external teacher API, retraining, endpoint, or public release.
