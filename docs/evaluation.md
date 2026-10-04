# Evaluation: frozen v0.1 experiment

These are results from the completed 2026-10-04 experiment, not new measurements made during repository preparation. A single candidate was trained from the fixed original Qwen3.5-4B revision. Validation occurred once per model, followed by candidate sealing, twenty test trials, and one rollout per model. All test inference trials completed without execution errors.

## Protocol and interpretation

The dataset contains 10,000 records: 8,000 training, 1,000 validation, and 1,000 test. Test and validation workflow families are held out, within the same restricted task generator and decision interface. Each test trial has 800 model records and 200 deterministic code records across 100 groups. Five trials repeat the same examples; they are latency repetitions, not five independent quality samples.

Quality combines deterministic checks of component choice, copies, state, code, and schema with factual judgments against synthetic stored facts. Accepted reference text can pass by exact match; novel sentences receive blind teacher review. A separate review call is not an independent source of factual truth. Schema success reflects the constraints and assembly runtime as well as the model. The code category is execution of registered deterministic rules, not generated program synthesis.

Both models used one NVIDIA H100 80 GB, shared-context-v3, FP16 inference, eager execution, choice bucketing, and a maximum of 128 output tokens. Stored weights are BF16. This is a fine-tuned-versus-original comparison in the same runtime, not a comparison against conventional whole-JSON generation, vLLM, SGLang, or a production API.

## Quality

| Stage | Model | Model-record accuracy | Code | Schema | Factual judgment | Whole trajectory |
|---|---|---:|---:|---:|---:|---:|
| validation | JSONLLM | 99.875% | 100.000% | 100.000% | 100.000% | 99.000% |
| validation | Original Qwen | 77.250% | 100.000% | 100.000% | 44.500% | 19.000% |
| test-c1 | JSONLLM | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| test-c1 | Original Qwen | 75.100% | 100.000% | 100.000% | 35.400% | 15.000% |
| test-c8 | JSONLLM | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| test-c8 | Original Qwen | 75.075% | 100.000% | 100.000% | 35.300% | 15.000% |
| rollout | JSONLLM | 100.000% | 100.000% | 100.000% | 100.000% | 100.000% |
| rollout | Original Qwen | 74.875% | 93.500% | 100.000% | 35.000% | 15.000% |

All four stages met the fixed candidate gate: code and schema 100%, and model accuracy no more than 2 percentage points below the original model. This gate is local to this benchmark and is not a production-readiness or public-release approval. The original model’s rollout code score was 93.5%; rollout feeds predicted state forward, so earlier model errors can affect later deterministic actions.

## Latency and throughput

Values are the median of five per-trial statistics, followed by [minimum, maximum]. Latency is for model records and includes inference, output assembly, validation, and lock wait. It excludes model loading, warmup, factual-review API time, network transport, and browser rendering. Completed throughput includes all 1,000 records; correct-model throughput counts only correct model records.

| Condition | Model | p50 ms | p95 ms | p99 ms | Completed records/s | Correct model records/s |
|---|---|---:|---:|---:|---:|---:|
| test-c1 | JSONLLM | 223.419 [218.905, 245.271] | 1421.418 [1391.579, 1516.254] | 1708.927 [1636.463, 2108.029] | 2.624 [2.428, 2.677] | 2.099 [1.942, 2.141] |
| test-c1 | Original Qwen | 227.145 [223.221, 228.997] | 2183.151 [2114.558, 2214.028] | 2600.052 [2553.035, 2811.448] | 2.267 [2.237, 2.286] | 1.360 [1.345, 1.374] |
| test-c8 | JSONLLM | 4862.183 [4409.787, 5690.534] | 6041.267 [5142.941, 9151.745] | 7058.047 [6014.541, 9776.886] | 1.983 [1.624, 2.241] | 1.587 [1.299, 1.793] |
| test-c8 | Original Qwen | 5641.818 [4961.752, 6213.303] | 7954.604 [7179.846, 10211.263] | 9261.575 [8185.580, 12058.484] | 1.735 [1.550, 1.981] | 1.041 [0.930, 1.191] |

Concurrency 8 increased latency and reduced throughput relative to concurrency 1. Records are serialized by a session lock; intra-record field batching does not imply efficient inter-record serving. In the single validation trial, JSONLLM p50 increased to 240.614 ms from 230.363 ms, while p95 decreased to 1,528.391 ms from 2,127.901 ms.

## Resources

| Condition | Model | Measured wall seconds | Output tokens/trial | Peak allocated GPU GiB | Load seconds | Warmup seconds |
|---|---|---:|---:|---:|---:|---:|
| test-c1 | JSONLLM | 381.118 [373.604, 411.896] | 4348.000 [4348.000, 4348.000] | 8.216 [8.216, 8.216] | 153.749 [78.377, 158.922] | 28.394 [27.003, 29.743] |
| test-c1 | Original Qwen | 441.091 [437.396, 446.944] | 4931.000 [4931.000, 4932.000] | 8.216 [8.216, 8.216] | 116.638 [76.507, 127.687] | 27.293 [26.555, 31.803] |
| test-c8 | JSONLLM | 504.210 [446.297, 615.922] | 4348.000 [4348.000, 4348.000] | 8.435 [8.435, 8.435] | 122.960 [81.583, 147.284] | 28.246 [24.161, 34.030] |
| test-c8 | Original Qwen | 576.533 [504.694, 645.085] | 4931.000 [4931.000, 4932.000] | 8.435 [8.435, 8.435] | 72.974 [51.949, 108.530] | 26.262 [24.998, 31.374] |

Memory is PyTorch peak allocated memory, not total device use or a supported minimum GPU size. Token counts describe the decision/scalar interface and are not directly comparable to serialized whole-JSON output tokens.

## Latency by record type

These percentiles pool all five trials, unlike the per-trial medians above. Semantic records use constrained decisions; generation records include a short factual sentence. Code records do not invoke the model.

| Condition | Model | Type | Count | p50 ms | p95 ms | p99 ms |
|---|---|---|---:|---:|---:|---:|
| test-c1 | JSONLLM | semantic | 3000 | 176.725 | 348.737 | 454.004 |
| test-c1 | JSONLLM | generation | 1000 | 1309.772 | 1713.404 | 2295.966 |
| test-c1 | JSONLLM | code | 1000 | 0.340 | 0.412 | 0.599 |
| test-c1 | Original Qwen | semantic | 3000 | 176.819 | 348.762 | 392.385 |
| test-c1 | Original Qwen | generation | 1000 | 1546.043 | 2581.933 | 3141.312 |
| test-c1 | Original Qwen | code | 1000 | 0.350 | 0.399 | 0.573 |
| test-c8 | JSONLLM | semantic | 3000 | 4893.877 | 7453.264 | 9155.128 |
| test-c8 | JSONLLM | generation | 1000 | 4919.201 | 7412.095 | 9104.781 |
| test-c8 | JSONLLM | code | 1000 | 0.421 | 0.659 | 0.828 |
| test-c8 | Original Qwen | semantic | 3000 | 5610.440 | 8502.267 | 10852.739 |
| test-c8 | Original Qwen | generation | 1000 | 5618.969 | 8223.002 | 10474.386 |
| test-c8 | Original Qwen | code | 1000 | 0.398 | 0.659 | 0.863 |

## Collection pilot and experiment budget

The accepted synthetic-data throughput pilot used 64 API workers over a five-minute confirmation window: 724.6 completed requests/minute and 378 accepted records/minute; API p50/p95 were 892.083/1,315.141 ms. The error rate was 0.6581%, with zero HTTP 429 responses and zero failed groups. This measures the teacher API collection process, not JSONLLM inference. Invalid earlier diagnostic probes were excluded from training.

The completed campaign cost estimate was $29.070653516: API conservative accounting $3.645124446 and GPU/storage/recovery operations $25.425529070. API accounting retains $0.638976 for two unresolved requests. The estimate includes failed attempts, installation, storage, transfer, recovery, and cleanup; it is not a final provider invoice or a serving-cost estimate.

## Limits and next experiments

The restricted synthetic benchmark does not establish general factual reliability, unseen schema performance, arbitrary tree generation, workflow synthesis, or a causal speedup from shared context. No additional paid API call or GPU performance run was made to prepare this repository. See [the follow-up plan](benchmark-plan.md).

Per-trial timings, token counts, memory, quality, and pooled type statistics are in [evaluation.json](evaluation.json). [Provenance](provenance.md) records the frozen hashes and terms references. Operational logs and account details are excluded.
