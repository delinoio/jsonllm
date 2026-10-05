# Follow-up benchmark plan and status

The approved execution study completed on 2026-10-05. See the [results](design-benchmark.md), [protocol](design-benchmark-protocol.md), and complete raw archive. The original research questions below remain useful context. Broader synthesis, larger models, and real UI usefulness remain untested.

## Isolate the mechanism

Compare whole-JSON constrained generation, serial field decisions, batched independent fields without shared context, and batched fields with shared context. Hold model weights, required information, precision, output constraints, hardware, and correctness criteria fixed. Report both original and fine-tuned weights where a prompt contract requires different training.

## Vary structure and workload

Sweep independent field count, context length, dependency depth, choice vocabulary, generated text length, and number of tree nodes. Include unseen specifications, genuine dynamic topology tasks, and workflow validity tests. Separate hand-authored structure from structure that the model actually generates.

## Measure an application boundary

Report time to first usable decision, full valid-output latency, p50/p95/p99, correct outputs per second, tokens, GPU memory, and error rates. Include cold start and warm execution separately. Compare against an optimized serving baseline with equivalent batching and constraints. Record request arrival rates and queueing behavior instead of interpreting a concurrency setting as throughput by itself.

Use independent quality samples and repeated timing trials. Keep profiling outside latency trials. Freeze candidates before the held-out test. Report quality failures and any accuracy/latency tradeoff. The existing test set should remain a historical reference, not a source for selecting new candidates.
