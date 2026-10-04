# Synthetic dataset v0.1

The model repository includes the exact data used in the frozen experiment under `data/`. It is distributed with this project under Apache-2.0. No separate dataset repository is created.

| Split | Records | Groups | Workflow families | English | Korean |
|---|---:|---:|---:|---:|---:|
| Train | 8,000 | 800 | 20 | 4,000 | 4,000 |
| Validation | 1,000 | 100 | 4 | 500 | 500 |
| Test | 1,000 | 100 | 4 | 500 | 500 |

Each group has ten related records: six semantic-decision records, two deterministic-code records, and two generation records. Groups and workflow families are separated across splits. The data seed is `20261004`. Collection completed with zero failed groups after independent review, global duplicate checks, and split integrity checks. The generator remains a restricted synthetic task distribution with eight intent types; held-out families do not establish arbitrary workflow generalization.

## Files and format

- `train.jsonl`, `validation.jsonl`, `test.jsonl`: complete immutable records.
- `prepared/train.jsonl`, `prepared/validation.jsonl`: field-level supervised examples, 15,619 and 1,946 respectively. There is no prepared test training file.
- `splits.json`: sanitized split description and canonical digests.
- `SHA256SUMS`: byte-level hashes of the five JSONL files.

Each source record contains `id`, `group`, `family`, `split`, `kind`, `language`, `context`, `spec`, `state`, `event`, `facts`, `answers`, `expected`, and `step`. The application specification supplies registered structure and rules; labels and expected outputs describe the intended decisions and state transitions. Names and statuses in the records are synthetic task facts. They are not a source of real-world factual knowledge.

The teacher was `deepseek/deepseek-v4.1-flash`, routed through OpenRouter to the official `deepseek` provider only, with fallback disabled, reasoning disabled, and strict JSON output. New sentences and review records were generated for this experiment; earlier provider experiments and diagnostic probes were not reused for training.

The accepted teacher-generated reference and later teacher review are related sources. Review calls can reduce formatting and consistency errors but do not provide independently established truth. The test set is released for reproducibility; future model selection should use a new untouched test set.

## Rights and integrity

The distribution basis is [DeepSeek Open Platform Terms of Service §4.2](https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html): subject to the terms and applicable law, output rights, if any, are assigned to the user, and derivative development and training other models are permitted. Input/output responsibilities remain applicable. See [provenance](provenance.md) for the captured terms hashes and upstream notices.

Canonical dataset digest: `48fdc7054fa7e93795196ecbed7fa79ae2e9b3ca7e2c86efcd0a46afb530a70c`. This digest is computed from canonical JSON split digests; it is not the SHA-256 of concatenated file bytes. Use the release manifest or `SHA256SUMS` for byte-for-byte verification. No source JSONL file was rewritten during packaging.
