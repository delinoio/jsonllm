# License and provenance

The JSONLLM code, model derivative, and this synthetic dataset are prepared under Apache-2.0. [LICENSE](../LICENSE) contains the exact Apache license file distributed by the upstream model. [NOTICE](../NOTICE) identifies the project and upstream origin. Third-party dependencies retain their own licenses.

## Base model and changes

The base is [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Its [license at that revision](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/LICENSE) is Apache-2.0. The upstream revision contains no separate NOTICE file.

One LoRA was trained from the original base and merged into text-model weights. The root weights are a modified derivative, not the original Qwen checkpoint. The original adapter and its base revision are retained under `adapter/`. The package was extracted from the author's experimental source at `1be7e10281bfcf87149b44e3168c45cd51febecd`, renamed from `typellm_model` to `jsonllm`, and given a new repository history, CLI, examples, and English documentation. Model/data bytes and prompt strings were preserved.

## Synthetic-data rights basis

[DeepSeek Open Platform Terms of Service §4.2](https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html) assigns any output rights to the user and permits derivative products and training other models, subject to the terms and applicable law. This is the recorded basis for using and distributing the synthetic outputs. It does not turn generated claims into verified facts or remove input/output responsibilities under §4.1.

Generation used the official DeepSeek provider through OpenRouter. Provider names identify provenance and do not imply endorsement. The API terms version inspected has release date April 22, 2026 and effective date April 29, 2026. Captured URLs, resolved URLs, dates, and SHA-256 hashes for the API terms, general terms, routing terms, provider page, and endpoint metadata are in [provenance.json](provenance.json). Raw provider/account logs and operational screenshots are not part of this release.

## Frozen identities

| Artifact | SHA-256 or Git commit |
|---|---|
| Canonical dataset | `48fdc7054fa7e93795196ecbed7fa79ae2e9b3ca7e2c86efcd0a46afb530a70c` |
| Candidate selection seal | `24495a5ca31df9a30600411d81af7409d53ac83ccd7a1542e37cf2ac3f167d44` |
| Merged weights | `3ee9a015cec965261b103ceb0fd469eb24fb45aadbb0f0c0aaae9d2c9c1c1b10` |
| Original LoRA | `9896496bfbd28cfba3f8386b7eec86943f41047768ce4bf02a76c4c0428308fe` |
| Original base revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |

The Hugging Face release commit is a separate identifier from the base revision. GitHub examples pin the former through `jsonllm.release.MODEL_REVISION`. The selection seal identifies the private experiment's pre-test decision record; it is not a hash of the repackaged model card. The release manifest covers the distributed files instead.
