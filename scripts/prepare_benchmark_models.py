"""Download frozen releases and verify an unchanged Qwen text-only export."""

import argparse
from pathlib import Path

from jsonllm.artifacts import file_hash
from jsonllm.checkpoint_identity import text_checkpoint_keys
from jsonllm.io import write_json
from jsonllm.release import BASE_MODEL, BASE_REVISION, MODEL_ID, MODEL_REVISION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    args = p.parse_args()
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    from jsonllm.backends.cuda import load_model

    patterns = [
        "model*.safetensors",
        "model.safetensors.index.json",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "preprocessor_config.json",
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    for name, repo, revision in [
        ("base", BASE_MODEL, BASE_REVISION),
        ("jsonllm", MODEL_ID, MODEL_REVISION),
    ]:
        raw = Path(snapshot_download(repo, revision=revision, allow_patterns=patterns))
        source = {f.name: file_hash(f) for f in raw.iterdir() if f.is_file()}
        output = args.output / name
        audit = {"repo": repo, "revision": revision, "source_sha256": source}
        if name == "base":
            model = load_model(str(raw), revision, device="cpu")
            if not (output / "model.safetensors").exists():
                model.save_pretrained(output, max_shard_size="10GB", safe_serialization=True)
            for f in raw.iterdir():
                if f.is_file() and (
                    "token" in f.name
                    or f.name in ["chat_template.jinja", "vocab.json", "merges.txt"]
                ):
                    if not (output / f.name).exists():
                        (output / f.name).symlink_to(f)
            state = model.state_dict()
            with safe_open(output / "model.safetensors", framework="pt") as exported:
                mapping = text_checkpoint_keys(
                    state, exported.keys(), tied_embeddings=model.config.tie_word_embeddings
                )
                for key, tensor in state.items():
                    target = exported.get_tensor(mapping[key])
                    if target.dtype != tensor.dtype or not torch.equal(tensor, target):
                        raise ValueError(f"Changed tensor: {key}")
            audit.update(
                base_text_export_tensor_equal=True,
                verified_state_tensors=len(state),
                serialization_key_map=mapping,
                export_sha256=file_hash(output / "model.safetensors"),
            )
            del model, state
            # Verify the ordinary inference loader reconstructs every parameter, including ties.
            reloaded = load_model(str(output), revision, device="cpu")
            audit["reload_succeeded"] = True
            del reloaded
        else:
            if not output.exists():
                output.symlink_to(raw)
            if (
                source["model.safetensors"]
                != "3ee9a015cec965261b103ceb0fd469eb24fb45aadbb0f0c0aaae9d2c9c1c1b10"
            ):
                raise ValueError("Frozen JSONLLM model SHA changed")
        write_json(args.audit / (name + "-model-integrity.json"), audit)


if __name__ == "__main__":
    main()
