"""Actual pinned 4B LoRA loss/gradient equivalence before the two-step training smoke test."""

import argparse
from pathlib import Path

from jsonllm.backends.cuda import load_model
from jsonllm.backends.supervised_loss import supervised_loss
from jsonllm.config import load_config
from jsonllm.io import read_jsonl, write_json
from jsonllm.prompts import encode_example
from jsonllm.training import load_tokenizer


def main():
    import torch
    from peft import LoraConfig, get_peft_model

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    config = load_config(args.config)
    torch.manual_seed(42)
    base = load_model(config.model, config.revision)
    targets = [
        name
        for name, module in base.named_modules()
        if name.startswith("model.layers.") and isinstance(module, torch.nn.Linear)
    ]
    model = get_peft_model(
        base,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=32,
            lora_dropout=0,
            bias="none",
            target_modules=targets,
            revision=config.revision,
        ),
    ).train()
    tokenizer = load_tokenizer(config.model, config.revision)
    # Three target types, with no test input or previous adapter involved.
    examples = read_jsonl(Path(config.data) / "train.jsonl")
    chosen = [
        examples[0],
        next(e for e in examples if e["schema"]["type"] == "boolean"),
        next(e for e in examples if not e["choices"]),
    ]
    report = []
    for example in chosen:
        encoded = encode_example(example, tokenizer)
        inputs = {k: torch.tensor([v], device="cuda") for k, v in encoded.items()}
        model.zero_grad(set_to_none=True)
        reference = model(**inputs, use_cache=False).loss
        reference.backward()
        gradients = {
            n: p.grad.detach().float().cpu().clone()
            for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        }
        model.zero_grad(set_to_none=True)
        sparse, _ = supervised_loss(model, inputs)
        sparse.backward()
        square_error, square_ref, square_new, dot, max_delta = 0.0, 0.0, 0.0, 0.0, 0.0
        for name, parameter in model.named_parameters():
            if name not in gradients:
                continue
            if parameter.grad is None:
                raise ValueError("A supervised gradient disappeared: " + name)
            before = gradients[name]
            after = parameter.grad.detach().float().cpu()
            square_error += (before - after).square().sum().item()
            square_ref += before.square().sum().item()
            square_new += after.square().sum().item()
            dot += (before * after).sum().item()
            max_delta = max(max_delta, (before - after).abs().max().item())
        cosine = dot / max((square_ref * square_new) ** 0.5, 1e-30)
        relative = (square_error / max(square_ref, 1e-30)) ** 0.5
        delta = abs(reference.item() - sparse.item())
        passed = delta <= 0.03 and relative <= 0.03 and cosine >= 0.999
        report.append(
            {
                "field": example["field"],
                "record_id": example["record_id"],
                "full_loss": reference.item(),
                "selective_loss": sparse.item(),
                "loss_delta": delta,
                "gradient_relative_l2": relative,
                "gradient_cosine": cosine,
                "gradient_max_delta": max_delta,
                "gradient_tensors": len(gradients),
                "passed": passed,
            }
        )
        write_json(
            args.output,
            {
                "bf16_tolerances": {
                    "loss_abs": 0.03,
                    "gradient_relative_l2": 0.03,
                    "cosine_min": 0.999,
                },
                "probes": report,
                "passed": all(r["passed"] for r in report),
            },
        )
        if not passed:
            raise ValueError("Selective BF16 loss/gradient equivalence failed")


if __name__ == "__main__":
    main()
