"""MLX LoRA with explicit labels shared with the CUDA collator."""

import math
import random
import time
from pathlib import Path

from ..io import append_jsonl, write_json


def load_base(model_id, revision):
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm import load

    model, tokenizer = load(model_id, revision=revision)
    if model.model_type != "qwen3_5":
        raise ValueError("The MLX baseline supports Qwen3.5 dense text backbones")
    if any("Quantized" in type(module).__name__ for _, module in model.named_modules()):
        raise ValueError("This baseline expects unquantized BF16 weights")
    # Qwen's recurrent A_log parameters intentionally remain in full precision.
    predicate = getattr(model, "cast_predicate", lambda _: True)
    model.update(
        tree_unflatten(
            [
                (
                    name,
                    value.astype(mx.bfloat16)
                    if predicate(name) and mx.issubdtype(value.dtype, mx.floating)
                    else value,
                )
                for name, value in tree_flatten(model.parameters())
            ]
        )
    )
    return model, tokenizer


def add_lora(model, rank, alpha):
    import mlx.nn as nn
    from mlx_lm.tuner.utils import linear_to_lora_layers

    keys = sorted(
        {
            name
            for layer in model.layers
            for name, module in layer.named_modules()
            if isinstance(module, nn.Linear)
        }
    )
    if not keys:
        raise ValueError("No text decoder linear layers found")
    model.freeze()
    params = {"rank": rank, "scale": alpha / rank, "dropout": 0.0, "keys": keys}
    linear_to_lora_layers(model, len(model.layers), params)
    return {"fine_tune_type": "lora", "num_layers": len(model.layers), "lora_parameters": params}


def loss_fn(model, inputs, labels, positions=None):
    import mlx.core as mx
    import mlx.nn as nn

    targets = labels[:, 1:]
    if positions is not None:
        # The frozen vocabulary head need only project positions that contribute to the loss.
        # All prefix hidden states still participate in the decoder and its gradients.
        text = model.language_model
        hidden = text.model(inputs[:, :-1])[:, positions, :]
        logits = (
            text.model.embed_tokens.as_linear(hidden)
            if text.args.tie_word_embeddings
            else text.lm_head(hidden)
        ).astype(mx.float32)
        return nn.losses.cross_entropy(logits, targets[:, positions]).mean()
    logits = model(inputs[:, :-1]).astype(mx.float32)
    mask = targets != -100
    safe_targets = mx.where(mask, targets, 0)
    loss = nn.losses.cross_entropy(logits, safe_targets)
    return (loss * mask).sum() / mask.sum()


def train_loop(model, config, tokenizer, datasets):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map

    rng = random.Random(config.seed)
    mx.random.seed(config.seed)
    optimizer = optim.AdamW(learning_rate=config.learning_rate, weight_decay=0.0)
    grad_fn = nn.value_and_grad(model, loss_fn)
    output = Path(config.output)
    step, losses, pending, gradients = 0, [], 0, None
    started = time.perf_counter()
    processed_examples, processed_tokens = 0, 0
    model.train()
    for _ in range(config.epochs):
        indices = list(range(len(datasets["train"])))
        rng.shuffle(indices)
        for start in range(0, len(indices), config.batch_size):
            rows = [datasets["train"][i] for i in indices[start : start + config.batch_size]]
            # Process each length separately: Qwen's recurrent layers must not see padding.
            # Weight microbatch gradients by supervised tokens, matching the loss contract.
            token_counts = [sum(x != -100 for x in row["labels"][1:]) for row in rows]
            micro_grad, micro_loss = None, 0.0
            for row, count in zip(rows, token_counts, strict=True):
                loss, grads = grad_fn(
                    model,
                    mx.array([row["input_ids"]]),
                    mx.array([row["labels"]]),
                    mx.array([i for i, label in enumerate(row["labels"][1:]) if label != -100]),
                )
                processed_examples += 1
                processed_tokens += len(row["input_ids"])
                weight = count / sum(token_counts)
                grads = tree_map(lambda g, w=weight: g * w, grads)
                micro_grad = (
                    grads if micro_grad is None else tree_map(lambda a, b: a + b, micro_grad, grads)
                )
                mx.eval(micro_grad)
                micro_loss += loss.item() * weight
            if not math.isfinite(micro_loss):
                raise ValueError("Non-finite training loss")
            gradients = (
                micro_grad
                if gradients is None
                else tree_map(lambda a, b: a + b, gradients, micro_grad)
            )
            # Materialize accumulated gradients so old activation graphs do not grow with steps.
            mx.eval(gradients)
            pending += 1
            losses.append(micro_loss)
            if pending == config.gradient_accumulation or start + len(rows) == len(indices):
                optimizer.update(model, tree_map(lambda g, n=pending: g / n, gradients))
                mx.eval(model.parameters(), optimizer.state)
                step += 1
                elapsed = time.perf_counter() - started
                progress = {
                    "step": step,
                    "train_loss": sum(losses[-pending:]) / pending,
                    "examples": processed_examples,
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": processed_tokens / elapsed,
                    "peak_memory_gb": mx.get_peak_memory() / 1e9,
                }
                append_jsonl(output / "metrics.jsonl", progress)
                if step == 1 or step % 100 == 0:
                    print(progress, flush=True)
                gradients, pending = None, 0
                if step % config.save_steps == 0:
                    mx.save_safetensors(
                        str(output / f"step-{step}-adapters.safetensors"),
                        dict(tree_flatten(model.trainable_parameters())),
                    )
                if config.max_steps is not None and step >= config.max_steps:
                    break
        if config.max_steps is not None and step >= config.max_steps:
            break
    model.eval()
    total, tokens = 0.0, 0
    for row in datasets["validation"]:
        count = sum(label != -100 for label in row["labels"][1:])
        total += (
            loss_fn(
                model,
                mx.array([row["input_ids"]]),
                mx.array([row["labels"]]),
                mx.array([i for i, label in enumerate(row["labels"][1:]) if label != -100]),
            ).item()
            * count
        )
        tokens += count
    val_loss = total / tokens
    if not math.isfinite(val_loss):
        raise ValueError("Non-finite validation loss")
    mx.save_safetensors(
        str(output / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters()))
    )
    metrics = {
        "optimizer_steps": step,
        "train_loss": sum(losses) / len(losses),
        "validation_loss": val_loss,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
    }
    append_jsonl(output / "metrics.jsonl", metrics)
    return metrics


def train(config, revision, tokenizer, datasets, *, warm_start=None):
    import mlx.core as mx

    mx.random.seed(config.seed)
    model, _ = load_base(config.model, revision)
    adapter_config = add_lora(model, config.rank, config.alpha)
    if warm_start:
        # load_adapters also installs modules; use only its validated tensors on fresh modules.
        from mlx.utils import tree_flatten

        weights = mx.load(str(Path(warm_start) / "adapters.safetensors"))
        expected = dict(tree_flatten(model.trainable_parameters()))
        if weights.keys() != expected.keys() or any(
            weights[k].shape != expected[k].shape for k in weights
        ):
            raise ValueError("Warm-start adapter tensors do not match this model")
        model.load_weights(list(weights.items()), strict=False)
    write_json(Path(config.output) / "adapter_config.json", adapter_config)
    originals = {}
    try:
        if config.gradient_checkpointing:
            from mlx_lm.tuner.trainer import grad_checkpoint

            for layer in model.layers:
                cls = type(layer)
                if cls not in originals:
                    originals[cls] = cls.__call__
                    grad_checkpoint(layer)
        return train_loop(model, config, tokenizer, datasets)
    finally:
        # mlx-lm wraps a class method; restore it before evaluation or another training run.
        for cls, original in originals.items():
            cls.__call__ = original


class Predictor:
    def __init__(self, model, revision, adapter=None):
        import mlx.core as mx

        self.model, self.tokenizer = load_base(model, revision)
        if adapter:
            from mlx_lm.tuner.utils import load_adapters

            self.model = load_adapters(self.model, adapter)
        self.model.eval()
        mx.eval(self.model.parameters())

    def scores(self, ids, candidates):
        import mlx.core as mx

        logits = self.model(mx.array([ids]))[0, -1].astype(mx.float32)
        return logits[mx.array(candidates)].tolist()

    def generate(self, ids, max_tokens):
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        return generate(
            self.model,
            self.tokenizer,
            prompt=ids,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0),
        )

    def synchronize(self):
        import mlx.core as mx

        mx.synchronize()

    def reset_peak_memory(self):
        import mlx.core as mx

        mx.reset_peak_memory()

    def peak_memory_bytes(self):
        import mlx.core as mx

        return mx.get_peak_memory()

    def generate_details(self, ids, max_tokens):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        self.synchronize()
        started = time.perf_counter()
        first, parts, response = None, [], None
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=ids,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0),
        ):
            if first is None:
                first = time.perf_counter() - started
            parts.append(response.text)
        self.synchronize()
        return {
            "text": "".join(parts),
            "output_tokens": response.generation_tokens if response else 0,
            "ttft_seconds": first,
            "generation_seconds": time.perf_counter() - started,
            "finish_reason": response.finish_reason if response else "empty",
        }
