"""Transformers/PEFT BF16 LoRA on Qwen's text model."""

import math
import time
from pathlib import Path

from ..io import write_json
from ..prompts import padded_batch


def load_model(model, revision, device="cuda"):
    import torch
    from transformers import AutoConfig, Qwen3_5ForCausalLM

    config = AutoConfig.from_pretrained(model, revision=revision, trust_remote_code=False)
    if config.model_type not in {"qwen3_5", "qwen3_5_text"}:
        raise ValueError("The CUDA baseline supports Qwen3.5 dense text backbones")
    text_config = getattr(config, "text_config", config)
    loaded, info = Qwen3_5ForCausalLM.from_pretrained(
        model,
        revision=revision,
        config=text_config,
        dtype=torch.bfloat16,
        output_loading_info=True,
        trust_remote_code=False,
    )
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise ValueError(f"Incomplete text checkpoint loading: {info}")
    return loaded.to(device)


class Collator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, examples):
        import torch

        return {
            key: torch.tensor(value) for key, value in padded_batch(examples, self.pad_id).items()
        }


def train(config, revision, tokenizer, datasets, *, resume=None):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA training requires an NVIDIA GPU with BF16 support")
    set_seed(config.seed)
    model = load_model(config.model, revision)
    targets = [
        name
        for name, module in model.named_modules()
        if name.startswith("model.layers.") and isinstance(module, torch.nn.Linear)
    ]
    if not targets:
        raise ValueError("No text decoder linear layers found")
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            revision=revision,
            r=config.rank,
            lora_alpha=config.alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=targets,
        ),
    )
    model.config.use_cache = False

    class FiniteLoss(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            for key in ("loss", "eval_loss"):
                if key in (logs or {}) and not math.isfinite(logs[key]):
                    raise ValueError(f"Non-finite {key}")

    arguments = TrainingArguments(
        output_dir=config.output,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        num_train_epochs=config.epochs,
        max_steps=min(
            config.max_steps,
            math.ceil(
                math.ceil(len(datasets["train"]) / config.batch_size) / config.gradient_accumulation
            )
            * config.epochs,
        )
        if config.max_steps
        else -1,
        learning_rate=config.learning_rate,
        lr_scheduler_type="constant",
        weight_decay=0.0,
        bf16=True,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=2,
        eval_strategy="epoch",
        logging_steps=1,
        logging_nan_inf_filter=False,
        report_to="none",
        seed=config.seed,
        data_seed=config.seed,
        remove_unused_columns=False,
        optim="adamw_torch",
        dataloader_num_workers=0,
    )
    trainer_class = Trainer
    if config.prompt_version == "shared-context-v3":
        from .supervised_loss import supervised_loss

        class SupervisedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                loss, outputs = supervised_loss(
                    model, inputs, num_items_in_batch=num_items_in_batch
                )
                return (loss, outputs) if return_outputs else loss

        trainer_class = SupervisedTrainer
    trainer = trainer_class(
        model=model,
        args=arguments,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=Collator(tokenizer),
        callbacks=[FiniteLoss()],
    )
    if config.prompt_version == "shared-context-v3":
        # Trainer sums valid labels across microbatches; the loss uses that same denominator.
        trainer.model_accepts_loss_kwargs = True
    result = trainer.train(resume_from_checkpoint=resume)
    metrics = {**result.metrics, **trainer.evaluate()}
    if any(not math.isfinite(v) for v in metrics.values() if isinstance(v, float)):
        raise ValueError("Training produced non-finite metrics")
    model.save_pretrained(config.output)
    write_json(Path(config.output) / "metrics.json", metrics)
    return metrics


class Predictor:
    def __init__(self, model, revision, adapter=None, *, last_token_only=False):
        import torch

        from ..training import load_tokenizer

        if not torch.cuda.is_available():
            raise ValueError("CUDA evaluation requires an NVIDIA GPU")
        self.model = load_model(model, revision)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.tokenizer = load_tokenizer(model, revision)
        self.last_token_only = last_token_only

    def scores(self, ids, candidates):
        import torch

        with torch.inference_mode():
            options = {"logits_to_keep": 1} if self.last_token_only else {}
            logits = self.model(input_ids=torch.tensor([ids], device="cuda"), **options).logits[
                0, -1
            ]
            return logits[candidates].float().cpu().tolist()

    def generate(self, ids, max_tokens):
        return self.generate_details(ids, max_tokens)["text"]

    def synchronize(self):
        import torch

        torch.cuda.synchronize()

    def reset_peak_memory(self):
        import torch

        torch.cuda.reset_peak_memory_stats()

    def peak_memory_bytes(self):
        import torch

        return torch.cuda.max_memory_allocated()

    def generate_details(self, ids, max_tokens):
        import torch

        class FirstToken:
            prompt = True
            first = None

            def put(self, value):
                if self.prompt:
                    self.prompt = False
                elif self.first is None:
                    self.first = time.perf_counter()

            def end(self):
                pass

        streamer = FirstToken()
        self.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=torch.tensor([ids], device="cuda"),
                attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device="cuda"),
                do_sample=False,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                streamer=streamer,
                use_cache=True,
            )
        self.synchronize()
        elapsed = time.perf_counter() - started
        tokens = output[0, len(ids) :].tolist()
        eos = self.tokenizer.eos_token_id
        return {
            "text": self.tokenizer.decode(tokens, skip_special_tokens=True),
            "output_tokens": len(tokens),
            "ttft_seconds": streamer.first - started if streamer.first is not None else None,
            "generation_seconds": elapsed,
            "finish_reason": "eos" if tokens and tokens[-1] == eos else "length",
        }
