"""Explicit Qwen3.5 prefix-state forks; no result cache or cross-record prefix reuse."""

import copy
import threading
import time
from collections import defaultdict

from ..fast_inference import decode_result
from ..prompts import SHARED_VERSION, context_ids, field_example, question_ids
from .stage_profile import StageProfiler

INFERENCE_DTYPE = "float16"


def inference_dtype(engine):
    # The campaign's separate vLLM server explicitly requests BF16.
    return "bfloat16" if engine == "vllm" else INFERENCE_DTYPE


def fork_cache(cache, batch_size, device):
    """Copy mutable recurrent/conv state and attention storage without copying weights."""
    import torch

    branch = copy.copy(cache)
    branch.layers = []
    copied_bytes = 0
    indices = torch.zeros(batch_size, dtype=torch.long, device=device)
    for original in cache.layers:
        layer = copy.copy(original)
        # Cache metadata dictionaries are mutable too (not just their tensor values).
        for key, value in vars(original).items():
            if isinstance(value, dict):
                setattr(layer, key, value.copy())
        layer.reorder_cache(indices)
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)
            if tensor is not None:
                copied_bytes += tensor.numel() * tensor.element_size()
        for name in ("conv_states", "recurrent_states"):
            copied_bytes += sum(
                t.numel() * t.element_size()
                for t in getattr(layer, name, {}).values()
                if t is not None
            )
        branch.layers.append(layer)
    return branch, copied_bytes


def candidate_logits(hidden, weight, token_ids):
    import torch
    import torch.nn.functional as functional

    indices = torch.tensor(token_ids, dtype=torch.long, device=hidden.device)
    return functional.linear(hidden, weight.index_select(0, indices)).float()


class SharedPredictor:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        share=True,
        max_length=2048,
        compile_model=False,
        bucket_choices=False,
        profile_stages=False,
    ):
        import torch

        self.model = model.eval()
        self.tokenizer = tokenizer
        self.share, self.max_length = share, max_length
        self.device = next(model.parameters()).device
        self.inference_dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
        self.lock = threading.Lock()
        self.bucket_choices = bucket_choices
        self.profile_stages = profile_stages
        self.grammar = None
        self.backbone = torch.compile(model.model, dynamic=True) if compile_model else model.model

    @classmethod
    def load(cls, path, revision="main", **kwargs):
        import torch

        from ..training import load_tokenizer
        from .cuda import load_model

        # Match the full H100 precision probe: load BF16, then cast weights AND
        # buffers. Training and stored checkpoints keep their original precision.
        model = load_model(path, revision).to(dtype=getattr(torch, INFERENCE_DTYPE))
        invalid = [
            name
            for name, parameter in model.named_parameters()
            if not torch.isfinite(parameter).all()
        ]
        if invalid:
            raise ValueError(f"Inference conversion produced nonfinite weights: {invalid}")
        return cls(model, load_tokenizer(path, revision), **kwargs)

    def synchronize(self):
        if self.device.type == "cuda":
            import torch

            torch.cuda.synchronize(self.device)

    def open_record(self, context, decision_count):
        if decision_count < 1:
            raise ValueError("A model session must have at least one decision")
        return SharedSession(self, context, decision_count)


class SharedSession:
    def __init__(self, predictor, context, decision_count):
        self.predictor = predictor
        self.closed = False
        self.cancelled = threading.Event()
        self.gpu_events = []
        self.cache = None
        self.use_shared = predictor.share and decision_count > 1
        start = time.perf_counter()
        self.prefix = context_ids(context, predictor.tokenizer)
        self.metrics = {
            "context_tokenizations": 1,
            "common_tokens": len(self.prefix),
            "common_prefills": 0,
            "common_token_evaluations": 0,
            "cache_copy_bytes": 0,
            "cache_copy_seconds": 0.0,
            "forwards": [],
            "calls": [],
            "tokenize_seconds": time.perf_counter() - start,
        }
        self.profile = StageProfiler(
            predictor.profile_stages, predictor.device, self.metrics, self.gpu_events
        )
        with self.profile("lock_wait"):
            lock_started = time.perf_counter()
            predictor.lock.acquire()
            self.metrics["lock_wait_seconds"] = time.perf_counter() - lock_started
        self.lock_owned = True

    def _forward(self, tokens, cache, *, kind, lengths=None):
        import torch

        if self.cancelled.is_set():
            raise ValueError("cancelled")
        batch, width = tokens.shape
        old = cache.get_seq_length() if cache is not None else 0
        mask = None
        if lengths is not None and len(set(lengths)) > 1:
            mask = torch.tensor(
                [[1] * (old + n) + [0] * (width - n) for n in lengths],
                device=tokens.device,
                dtype=torch.long,
            )
        self.metrics["forwards"].append(
            {
                "kind": kind,
                "batch": batch,
                "tokens": width,
                "lengths": lengths,
                "cached_tokens": old,
            }
        )
        start, end = self._start_gpu_timer(kind)
        output = self.predictor.backbone(
            input_ids=tokens, attention_mask=mask, past_key_values=cache, use_cache=True
        )
        if end is not None:
            end.record()
            self.gpu_events.append((kind, start, end))
        return output

    def _start_gpu_timer(self, kind):
        import torch

        if self.predictor.device.type != "cuda":
            return None, None
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        return start, end

    def _prefill(self):
        import torch

        if self.cache is None:
            tokens = torch.tensor([self.prefix], device=self.predictor.device)
            self.cache = self._forward(tokens, None, kind="common").past_key_values
            self.metrics["common_prefills"] += 1
            self.metrics["common_token_evaluations"] += len(self.prefix)

    def _batch(self, inputs, *, selection):
        import torch

        predictor = self.predictor
        if self.use_shared:
            self._prefill()
            started = time.perf_counter()
            start, end = self._start_gpu_timer("cache_copy")
            cache, nbytes = fork_cache(self.cache, len(inputs), predictor.device)
            if end is not None:
                end.record()
                self.gpu_events.append(("cache_copy", start, end))
            self.metrics["cache_copy_bytes"] += nbytes
            self.metrics["cache_copy_seconds"] += time.perf_counter() - started
        else:
            cache = None
            inputs = [self.prefix + suffix for suffix in inputs]
            self.metrics["common_prefills"] += len(inputs)
            self.metrics["common_token_evaluations"] += len(self.prefix) * len(inputs)
        lengths = list(map(len, inputs))
        if not selection and len(set(lengths)) != 1:
            raise ValueError("Generation must bucket exact lengths to preserve recurrent state")
        width = max(lengths)
        tokens = torch.tensor(
            [x + [self.predictor.tokenizer.pad_token_id] * (width - len(x)) for x in inputs],
            device=predictor.device,
        )
        output = self._forward(tokens, cache, kind="questions", lengths=lengths)
        indices = torch.tensor(lengths, device=predictor.device) - 1
        hidden = output.last_hidden_state[
            torch.arange(len(inputs), device=predictor.device), indices
        ]
        return hidden, output.past_key_values

    def _compact_generation(self, cache, active, tokens, keep):
        """Retain row order; identity compaction needs neither copies nor indices."""
        import torch

        if len(keep) == len(active):
            return active, tokens
        with self.profile("cache_reorder", gpu=True):
            indices = torch.tensor(keep, dtype=torch.long, device=tokens.device)
            cache.reorder_cache(indices)
            selected = tokens.index_select(0, indices)
        if self.profile.enabled:
            copied = 0
            for layer in cache.layers:
                for name in ("keys", "values", "conv_states", "recurrent_states"):
                    value = getattr(layer, name, None)
                    tensors = value.values() if isinstance(value, dict) else [value]
                    copied += sum(t.numel() * t.element_size() for t in tensors if t is not None)
            self.profile.add("cache_reorder_count", 1)
            self.profile.add("cache_reorder_bytes", copied)
        return [active[row] for row in keep], selected

    def predict(self, record, names, answers, max_tokens):
        import torch

        if self.closed:
            raise ValueError("Shared session is closed")
        started = time.perf_counter()
        examples = [
            field_example(record, name, dependency_answers=answers, prompt_version=SHARED_VERSION)
            for name in names
        ]
        inputs = [question_ids(e, self.predictor.tokenizer) for e in examples]
        self.metrics["tokenize_seconds"] += time.perf_counter() - started
        if any(
            len(self.prefix) + len(ids) + (1 if e["choices"] else max_tokens)
            > self.predictor.max_length
            for ids, e in zip(inputs, examples, strict=True)
        ):
            raise ValueError("overlength")
        values = {}
        with torch.inference_mode():
            selection = [i for i, e in enumerate(examples) if e["choices"]]
            choice_groups = defaultdict(list)
            for i in selection:
                choice_groups[len(inputs[i]) if self.predictor.bucket_choices else 0].append(i)
            for group in choice_groups.values():
                hidden, _ = self._batch([inputs[i] for i in group], selection=True)
                # One projection and host transfer for every field in this GPU batch.
                labels = max((list(examples[i]["choices"]) for i in group), key=len)
                ids = [
                    self.predictor.tokenizer.encode(label, add_special_tokens=False)[0]
                    for label in labels
                ]
                with self.profile("lm_head", gpu=True):
                    scores = candidate_logits(hidden, self.predictor.model.lm_head.weight, ids)
                counts = torch.tensor(
                    [len(examples[i]["choices"]) for i in group], device=hidden.device
                )
                scores.masked_fill_(
                    torch.arange(len(labels), device=hidden.device)[None, :] >= counts[:, None],
                    -float("inf"),
                )
                with self.profile("choice_readback", gpu=True):
                    probabilities_batch = scores.softmax(-1).cpu().tolist()
                for row, index in enumerate(group):
                    probabilities = probabilities_batch[row][: len(examples[index]["choices"])]
                    raw = labels[max(range(len(probabilities)), key=probabilities.__getitem__)]
                    values[names[index]] = examples[index]["choices"][raw]
                    self.metrics["calls"].append(
                        {
                            "field": names[index],
                            "kind": "selection",
                            "probabilities": probabilities,
                            "output_tokens": 1,
                            "raw": raw,
                        }
                    )
            buckets = defaultdict(list)
            for i, example in enumerate(examples):
                if not example["choices"]:
                    buckets[len(inputs[i])].append(i)
            for group in buckets.values():
                with self.profile("grammar_compile"):
                    if self.predictor.grammar is None:
                        from .scalar_grammar import ScalarGrammar

                        self.predictor.grammar = ScalarGrammar(
                            self.predictor.tokenizer, self.predictor.model.lm_head.weight.shape[0]
                        )
                    grammar = self.predictor.grammar
                    matchers = {i: grammar.matcher(examples[i]["schema"]) for i in group}
                bitmask = grammar.allocate(len(group), self.predictor.device)
                hidden, cache = self._batch([inputs[i] for i in group], selection=False)
                active, generated = list(group), {i: [] for i in group}
                for step in range(max_tokens):
                    with self.profile("lm_head", gpu=True):
                        logits = self.predictor.model.lm_head(hidden)
                    grammar.mask(
                        logits, [matchers[i] for i in active], bitmask, profile=self.profile
                    )
                    tokens = logits.argmax(-1)
                    with self.profile("token_readback", gpu=True):
                        chosen = tokens.cpu().tolist()
                    keep = []
                    for row, (index, token) in enumerate(zip(active, chosen, strict=True)):
                        with self.profile("grammar_accept"):
                            accepted = matchers[index].accept_token(token)
                            terminated = (
                                accepted
                                and token == self.predictor.tokenizer.eos_token_id
                                and matchers[index].is_terminated()
                            )
                        if not accepted:
                            raise ValueError("grammar_rejected_token")
                        if token == self.predictor.tokenizer.eos_token_id:
                            if not terminated:
                                raise ValueError("premature_eos")
                            raw = self.predictor.tokenizer.decode(
                                generated[index], skip_special_tokens=False
                            )
                            response = decode_result(examples[index], raw)
                            if not response["valid"]:
                                raise ValueError(f"invalid_output:{names[index]}")
                            values[names[index]] = response["value"]
                            self.metrics["calls"].append(
                                {
                                    "field": names[index],
                                    "kind": "generation",
                                    "output_tokens": len(generated[index]) + 1,
                                    "raw": raw,
                                }
                            )
                        else:
                            generated[index].append(token)
                            keep.append(row)
                    if not keep:
                        break
                    if step == max_tokens - 1:
                        raise ValueError("truncated_output")
                    active, tokens = self._compact_generation(cache, active, tokens, keep)
                    output = self._forward(tokens[:, None], cache, kind="decode")
                    cache, hidden = output.past_key_values, output.last_hidden_state[:, -1]
        return values

    def cancel(self):
        self.cancelled.set()

    def close(self):
        if not self.closed:
            try:
                self.predictor.synchronize()
                self.metrics["gpu_seconds"] = {}
                for kind, start, end in self.gpu_events:
                    elapsed = start.elapsed_time(end) / 1000
                    self.metrics["gpu_seconds"][kind] = (
                        self.metrics["gpu_seconds"].get(kind, 0) + elapsed
                    )
            finally:
                self.cache = None
                self.closed = True
                if self.lock_owned:
                    self.predictor.lock.release()
                    self.lock_owned = False
