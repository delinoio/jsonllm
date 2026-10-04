import pytest

from jsonllm.backends.shared_cuda import SharedPredictor, candidate_logits, fork_cache

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


class Tokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def encode(self, text, **kwargs):
        return [1 + ord(c) % 100 for c in text]


@pytest.fixture
def model():
    torch.manual_seed(42)
    config = transformers.Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        vocab_size=128,
        eos_token_id=127,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [2, 3, 3],
            "mrope_interleaved": True,
        },
    )
    return transformers.Qwen3_5ForCausalLM(config).eval()


@pytest.mark.parametrize("share", [True, False])
def test_inference_loader_casts_weights_and_buffers_without_changing_constructor(
    model, monkeypatch, share
):
    from jsonllm import training
    from jsonllm.backends import cuda

    model.to(torch.bfloat16)
    model.register_buffer("precision_probe", torch.tensor([0.125], dtype=torch.bfloat16))
    assert SharedPredictor(model, Tokenizer()).inference_dtype == "bfloat16"

    def load(path, revision):
        assert (path, revision) == ("stock", "pinned")
        assert next(model.parameters()).dtype == torch.bfloat16
        return model

    monkeypatch.setattr(cuda, "load_model", load)
    monkeypatch.setattr(training, "load_tokenizer", lambda *args: Tokenizer())
    predictor = SharedPredictor.load("stock", "pinned", share=share)
    assert predictor.inference_dtype == "float16" and predictor.share == share
    assert all(p.dtype == torch.float16 for p in model.parameters())
    assert model.precision_probe.dtype == torch.float16
    assert model.precision_probe.item() == 0.125


def test_inference_loader_rejects_fp16_overflow(model, monkeypatch):
    from jsonllm.backends import cuda

    model.to(torch.bfloat16)
    with torch.no_grad():
        next(model.parameters()).flatten()[0] = 1e8
    monkeypatch.setattr(cuda, "load_model", lambda *args: model)
    with pytest.raises(ValueError, match="nonfinite weights"):
        SharedPredictor.load("stock", "pinned")


@pytest.mark.parametrize("length", [8, 527, 528, 529])
def test_ragged_choice_batch_matches_full_recompute_and_isolates_state(model, length):
    predictor = SharedPredictor(model, Tokenizer())
    session = predictor.open_record("x" * length, 4)
    suffixes = [[11, 12, 13], [21, 22, 23, 24], [31, 32], [41, 42, 43, 44, 45]]
    try:
        with torch.inference_mode():
            hidden, _ = session._batch(suffixes, selection=True)
            assert session.cache.get_seq_length() == length
            snapshot = [layer.recurrent_states[0].clone() for layer in session.cache.layers[:3]]
            for row, suffix in enumerate(suffixes):
                full = model.model(
                    input_ids=torch.tensor([session.prefix + suffix]), use_cache=False
                ).last_hidden_state[0, -1]
                torch.testing.assert_close(hidden[row], full, atol=2e-5, rtol=2e-5)
            session._batch([[51, 52]], selection=True)
            for original, layer in zip(snapshot, session.cache.layers[:3], strict=True):
                assert torch.equal(original, layer.recurrent_states[0])
        assert session.metrics["common_prefills"] == 1
        assert session.metrics["common_token_evaluations"] == length
        assert session.metrics["forwards"][1]["batch"] == 4
    finally:
        session.close()
    assert session.cache is None and not predictor.lock.locked()


def test_fork_continuation_and_candidate_projection(model):
    with torch.inference_mode():
        prefix = torch.tensor([[1, 2, 3, 4]])
        cached = model.model(input_ids=prefix, use_cache=True).past_key_values
        branch, nbytes = fork_cache(cached, 2, "cpu")
        suffix = torch.tensor([[5, 6], [7, 8]])
        output = model.model(input_ids=suffix, past_key_values=branch, use_cache=True)
        more = model.model(
            input_ids=torch.tensor([[9], [10]]),
            past_key_values=output.past_key_values,
            use_cache=True,
        )
        full = model.model(
            input_ids=torch.tensor([[1, 2, 3, 4, 5, 6, 9], [1, 2, 3, 4, 7, 8, 10]]), use_cache=False
        )
        torch.testing.assert_close(
            more.last_hidden_state[:, -1], full.last_hidden_state[:, -1], atol=2e-5, rtol=2e-5
        )
        hidden = more.last_hidden_state[:, -1]
        torch.testing.assert_close(
            candidate_logits(hidden, model.lm_head.weight, [3, 11, 25]),
            model.lm_head(hidden)[:, [3, 11, 25]],
            atol=1e-6,
            rtol=1e-5,
        )
        assert cached.get_seq_length() == 4 and nbytes > 0


def test_single_decision_has_no_extra_prefill(model):
    predictor = SharedPredictor(model, Tokenizer())
    session = predictor.open_record("abc", 1)
    try:
        with torch.inference_mode():
            session._batch([[1, 2]], selection=True)
        assert len(session.metrics["forwards"]) == 1
        assert session.metrics["cache_copy_bytes"] == 0
    finally:
        session.close()


def test_mixed_choice_counts_use_only_candidate_rows_and_release_on_cancel(model, monkeypatch):
    import jsonllm.backends.shared_cuda as backend

    choices = {"one": {"A": False, "B": True}, "two": {"A": "x", "B": "y", "C": "z"}}
    monkeypatch.setattr(
        backend,
        "field_example",
        lambda record, name, **kw: {
            "choices": choices[name],
            "field": name,
        },
    )
    monkeypatch.setattr(backend, "question_ids", lambda example, tokenizer: [11, 12])
    monkeypatch.setattr(
        model.lm_head, "forward", lambda *args: pytest.fail("Full vocab projection")
    )
    predictor = SharedPredictor(model, Tokenizer())
    session = predictor.open_record("abc", 2)
    try:
        result = session.predict({}, ["one", "two"], {}, 4)
        assert result["one"] in [False, True] and result["two"] in ["x", "y", "z"]
        assert [len(c["probabilities"]) for c in session.metrics["calls"]] == [2, 3]
        assert session.metrics["forwards"][-1]["batch"] == 2
        session.cancel()
        with torch.inference_mode(), pytest.raises(ValueError, match="cancelled"):
            session._batch([[1, 2]], selection=True)
    finally:
        session.close()
    assert session.cache is None and not predictor.lock.locked()


def test_split_diagnostics_trace_real_hybrid_model_and_restore_kernel(model):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module

    from jsonllm.backends.shared_diagnostics import split_prefill_diagnostics

    original = module.torch_chunk_gated_delta_rule
    hooks_before = sum(len(layer._forward_hooks) for layer in model.modules())
    prefix = [11, 12, 13, 14] * 17
    with torch.inference_mode():
        report = split_prefill_diagnostics(
            model,
            "cpu",
            prefix,
            [21, 22, 23],
            [],
            lambda a, b: {"max_delta": (a - b).abs().max().item()},
        )
    assert module.torch_chunk_gated_delta_rule is original
    assert sum(len(layer._forward_hooks) for layer in model.modules()) == hooks_before
    assert report["module_outputs"][0]["module"] == "embed_tokens"
    assert set(report["isolated_first_kernel"]) == {
        "native_chunk",
        "native_recurrent",
        "torch_fp32_chunk",
    }
    for probes in report["isolated_first_kernel"].values():
        assert set(probes) == {"64", "68"}
        for row in probes.values():
            assert row["output"]["max_absolute_delta"] < 1e-5
            assert row["state"]["max_absolute_delta"] < 1e-5
    for variant in report["model_kernel_variants"].values():
        assert variant["full_vs_unchanged_native_full"]["max_delta"] < 1e-5
        assert set(variant["split_vs_variant_full"]) == {"64", "68"}


def test_split_trace_restores_hooks_and_kernel_after_exception(model, monkeypatch):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module

    from jsonllm.backends.shared_diagnostics import prefill_trace

    def fail(*args, **kwargs):
        raise RuntimeError("kernel interrupted")

    monkeypatch.setattr(module, "torch_chunk_gated_delta_rule", fail)
    hooks_before = sum(len(layer._forward_hooks) for layer in model.modules())
    with torch.inference_mode(), pytest.raises(RuntimeError, match="kernel interrupted"):
        prefill_trace(model, "cpu", [11, 12], [21, 22], module)
    assert module.torch_chunk_gated_delta_rule is fail
    assert sum(len(layer._forward_hooks) for layer in model.modules()) == hooks_before


def test_precision_diagnostics_restore_exact_weights_and_buffers_on_exception():
    from jsonllm.backends.shared_diagnostics import model_precision

    # Include values that overflow/underflow FP16: a round-trip cast cannot restore them.
    model = torch.nn.Linear(3, 1, bias=False).to(torch.bfloat16)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1e8, 1e-12, 1.5]]))
    model.register_buffer("probe", torch.tensor([1e8, 1e-12], dtype=torch.bfloat16))
    weight, buffer = model.weight.detach().clone(), model.probe
    original_storage = model.weight.data_ptr()
    with pytest.raises(RuntimeError, match="diagnostic interrupted"):
        with model_precision(model, torch.float16):
            assert model.weight.dtype == torch.float16
            assert torch.isinf(model.weight).any()
            raise RuntimeError("diagnostic interrupted")
    assert model.weight.dtype == torch.bfloat16
    assert model.weight.data_ptr() == original_storage
    assert torch.equal(model.weight, weight)
    assert model.probe is buffer


def cache_snapshot(cache):
    """Include every hybrid state tensor, not just attention KV."""
    result = {}
    for i, layer in enumerate(cache.layers):
        for name in ("keys", "values", "conv_states", "recurrent_states"):
            value = getattr(layer, name, None)
            if isinstance(value, dict):
                for key, tensor in value.items():
                    if tensor is not None:
                        result[i, name, key] = tensor.clone()
            elif value is not None:
                result[i, name] = value.clone()
    return result


class GenerationTokenizer(Tokenizer):
    eos_token_id = 127

    def decode(self, tokens, **kwargs):
        return "".join({5: '"', 6: "a", 7: "b"}[t] for t in tokens)


class ScriptedMatcher:
    def __init__(self, tokens):
        self.tokens = tokens
        self.accepted = []

    def accept_token(self, token):
        assert token == self.tokens[len(self.accepted)]
        self.accepted.append(token)
        return True

    def is_terminated(self):
        return self.accepted[-1:] == [127]


class ScriptedGrammar:
    def __init__(self, schedules):
        self.schedules = schedules
        self.matchers = []

    def matcher(self, schema):
        matcher = ScriptedMatcher(self.schedules[len(self.matchers)])
        self.matchers.append(matcher)
        return matcher

    def allocate(self, count, device="cpu"):
        return None

    def mask(self, logits, matchers, bitmask, **kwargs):
        # Constrain the real model's logits to deterministic lengths, so every
        # termination pattern is exercised independently of random model weights.
        allowed = torch.zeros_like(logits, dtype=torch.bool)
        for row, matcher in enumerate(matchers):
            allowed[row, matcher.tokens[len(matcher.accepted)]] = True
        logits.masked_fill_(~allowed, -float("inf"))


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.cuda,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA"),
            ],
        ),
    ],
)
@pytest.mark.parametrize("lengths", [[3], [3, 3, 3], [3, 1, 3], [1, 3, 1, 3], [0, 0]])
def test_generation_compaction_matches_legacy_tokens_and_all_hybrid_states(
    model, monkeypatch, lengths, device
):
    model = model.to(device=device, dtype=torch.float16 if device == "cuda" else torch.float32)
    import jsonllm.backends.shared_cuda as backend

    monkeypatch.setattr(
        backend,
        "field_example",
        lambda record, name, **kw: {
            "choices": None,
            "schema": {"type": "string"},
            "prefill": "",
            "prompt_version": backend.SHARED_VERSION,
        },
    )
    monkeypatch.setattr(backend, "question_ids", lambda example, tokenizer: [11, 12])
    schedules = [[5] + [6 + i % 2] * n + [5, 127] for i, n in enumerate(lengths)]
    names = [f"field{i}" for i in range(len(lengths))]

    def run(legacy, profile=False):
        predictor = SharedPredictor(model, GenerationTokenizer(), profile_stages=profile)
        predictor.grammar = ScriptedGrammar(schedules)
        session = predictor.open_record("abc", len(lengths))
        snapshots, inputs, reorders, reorder_bytes = [], [], [], []
        original_forward = session._forward
        common_before = None

        def traced_forward(tokens, cache, **kwargs):
            nonlocal common_before
            if cache is not None:
                original_reorder = cache.reorder_cache

                def reorder(indices):
                    reorders.append(indices.tolist())
                    result = original_reorder(indices)
                    reorder_bytes.append(
                        sum(t.numel() * t.element_size() for t in cache_snapshot(cache).values())
                    )
                    return result

                # Re-wrap only once per cache object.
                if not getattr(cache, "_test_traced", False):
                    cache.reorder_cache = reorder
                    cache._test_traced = True
            inputs.append(tokens.clone())
            result = original_forward(tokens, cache, **kwargs)
            snapshots.append(cache_snapshot(result.past_key_values))
            if kwargs["kind"] == "common":
                common_before = cache_snapshot(result.past_key_values)
            return result

        session._forward = traced_forward
        if legacy:

            def compact(cache, active, tokens, keep):
                indices = torch.tensor(keep, device=tokens.device)
                cache.reorder_cache(indices)
                return [active[row] for row in keep], tokens.index_select(0, indices)

            session._compact_generation = compact
        try:
            values = session.predict({}, names, {}, 10)
            if common_before is not None:
                after = cache_snapshot(session.cache)
                assert after.keys() == common_before.keys()
                assert all(torch.equal(after[k], v) for k, v in common_before.items())
        finally:
            session.close()
        assert session.cache is None and not predictor.lock.locked()
        return (
            values,
            inputs,
            snapshots,
            reorders,
            predictor.grammar.matchers,
            session.metrics,
            reorder_bytes,
        )

    old, new = run(True), run(False)
    profiled = run(False, profile=True)
    assert profiled[0] == new[0]
    assert all(torch.equal(a, b) for a, b in zip(profiled[1], new[1], strict=True))
    for a, b in zip(profiled[2], new[2], strict=True):
        assert a.keys() == b.keys()
        assert all(torch.equal(a[k], b[k]) for k in a)
    assert profiled[5]["cache_reorder_bytes"] == sum(profiled[6])
    assert profiled[5]["cache_reorder_count"] == len(set(lengths)) - 1
    assert (profiled[5]["cache_reorder_bytes"] > 0) == (len(set(lengths)) > 1)
    assert {"lock_wait", "grammar_compile", "lm_head", "token_readback", "grammar_accept"} <= (
        profiled[5]["stage_host_seconds"].keys()
    )
    if device == "cuda":
        assert profiled[5]["gpu_seconds"]["lm_head"] > 0
    else:
        assert not profiled[5]["gpu_seconds"]
    assert "stage_host_seconds" not in new[5]
    assert (
        old[0]
        == new[0]
        == {name: ("a" if i % 2 == 0 else "b") * lengths[i] for i, name in enumerate(names)}
    )
    assert len(old[1]) == len(new[1])
    assert all(torch.equal(a, b) for a, b in zip(old[1], new[1], strict=True))
    for a, b in zip(old[2], new[2], strict=True):
        assert a.keys() == b.keys()
        assert all(torch.equal(a[k], b[k]) for k in a)
    assert [m.accepted for m in old[4]] == [m.accepted for m in new[4]] == schedules
    assert len(new[3]) == len(set(lengths)) - 1
    assert len(old[3]) > len(new[3])


def test_identity_compaction_allocates_nothing(model, monkeypatch):
    predictor = SharedPredictor(model, Tokenizer())
    session = predictor.open_record("abc", 1)
    active, tokens = [1, 2], torch.tensor([6, 7])
    monkeypatch.setattr(torch, "tensor", lambda *a, **kw: pytest.fail("Unexpected allocation"))
    try:
        kept, selected = session._compact_generation(None, active, tokens, [0, 1])
        assert kept is active and selected is tokens
    finally:
        session.close()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.cuda,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA"),
            ],
        ),
    ],
)
@pytest.mark.parametrize("failure", ["cancel", "grammar", "projection"])
def test_generation_failure_releases_profiled_session_cache_and_lock(
    model, monkeypatch, failure, device
):
    model = model.to(device=device, dtype=torch.float16 if device == "cuda" else torch.float32)
    import jsonllm.backends.shared_cuda as backend
    from jsonllm.ui import compile_ui, run_ui

    monkeypatch.setattr(
        backend,
        "field_example",
        lambda record, name, **kw: {
            "choices": None,
            "schema": {"type": "string"},
            "prefill": "",
            "prompt_version": backend.SHARED_VERSION,
        },
    )
    monkeypatch.setattr(backend, "question_ids", lambda *a: [11, 12])
    predictor = SharedPredictor(model, GenerationTokenizer(), profile_stages=True)
    predictor.grammar = ScriptedGrammar([[5, 6, 5, 127]] * 2)
    sessions = []
    original_open = predictor.open_record

    def fail(*a, **kw):
        raise ValueError("injected generation failure")

    def open_record(*args):
        session = original_open(*args)
        sessions.append(session)
        if failure == "cancel":
            session.cancel()
        return session

    monkeypatch.setattr(predictor, "open_record", open_record)
    if failure == "grammar":
        monkeypatch.setattr(predictor.grammar, "matcher", fail)
    if failure == "projection":
        monkeypatch.setattr(model.lm_head, "forward", fail)
    source = {
        "version": "genui-v1",
        "components": ["Card"],
        "decisions": {
            "a": {"type": "string", "instructions": "Write a title"},
            "b": {"type": "string", "instructions": "Write a subtitle"},
        },
        "view": {"component": "Card", "props": {"title": {"path": "decisions.a"}}},
    }
    result = run_ui(compile_ui(source), "text", {}, {}, {}, predictor=predictor)
    assert result["output"] is None
    assert result["diagnostics"]["exception"]["type"] == "ValueError"
    assert sessions[0].cache is None and sessions[0].closed and not predictor.lock.locked()
    assert result["diagnostics"]["model"]["profile_stages"]


def test_lock_wait_is_measured_separately(model, monkeypatch):
    import jsonllm.backends.stage_profile as profiling

    elapsed = 0.0

    class Lock:
        owned = False

        def acquire(self):
            nonlocal elapsed
            elapsed += 0.25
            self.owned = True

        def release(self):
            self.owned = False

    predictor = SharedPredictor(model, Tokenizer(), profile_stages=True)
    predictor.lock = Lock()
    monkeypatch.setattr(profiling.time, "perf_counter", lambda: elapsed)
    session = predictor.open_record("abc", 1)
    assert session.metrics["stage_host_seconds"]["lock_wait"] == 0.25
    assert session.metrics["tokenize_seconds"] == 0.0
    session.close()
    assert not predictor.lock.owned
