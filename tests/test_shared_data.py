from collections import Counter

import pytest

from jsonllm.shared_data import (
    FAMILIES,
    blueprint,
    generation_request,
    records,
    semantic_cases,
    training_record,
    validate_generated,
)
from jsonllm.ui import compile_ui, run_ui


def fixture_messages(bp):
    return {
        "cases": [
            {
                "step": c["step"],
                "text": f"Request {c['step']}: "
                + bp["entities"].get(c["entity"], {}).get("name", "none"),
                "summary": (
                    bp["entities"][c["entity"]]["name"]
                    + ": "
                    + bp["entities"][c["entity"]]["status"]
                )
                if c["kind"] == "generation"
                else "",
            }
            for c in semantic_cases(bp)
        ]
    }


def test_allocation_is_disjoint_and_has_exact_mix():
    counts = Counter()
    sizes = set()
    for group in range(1000):
        bp = blueprint(group)
        sizes.add(len(bp["registry"]))
        counts.update((bp["split"], c["kind"]) for c in bp["cases"])
    assert sizes == {4, 5, 6, 7, 8}
    for split, n in (("train", 800), ("validation", 100), ("test", 100)):
        assert counts[split, "code"] == n * 2
        assert counts[split, "semantic"] == n * 6
        assert counts[split, "generation"] == n * 2
    assert sum(map(len, FAMILIES.values())) == len(set(sum(FAMILIES.values(), [])))


@pytest.mark.parametrize("group", [0, 1, 2, 3, 800, 801, 900, 901])
def test_rules_match_independent_reference_through_entire_trajectory(group):
    bp = blueprint(group)
    rows = records(bp, fixture_messages(bp))

    class Predictor:
        closed = True
        metrics = {}

        def open_record(self, context, count):
            assert self.closed
            self.closed = False
            assert "UI inputs (JSON)" in context and count in (2, 3, 4)
            return self

        def predict(self, record, names, answers, max_tokens):
            return {name: item["answers"][name] for name in names}

        def close(self):
            self.closed = True

    predictor = Predictor()
    state = rows[0]["state"]
    for item in rows:
        previous = state.copy()
        result = run_ui(
            compile_ui(item["spec"]),
            item["context"],
            state,
            item["event"],
            item["facts"],
            predictor=predictor,
        )
        assert result["output"] == item["expected"], result["diagnostics"]
        assert predictor.closed
        state = result["output"]["state"]
        if item["step"] == 3:
            assert state == previous  # Stale event preserves the earlier predicted state.
        if item["kind"] == "code":
            assert result["diagnostics"]["model_fields"] == []
        else:
            prepared = training_record(item)
            assert prepared["answers"] == item["answers"]
            assert all("when" not in f for f in prepared["questions"].values())
            if item["answers"]["entity_choice"] is not None:
                assert "entity_choice" not in prepared["model_fields"]


def test_reject_bad_copy_reference_and_unfaithful_sentence():
    bp = blueprint(0)
    generated = fixture_messages(bp)
    generated["cases"][-1]["summary"] = "Invented summary."
    with pytest.raises(ValueError, match="Ungrounded"):
        validate_generated(bp, generated)
    generated = fixture_messages(bp)
    generated["cases"][0]["text"] += next(iter(bp["registry"]))
    with pytest.raises(ValueError, match="leaked"):
        validate_generated(bp, generated)


def test_unregistered_translation_requests_separate_prose_from_stored_entry_edits():
    bp = blueprint(630)
    assert bp["cases"][2]["component"] is None
    messages, _ = generation_request(bp, [2])
    prompt = messages[-1]["content"]
    assert "distinct short quotation or message to translate" in prompt
    assert "leave the stored entry unchanged" in prompt
    # Existing registered edits retain their intended semantics.
    messages, _ = generation_request(bp, [9])
    assert "leave the stored entry unchanged" not in messages[-1]["content"]


def test_seed_changes_blueprint_without_changing_split_contract():
    from jsonllm.shared_data import allocation

    old, new = blueprint(10), blueprint(10, 20261004)
    assert old == blueprint(10, 20260929)
    assert old != new
    assert (old["split"], old["family"], old["language"]) == (
        new["split"],
        new["family"],
        new["language"],
    )
    assert allocation()["seed"] == 20260929
    assert allocation(20261004, "deepseek", 50)["teacher"] == "deepseek"
