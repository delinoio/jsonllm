import copy

import pytest

from jsonllm.benchmark_data import (
    assemble_topology,
    inference_record,
    make_case,
    reference,
    suite,
    topology_case,
)


def test_oracles_use_facts_not_answers():
    for kind in ("choice", "number", "string", "dependency"):
        row = make_case("oracle", 4, kind=kind, depth=4)
        expected = row.pop("answers")
        assert reference(row) == expected
        payload = inference_record(row)
        assert set(payload) == {"id", "context", "questions"}
        assert "source" not in payload


def test_choice_and_dependency_reference():
    row = make_case("oracle", 0, kind="choice", width=1, choices=2)
    row["source"] = {
        "entries": [{"key": "x", "score": 9}, {"key": "y", "score": 3}],
        "queries": [{"keys": ["y", "x"]}],
    }
    assert reference(row) == {"f0": "x"}
    row["kind"] = "dependency"
    row["source"] = {
        "entries": [{"key": "x", "next": "y"}, {"key": "y", "next": "x"}],
        "queries": [{"key": "x", "parent": None}, {"parent": 0}],
    }
    assert reference(row) == {"f0": "y", "f1": "x"}


def test_fresh_reproducible_splits_and_balance():
    data = suite()
    assert data == suite()
    assert len(data["dev"]) == 64 and len(data["core"]) == 256
    rows = [r for group in data.values() for r in group]
    assert len({r["id"] for r in rows}) == len(rows)
    for group in data.values():
        assert sum(r["language"] == "en" for r in group) == len(group) // 2
    assert not {r["context"] for r in data["dev"]} & {r["context"] for r in data["core"]}


def test_scaling_axes_do_not_change_source_pool_and_field_count_together():
    data = suite()
    for level in (1, 4, 8, 16):
        row = data[f"scale-width-{level}"][0]
        assert len(row["questions"]) == level
        assert len(row["source"]["entries"]) == 16
        assert row["condition"]["context_tokens"] == 512
    for level in (2, 4, 8, 16):
        row = data[f"scale-choices-{level}"][0]
        assert len(row["questions"]) == 4
        assert len(row["source"]["entries"]) == 16
    for level in (8, 32, 96):
        row = data[f"scale-text_words-{level}"][0]
        assert len(row["questions"]) == 1
        assert row["condition"]["context_tokens"] == 512


@pytest.mark.parametrize("kind", ["tree", "workflow"])
def test_topology_oracle_and_invalid_reference(kind):
    row = topology_case(kind, 3)
    gold = reference(row)
    assert assemble_topology(gold, kind)["nodes"]
    bad = copy.deepcopy(gold)
    bad["parent_0"] = 7
    with pytest.raises(ValueError):
        assemble_topology(bad, kind)
