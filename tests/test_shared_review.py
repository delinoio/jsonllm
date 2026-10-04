import json

from jsonllm.io import write_jsonl
from jsonllm.shared_review import factual_judgment, review_trials


class Teacher:
    calls = 0

    def complete(self, messages, schema, *, request_tag):
        self.calls += 1
        payload = json.loads(messages[1]["content"])
        assert set(payload) == {"facts", "language", "sentence"}
        assert "expected" not in messages[1]["content"]
        return {"faithful": True, "reason": "Matches stored facts"}, {}


def review():
    return {
        "entries": {"ref": {"name": "A", "status": "ready"}},
        "intended_entry": "ref",
        "language": "en",
        "sentence": "A is ready",
    }


def test_factual_grounding_and_deduplication_preserve_failures(tmp_path):
    teacher = Teacher()
    assert not factual_judgment(review() | {"sentence": "Done"}, teacher)["faithful"]
    assert teacher.calls == 0
    row = {
        "group": 0,
        "kind": "generation",
        "faithful": None,
        "factual_review": review(),
        "structure_correct": True,
        "component_correct": True,
        "copy_correct": True,
        "state_correct": True,
        "schema_valid": True,
        "seconds": 0.1,
    }
    for i in range(2):
        write_jsonl(
            tmp_path / str(i) / "raw.jsonl",
            [
                row,
                row
                | {
                    "kind": "semantic",
                    "schema_valid": False,
                    "faithful": False,
                    "structure_correct": False,
                    "factual_review": None,
                },
            ],
        )
    result = review_trials(
        [tmp_path / "0", tmp_path / "1"], tmp_path / "reviewed", teacher, regression_passed=True
    )
    assert teacher.calls == 1
    assert result["model_accuracy"] == 0.5 and result["records"] == 4
    assert result["judgments_complete"] and result["failed_records"] == 2
