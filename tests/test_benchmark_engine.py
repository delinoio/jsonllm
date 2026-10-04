import time

import pytest
from jsonschema.exceptions import ValidationError

from jsonllm.benchmark_data import make_case
from jsonllm.benchmark_engine import (
    decision_values,
    field_inference,
    object_schema,
    prepare_schemas,
    render_output,
    validate_object,
)
from jsonllm.benchmark_measure import measure, summarize


class Session:
    def __init__(self, owner):
        self.owner = owner
        self.metrics = {"calls": []}

    def predict(self, record, names, answers, max_tokens):
        assert "answers" not in record and "source" not in record
        self.owner.waves.append((names[:], answers.copy()))
        if self.owner.fail:
            raise ValueError("bad inference")
        return {name: self.owner.gold[name] for name in names}

    def close(self):
        self.owner.closed = True


class Fake:
    def __init__(self, row, fail=False):
        self.gold = row["answers"]
        self.waves = []
        self.fail = fail
        self.closed = False

    def open_record(self, context, count):
        return Session(self)


def test_serial_and_parallel_waves_and_predicted_dependencies():
    row = make_case("dev", 1, kind="dependency", width=4, depth=2)
    parallel = Fake(row)
    result, _ = field_inference(row, parallel)
    assert result == row["answers"]
    assert parallel.waves[0] == (["f0", "f2"], {})
    assert parallel.waves[1][0] == ["f1", "f3"]
    assert parallel.waves[1][1] == {k: row["answers"][k] for k in ["f0", "f2"]}
    serial = Fake(row)
    field_inference(row, serial, serial=True)
    assert all(len(names) == 1 for names, _ in serial.waves)


def test_error_releases_session_and_counts_failure():
    row = make_case("dev", 0)
    fake = Fake(row, fail=True)
    result = measure(row, "shared_fields", fake)
    assert fake.closed and not result["correct"] and not result["schema_valid"]
    good = measure(row, "shared_fields", Fake(row))
    summary = summarize([result, good], 2)
    assert summary["accuracy"] == 0.5
    assert summary["errors"] == 1
    assert summary["correct_records_per_second"] == 0.5
    assert summary["latency_ms"]["p95"] > 0


def test_first_usable_field_includes_dispatch_queue_and_input_processing():
    row = make_case("dev", 0)
    scheduled = time.perf_counter() - 0.25
    result = measure(row, "shared_fields", Fake(row), scheduled=scheduled)
    assert 0.25 <= result["first_usable_seconds"] <= result["latency_seconds"]
    assert "first_usable_at" not in result["metrics"]


def test_expired_arrival_never_starts_gpu():
    row = make_case("dev", 0)
    fake = Fake(row)
    result = measure(row, "shared_fields", fake, scheduled=time.perf_counter() - 2, timeout=1)
    assert not fake.waves
    assert result["error"]["type"] == "TimeoutError"
    assert result["latency_seconds"] >= 2


def test_schema_rejects_extra_missing_and_wrong_value():
    row = make_case("dev", 0, width=1)
    assert object_schema(row)["additionalProperties"] is False
    for result in ({}, row["answers"] | {"extra": 1}, {"f0": "not-in-enum"}):
        with pytest.raises((ValueError, ValidationError)):
            validate_object(row, result)


def test_schema_preparation_deduplicates_and_does_not_receive_oracle():
    class Compiler:
        def __init__(self):
            self.records = []

        def prepare_schema(self, record):
            assert set(record) == {"id", "context", "questions"}
            self.records.append(record)

    row = make_case("dev", 0, width=1)
    backend = Compiler()
    result = prepare_schemas([row, row], "whole_json", backend)
    assert len(backend.records) == result["distinct_object_schemas"] == 1


def test_application_output_is_assembled_only_from_predictions():
    row = make_case("application", 0, width=2)
    row["output_mode"] = "ui_tree"
    predicted = dict(row["answers"])
    predicted["f0"] = next(v for v in row["questions"]["f0"]["enum"] if v != predicted["f0"])
    output = render_output(row, predicted)
    assert decision_values(row, output) == predicted
    assert output["children"][0]["value"] != row["answers"]["f0"]
    output["children"][0]["name"] = "unexpected"
    with pytest.raises(ValidationError):
        decision_values(row, output)


def test_field_and_whole_application_paths_score_the_same_decisions():
    row = make_case("application", 0, width=2)
    row.update(output_mode="ui_tree", json_max_tokens=128)

    class Whole:
        def infer(self, record, max_tokens, deadline):
            assert "answers" not in record
            return render_output(record, row["answers"]), {"output_tokens": 50}

    fields = measure(row, "shared_fields", Fake(row))
    whole = measure(row, "whole_json", Whole())
    assert fields["correct"] and whole["correct"]
    assert fields["output"] == whole["output"]
    assert fields["metrics"]["rendered_output"] == whole["metrics"]["rendered_output"]
