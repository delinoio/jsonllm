import time

import pytest
from jsonschema.exceptions import ValidationError

from jsonllm.benchmark_data import make_case
from jsonllm.benchmark_engine import field_inference, object_schema, validate_object
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
