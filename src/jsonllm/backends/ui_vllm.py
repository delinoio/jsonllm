"""Existing vLLM field-batch engine behind the same registered-rule UI boundary."""

from ..fast_inference import decode_result
from ..prompts import field_example, prompt_ids


class UIFieldPredictor:
    def __init__(self, predictor, prompt_version):
        self.predictor, self.prompt_version = predictor, prompt_version

    def open_record(self, context, count):
        self.predictor.begin_record()
        return FieldSession(self)


class FieldSession:
    def __init__(self, owner):
        self.owner = owner
        self.metrics = {"calls": [], "common_prefills": 0, "context_tokenizations": 0}

    def predict(self, record, names, answers, max_tokens):
        predictor = self.owner.predictor
        examples = [
            field_example(
                record, name, dependency_answers=answers, prompt_version=self.owner.prompt_version
            )
            for name in names
        ]
        inputs = [prompt_ids(e, predictor.tokenizer) for e in examples]
        self.metrics["context_tokenizations"] += len(names)
        self.metrics["common_prefills"] += len(names)
        outputs = predictor.infer_many(examples, inputs, max_tokens)
        values = {}
        for name, example, output in zip(names, examples, outputs, strict=True):
            self.metrics["calls"].append({"field": name, **output["metrics"]})
            result = decode_result(example, output["text"])
            if not result["valid"]:
                raise ValueError("invalid_output:" + name)
            values[name] = result["value"]
        return values

    def close(self):
        pass
