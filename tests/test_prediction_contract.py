from jsonllm.evaluation import predict
from jsonllm.prompts import field_example


class FixedPredictor:
    def __init__(self, tokenizer, output=" 5}"):
        self.tokenizer = tokenizer
        self.output = output
        self.prompts = []

    def scores(self, ids, candidates):
        self.prompts.append(self.tokenizer.decode(ids))
        return list(range(len(candidates), 0, -1))

    def generate(self, ids, max_tokens):
        self.prompts.append(self.tokenizer.decode(ids))
        return self.output


def test_invalid_numeric_is_not_repaired(record, tokenizer):
    predictor = FixedPredictor(tokenizer, ' "5"}')
    response = predict(predictor, field_example(record, "weight"), max_length=4096, max_tokens=64)
    assert not response["valid"] and response["error"] == "invalid_output"
