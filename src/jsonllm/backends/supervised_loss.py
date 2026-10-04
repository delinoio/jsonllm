"""Causal loss with vocabulary projection only at supervised prediction positions."""


def supervised_loss(model, inputs, *, num_items_in_batch=None):
    import torch.nn.functional as functional

    labels = inputs["labels"]
    if (labels[:, 0] != -100).any():
        raise ValueError("The first token cannot be a causal prediction target")
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    output = base.model(
        **{k: v for k, v in inputs.items() if k not in {"labels", "num_items_in_batch"}},
        use_cache=False,
    )
    targets = labels[:, 1:]
    selected = targets != -100
    hidden = output.last_hidden_state[:, :-1][selected]
    if not hidden.shape[0]:
        raise ValueError("No supervised tokens in this batch")
    logits = base.lm_head(hidden).float()
    denominator = num_items_in_batch if num_items_in_batch is not None else hidden.shape[0]
    loss = functional.cross_entropy(logits, targets[selected], reduction="sum") / denominator
    return loss, {"loss": loss, "logits": logits}
