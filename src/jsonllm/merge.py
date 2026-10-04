"""Export a standalone text model, leaving the original PEFT artifact untouched."""

import gc
import json
from pathlib import Path

from .artifacts import artifact_manifest
from .config import load_config
from .io import write_json


def merge_adapter(adapter, output):
    import torch
    from peft import PeftModel

    from .backends.cuda import load_model
    from .training import load_tokenizer

    source, target = Path(adapter), Path(output)
    if target.exists() and any(target.iterdir()):
        raise ValueError("Merged output must be empty")
    run = json.loads((source / "run.json").read_text())
    if run.get("status") != "complete":
        raise ValueError("Only a completed adapter may be merged")
    config = load_config(source / "config.yaml")
    model = PeftModel.from_pretrained(load_model(config.model, config.revision), source).eval()
    tokenizer = load_tokenizer(config.model, config.revision)
    inputs = tokenizer("A package contains 31 items.", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        before = model(**inputs, logits_to_keep=1).logits.float().cpu()
        model = model.merge_and_unload(safe_merge=True).eval()
        after = model(**inputs, logits_to_keep=1).logits.float().cpu()
    if not torch.isfinite(after).all():
        raise ValueError("Non-finite merged logits")
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    loaded = load_model(str(target), config.revision).eval()
    with torch.inference_mode():
        reloaded = loaded(**inputs, logits_to_keep=1).logits.float().cpu()
    if not torch.equal(after, reloaded):
        raise ValueError("Merged checkpoint reload changed logits")
    report = {
        "source": str(source),
        "base": run["model"],
        "prompt_version": config.prompt_version,
        "source_artifacts": artifact_manifest(source),
        "max_logit_delta": (before - after).abs().max().item(),
        "top_token_match": before.argmax(-1).equal(after.argmax(-1)),
        "reload_exact": True,
        "note": "BF16 merge rounding can change close decisions; full evaluation is required.",
    }
    write_json(target / "merge.json", report)
    return report
