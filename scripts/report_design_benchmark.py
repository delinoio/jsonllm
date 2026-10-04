"""Build a shareable benchmark archive and plots from immutable local trials."""

import argparse
import gzip
import json
import shutil
from pathlib import Path

from jsonllm.artifacts import artifact_manifest, file_hash, verify_artifacts
from jsonllm.benchmark_report import aggregate_trials
from jsonllm.io import dumps, read_jsonl, write_json


def provenance(run, output):
    """Export an explicit evidence allowlist; keep private operation logs local."""
    evidence = output / "evidence"
    evidence.mkdir()
    for name in (
        "base-gate.json",
        "jsonllm-gate.json",
        "kernels.json",
        "input-audit.json",
        "costs.json",
        "recovery-verification.json",
        "validation.json",
        "schema-preflight.json",
        "tokenizer-preflight.json",
        "development-repair.json",
    ):
        if (run / name).exists():
            shutil.copy2(run / name, evidence / name)
    for name in ("environment-cuda.txt", "environment-vllm.txt"):
        if (run / name).exists():
            # Installed package versions suffice; editable paths and direct URLs do not.
            versions = [
                line
                for line in (run / name).read_text().splitlines()
                if "==" in line and " @ " not in line and not line.startswith("-e")
            ]
            (evidence / name).write_text("\n".join(versions) + "\n")
    hardware = run / "gpu-environment.txt"
    if hardware.exists():
        fields = {}
        for line in hardware.read_text().splitlines():
            name, separator, value = line.strip().partition(":")
            name = name.strip()
            if separator and name in {"Driver Version", "CUDA Version", "Product Name"}:
                fields.setdefault(name, value.strip())
        write_json(
            evidence / "hardware.json",
            {
                "fields": fields,
                "private_inventory_sha256": file_hash(hardware),
            },
        )
    invalidated = []
    for path in sorted((run / "invalidated").glob("*/reason.json")):
        invalidated.append(
            {
                "name": path.parent.name,
                "reason": json.loads(path.read_text()),
                "retained_local_files": {
                    p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size}
                    for p in sorted(path.parent.iterdir())
                    if p.is_file()
                },
            }
        )
    write_json(evidence / "invalidated.json", invalidated)
    startup = []
    for path in sorted(run.glob("*-server.json")):
        record = json.loads(path.read_text())
        command = list(record["command"])
        command[0] = "VLLM_PYTHON"
        position = command.index("--model") + 1
        command[position] = "models/" + Path(command[position]).name
        startup.append(
            {"name": path.stem, "startup_seconds": record["startup_seconds"], "command": command}
        )
    write_json(evidence / "server-startup.json", startup)


def format_span(value, digits=3):
    if value is None:
        return "not measured"
    return f"{value['median']:.{digits}f} [{value['min']:.{digits}f}, {value['max']:.{digits}f}]"


def markdown(report):
    lines = [
        "# Design benchmark measurements",
        "",
        "Values are trial medians [minimum, maximum]. "
        "Repeated executions are not independent quality samples. p99 is exploratory.",
        "",
        "| Stage / condition | Model | Method | Requests / arrival | Trials | Exact accuracy | "
        "p50 ms | p95 ms | Completed/s | Correct/s |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for g in report["groups"]:
        if g["stage"] == "development":
            continue
        m = g["trial_metrics"]
        lines.append(
            f"| {g['stage']} / {g['group']} | {g['model']} | {g['method']} | "
            f"{g['concurrency']} / {g['arrival_rate'] or 'closed'} | {g['repeats']} | "
            f"{g['quality']['execution_accuracy']:.2%} | "
            f"{format_span(g['latency_ms']['p50'], 1)} | "
            f"{format_span(g['latency_ms']['p95'], 1)} | "
            f"{format_span(m['completed_records_per_second'])} | "
            f"{format_span(m['correct_records_per_second'])} |"
        )
    lines += [
        "",
        "## Interpretation boundaries",
        "",
        "Each paired comparison uses the same model revision. The separate model rows must not "
        "be treated as a causal effect of runtime design. Exact full-record accuracy requires "
        "every model decision to match the independent oracle. No teacher API is used.",
        "",
        "A speed claim requires schema validity and a quality gate. `report.json` includes "
        "paired record-cluster bootstrap intervals and a conservative Wilson bound that stays "
        "nonzero even for perfect observed scores. Finite synthetic samples cannot establish "
        "general GenUI quality. See the protocol for queueing, first usable field, token-budget, "
        "and memory definitions.",
        "",
        "## Type-specific latency",
        "",
        "| Condition | Model | Method | Type | Unique records | Accuracy | "
        "p50 ms | p95 ms | p99 ms |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for g in report["groups"]:
        if g["stage"] != "core":
            continue
        for kind, k in g["by_kind"].items():
            lines.append(
                f"| {g['group']} | {g['model']} | {g['method']} | {kind} | "
                f"{k['quality']['unique_records']} | {k['quality']['execution_accuracy']:.2%} | "
                + " | ".join(format_span(k["latency_ms"][p], 1) for p in ("p50", "p95", "p99"))
                + " |"
            )
    return "\n".join(lines) + "\n"


def plots(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    core = [g for g in report["groups"] if g["stage"] == "core"]
    if not core:
        return
    colors = {"base": "#64748b", "jsonllm": "#0d9488"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for model, offset in [("base", -0.18), ("jsonllm", 0.18)]:
        items = [g for g in core if g["model"] == model]
        for ax, metric, title in [
            (axes[0], "latency", "End-to-end p95 latency (ms; lower is better)"),
            (
                axes[1],
                "correct_records_per_second",
                "Correct complete outputs/s (higher is better)",
            ),
        ]:
            values = [
                g["latency_ms"]["p95"] if metric == "latency" else g["trial_metrics"][metric]
                for g in items
            ]
            xs = [i + offset for i in range(len(items))]
            ax.bar(
                xs,
                [v["median"] for v in values],
                width=0.34,
                label=model,
                color=colors[model],
                yerr=[
                    [v["median"] - v["min"] for v in values],
                    [v["max"] - v["median"] for v in values],
                ],
                capsize=3,
            )
            ax.set_xticks(
                range(len(items)), [g["method"].replace("_", "\n") for g in items], fontsize=9
            )
            ax.set_title(title, fontsize=11)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
    axes[0].legend()
    fig.suptitle(
        "Same-model execution comparisons · trial median and range\n"
        "Synthetic exact-oracle tasks; consult quality gates before claiming a speedup",
        fontsize=12,
    )
    fig.savefig(output / "core-comparison.png", dpi=180)
    fig.savefig(output / "core-comparison.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    trials, incomplete = [], []
    raw_dir = args.output / "raw"
    raw_dir.mkdir()
    for path in sorted((args.run / "trials").iterdir()):
        if not path.is_dir():
            continue
        if not all((path / name).exists() for name in ("job.json", "summary.json", "raw.jsonl")):
            incomplete.append(path.name)
            continue
        job = json.loads((path / "job.json").read_text())
        summary = json.loads((path / "summary.json").read_text())
        summary["model"] = job["model"]  # Do not publish machine-local model paths.
        summary.pop("pid", None)
        if str(summary.get("device_after", "")).startswith("GPU-"):
            summary["device_after"] = summary["device_after"].partition(",")[2].strip()
        rows = read_jsonl(path / "raw.jsonl")
        trials.append({"job": job, "summary": summary, "rows": rows})
        with (raw_dir / (path.name + ".jsonl.gz")).open("wb") as stream:
            with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as archive:
                archive.write(("\n".join(dumps(r) for r in rows) + "\n").encode())
        write_json(raw_dir / (path.name + ".summary.json"), {"job": job, "summary": summary})
        if (path / "warmup.jsonl").exists():
            shutil.copy2(path / "warmup.jsonl", raw_dir / (path.name + ".warmup.jsonl"))
    result = aggregate_trials(trials)
    result["incomplete_trials"] = incomplete
    for name in (
        "omitted.json",
        "seal.json",
        "schedule.json",
        "admission.json",
        "base-model-integrity.json",
        "jsonllm-model-integrity.json",
    ):
        if (args.run / name).exists():
            shutil.copy2(args.run / name, args.output / name)
    shutil.copytree(args.data, args.output / "data")
    provenance(args.run, args.output)
    write_json(args.output / "report.json", result)
    (args.output / "README.md").write_text(markdown(result))
    if not args.no_plots:
        plots(result, args.output)
    manifest = artifact_manifest(args.output)
    write_json(args.output / "artifacts.json", manifest)
    verify_artifacts(args.output, manifest)
    print(dumps({"trials": len(trials), "groups": len(result["groups"]), "incomplete": incomplete}))


if __name__ == "__main__":
    main()
