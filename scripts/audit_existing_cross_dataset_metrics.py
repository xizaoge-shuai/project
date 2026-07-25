#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATASETS = ("gsm8k", "svamp", "asdiv", "math500", "mathqa")
ROOTS = [
    Path("outputs/metrics"),
    Path("outputs/final_selected_results"),
]

OUT = Path(
    "outputs/logs/cross_dataset_tables/"
    "existing_metrics_inventory.md"
)


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    result = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(value, name))
    else:
        result[prefix] = obj

    return result


def find_dataset(path: Path, data: dict[str, Any]) -> str | None:
    text = f"{path} {data.get('dataset', '')}".lower()

    for dataset in DATASETS:
        if dataset in text:
            return dataset

    return None


def pick(flat: dict[str, Any], names: tuple[str, ...]):
    for key, value in flat.items():
        leaf = key.split(".")[-1].lower()
        if leaf in names and value is not None:
            return value
    return None


pce_rows = []
pipeline_rows = []

for root in ROOTS:
    if not root.exists():
        continue

    for path in root.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        dataset = find_dataset(path, data)
        if dataset is None:
            continue

        flat = flatten(data)

        roc = pick(flat, ("roc_auc", "auc", "test_roc_auc"))
        pr = pick(
            flat,
            ("pr_auc", "average_precision", "test_pr_auc"),
        )
        brier = pick(
            flat,
            ("brier", "brier_score", "test_brier"),
        )
        ece = pick(flat, ("ece", "test_ece"))

        if any(value is not None for value in (roc, pr, brier, ece)):
            pce_rows.append(
                (dataset, path, roc, pr, brier, ece)
            )

        base = pick(
            flat,
            ("base_acc", "majority_acc", "before_acc"),
        )
        final = pick(
            flat,
            (
                "final_acc",
                "estimated_global_acc",
                "after_acc",
                "accuracy",
            ),
        )
        fixed = pick(flat, ("fixed", "n_fixed"))
        broken = pick(flat, ("broken", "n_broken"))
        net = pick(flat, ("net", "net_gain"))
        changed = pick(flat, ("changed", "n_changed"))
        extra = pick(
            flat,
            (
                "extra_per_sample",
                "extra_calls_per_sample",
                "extra_q",
            ),
        )

        if any(
            value is not None
            for value in (
                base,
                final,
                fixed,
                broken,
                net,
                changed,
                extra,
            )
        ):
            pipeline_rows.append(
                (
                    dataset,
                    path,
                    base,
                    final,
                    extra,
                    fixed,
                    broken,
                    net,
                    changed,
                )
            )


def fmt(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


lines = [
    "# Existing Cross-Dataset Metrics Inventory",
    "",
    "## Prefix-level PCE metric candidates",
    "",
    "| Dataset | File | ROC-AUC | PR-AUC | Brier | ECE |",
    "|---|---|---:|---:|---:|---:|",
]

for dataset, path, roc, pr, brier, ece in sorted(
    pce_rows,
    key=lambda row: (row[0], str(row[1])),
):
    lines.append(
        f"| {dataset} | `{path}` | {fmt(roc)} | {fmt(pr)} "
        f"| {fmt(brier)} | {fmt(ece)} |"
    )

lines += [
    "",
    "## Sample-level metric candidates",
    "",
    "| Dataset | File | Base | Final | Extra/Q | Fixed | Broken | Net | Changed |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]

for row in sorted(
    pipeline_rows,
    key=lambda item: (item[0], str(item[1])),
):
    (
        dataset,
        path,
        base,
        final,
        extra,
        fixed,
        broken,
        net,
        changed,
    ) = row

    lines.append(
        f"| {dataset} | `{path}` | {fmt(base)} | {fmt(final)} "
        f"| {fmt(extra)} | {fmt(fixed)} | {fmt(broken)} "
        f"| {fmt(net)} | {fmt(changed)} |"
    )

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("saved:", OUT)
print("PCE candidates:", len(pce_rows))
print("Pipeline candidates:", len(pipeline_rows))
