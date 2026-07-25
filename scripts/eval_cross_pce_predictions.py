#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


LABEL_NAMES = {
    "label",
    "target",
    "y",
    "gold_label",
    "success",
    "success_label",
    "is_correct",
    "trajectory_success",
    "final_ok",
}

SCORE_NAMES = {
    "score",
    "prob",
    "probability",
    "confidence",
    "p_success",
    "prob_success",
    "p_correct",
    "pred_prob",
    "prediction_probability",
    "pce_score",
    "y_prob",
}


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    result = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(value, name))
    else:
        result[prefix] = obj

    return result


def find_value(flat: dict[str, Any], allowed: set[str]):
    for key, value in flat.items():
        leaf = key.split(".")[-1].lower()
        if leaf in allowed and value is not None:
            return key, value
    return None, None


def to_label(value):
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "positive", "success"}:
            return 1
        if normalized in {"false", "no", "negative", "failure"}:
            return 0

    return int(float(value))


def to_score(value):
    if isinstance(value, list):
        if len(value) == 2:
            return float(value[1])
        if len(value) == 1:
            return float(value[0])

    return float(value)


def expected_calibration_error(y, p, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for index in range(n_bins):
        left = edges[index]
        right = edges[index + 1]

        if index == n_bins - 1:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p >= left) & (p < right)

        if not np.any(mask):
            continue

        accuracy = np.mean(y[mask])
        confidence = np.mean(p[mask])
        ece += np.mean(mask) * abs(accuracy - confidence)

    return float(ece)


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--bins", type=int, default=10)
args = parser.parse_args()

rows = []

with open(args.input, encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        if not line.strip():
            continue

        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"JSON error: {args.input}:{line_no}"
            ) from exc

if not rows:
    raise RuntimeError(f"No rows: {args.input}")

labels = []
scores = []
label_keys = set()
score_keys = set()

for index, row in enumerate(rows):
    flat = flatten(row)

    label_key, label_value = find_value(flat, LABEL_NAMES)
    score_key, score_value = find_value(flat, SCORE_NAMES)

    if label_value is None or score_value is None:
        print("First unresolved row index:", index)
        print("Available keys:", sorted(flat.keys()))
        raise KeyError(
            "Cannot identify label/score field. "
            "Add the actual field name to LABEL_NAMES or SCORE_NAMES."
        )

    labels.append(to_label(label_value))
    scores.append(to_score(score_value))
    label_keys.add(label_key)
    score_keys.add(score_key)

y = np.asarray(labels, dtype=np.int64)
p = np.asarray(scores, dtype=np.float64)
p = np.clip(p, 1e-7, 1 - 1e-7)

if len(np.unique(y)) < 2:
    roc_auc = None
    pr_auc = None
else:
    roc_auc = float(roc_auc_score(y, p))
    pr_auc = float(average_precision_score(y, p))

metrics = {
    "n": int(len(y)),
    "n_positive": int(y.sum()),
    "positive_rate": float(y.mean()),
    "label_fields": sorted(label_keys),
    "score_fields": sorted(score_keys),
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "brier": float(brier_score_loss(y, p)),
    "log_loss": float(log_loss(y, p, labels=[0, 1])),
    "ece": expected_calibration_error(y, p, args.bins),
    "n_bins": args.bins,
}

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

print(json.dumps(metrics, ensure_ascii=False, indent=2))
print("saved:", output)
