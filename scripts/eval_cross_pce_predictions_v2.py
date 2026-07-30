#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def ece_score(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int,
) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    result = 0.0

    for i in range(n_bins):
        left = edges[i]
        right = edges[i + 1]

        if i == n_bins - 1:
            mask = (
                (probabilities >= left)
                & (probabilities <= right)
            )
        else:
            mask = (
                (probabilities >= left)
                & (probabilities < right)
            )

        if not np.any(mask):
            continue

        bin_acc = float(labels[mask].mean())
        bin_conf = float(probabilities[mask].mean())
        bin_weight = float(mask.mean())

        result += bin_weight * abs(bin_acc - bin_conf)

    return result


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--bins", type=int, default=10)
args = parser.parse_args()

labels = []
scores = []

with open(args.input, encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        if not line.strip():
            continue

        row = json.loads(line)

        if "label_success" not in row:
            raise KeyError(
                f"{args.input}:{line_no} missing label_success; "
                f"keys={sorted(row.keys())}"
            )

        if "success_prob" not in row:
            raise KeyError(
                f"{args.input}:{line_no} missing success_prob; "
                f"keys={sorted(row.keys())}"
            )

        labels.append(int(row["label_success"]))
        scores.append(float(row["success_prob"]))

if not labels:
    raise RuntimeError(f"No rows found in {args.input}")

y = np.asarray(labels, dtype=np.int64)
p = np.asarray(scores, dtype=np.float64)
p = np.clip(p, 1e-7, 1.0 - 1e-7)

metrics = {
    "input": args.input,
    "n": int(len(y)),
    "n_positive": int(y.sum()),
    "positive_rate": float(y.mean()),
    "roc_auc": (
        float(roc_auc_score(y, p))
        if len(np.unique(y)) == 2
        else None
    ),
    "pr_auc": (
        float(average_precision_score(y, p))
        if len(np.unique(y)) == 2
        else None
    ),
    "brier": float(brier_score_loss(y, p)),
    "log_loss": float(log_loss(y, p, labels=[0, 1])),
    "ece": ece_score(y, p, args.bins),
    "n_bins": args.bins,
    "label_field": "label_success",
    "score_field": "success_prob",
}

out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

print(json.dumps(metrics, ensure_ascii=False, indent=2))
print("saved:", out)
