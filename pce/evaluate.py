from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from utils.io import read_jsonl


def ece_score(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        else:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])

        if mask.any():
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += abs(acc - conf) * mask.mean()

    return float(ece)


def prefix_wqd(y_true, y_prob) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_true - y_prob) ** 2))


def safe_roc_auc(y_true, y_prob) -> Optional[float]:
    y_true = list(y_true)
    if len(set(y_true)) <= 1:
        return None
    return float(roc_auc_score(y_true, y_prob))


def safe_pr_auc(y_true, y_prob) -> Optional[float]:
    y_true = list(y_true)
    if len(y_true) == 0:
        return None
    return float(average_precision_score(y_true, y_prob))


def compute_metrics(
    y_true: List[int],
    y_prob: List[float],
    n_bins: int = 10,
) -> Dict[str, Any]:
    y_prob_clip = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    y_true_np = np.asarray(y_true, dtype=int)

    metrics = {
        "n": int(len(y_true)),
        "positive_rate": float(y_true_np.mean()) if len(y_true_np) > 0 else None,
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "pr_auc": safe_pr_auc(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)) if len(y_true) > 0 else None,
        "log_loss": float(log_loss(y_true, y_prob_clip)) if len(y_true) > 0 else None,
        "ece": ece_score(y_true, y_prob, n_bins=n_bins) if len(y_true) > 0 else None,
        "prefix_wqd": prefix_wqd(y_true, y_prob) if len(y_true) > 0 else None,
    }
    return metrics


def read_prediction_rows(path: str) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    required = {"label_success", "success_prob"}
    for i, r in enumerate(rows):
        missing = required - set(r.keys())
        if missing:
            raise ValueError(f"Row {i} in {path} missing fields: {missing}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        required=True,
        help="jsonl 文件，至少包含字段 label_success 和 success_prob",
    )
    parser.add_argument(
        "--out",
        default="",
        help="可选，保存 metrics 的 json 文件路径",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=10,
        help="ECE 的 bin 数",
    )
    args = parser.parse_args()

    rows = read_prediction_rows(args.predictions)
    y_true = [int(r["label_success"]) for r in rows]
    y_prob = [float(r["success_prob"]) for r in rows]

    metrics = compute_metrics(y_true=y_true, y_prob=y_prob, n_bins=args.n_bins)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
