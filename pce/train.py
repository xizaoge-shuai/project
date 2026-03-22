from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from pce.dataset import (
    build_text_label_meta,
    load_pce_training_rows,
)
from pce.evaluate import compute_metrics
from pce.models.verifier import VerifierPCE
from utils.io import ensure_dir, read_yaml


def parse_csv_splits(x: str) -> List[str]:
    x = (x or "").strip()
    if not x:
        return []
    return [p.strip() for p in x.split(",") if p.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stratify_or_none(labels: List[int]):
    return labels if len(set(labels)) > 1 else None


def predict_success_probs(model: Any, texts: List[str]) -> List[float]:
    """
    尽量兼容不同 VerifierPCE 接口：
    1. predict_proba_texts(texts) -> List[float]
    2. predict_proba(texts) -> List[float] 或 ndarray[:, 2]
    3. predict("", text)["success_prob"]
    4. predict(text)["success_prob"]
    """
    if hasattr(model, "predict_proba_texts"):
        probs = model.predict_proba_texts(texts)
        return [float(x) for x in probs]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(texts)
        probs = np.asarray(probs)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return [float(x) for x in probs[:, 1]]
        return [float(x) for x in probs.reshape(-1)]

    results: List[float] = []
    for txt in texts:
        try:
            out = model.predict("", txt)
        except TypeError:
            out = model.predict(txt)

        if isinstance(out, dict) and "success_prob" in out:
            results.append(float(out["success_prob"]))
        else:
            results.append(float(out))
    return results


def build_prediction_rows(
    metas: List[Dict[str, Any]],
    labels: List[int],
    probs: List[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for meta, y, p in zip(metas, labels, probs):
        rows.append(
            {
                **meta,
                "label_success": int(y),
                "success_prob": float(p),
            }
        )
    return rows


def maybe_split_train_val_test(
    texts: List[str],
    labels: List[int],
    metas: List[Dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, Tuple[List[str], List[int], List[Dict[str, Any]]]]:
    """
    在没有现成 validation/test split 时，从输入样本随机切。
    """
    if not (
        0.0 <= val_ratio < 1.0
        and 0.0 <= test_ratio < 1.0
        and val_ratio + test_ratio < 1.0
    ):
        raise ValueError("val_ratio/test_ratio invalid")

    idx = list(range(len(texts)))
    y = labels

    # 先切 test
    if test_ratio > 0:
        train_idx, test_idx = train_test_split(
            idx,
            test_size=test_ratio,
            random_state=seed,
            stratify=stratify_or_none(y),
        )
    else:
        train_idx, test_idx = idx, []

    # 再从 train 部分切 val
    if val_ratio > 0 and len(train_idx) > 1:
        train_labels = [labels[i] for i in train_idx]
        relative_val_ratio = val_ratio / (1.0 - test_ratio)
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=relative_val_ratio,
            random_state=seed,
            stratify=stratify_or_none(train_labels),
        )
    else:
        val_idx = []

    def pack(idxs: List[int]):
        return (
            [texts[i] for i in idxs],
            [labels[i] for i in idxs],
            [metas[i] for i in idxs],
        )

    return {
        "train": pack(train_idx),
        "val": pack(val_idx),
        "test": pack(test_idx),
    }


def build_pce_from_config(cfg: Dict[str, Any]) -> VerifierPCE:
    """
    从 yaml 配置完整构造 VerifierPCE。
    """
    return VerifierPCE(
        max_features=int(cfg.get("max_features", 4000)),
        ngram_range=tuple(cfg.get("ngram_range", [1, 2])),
        min_df=int(cfg.get("min_df", 1)),
        max_df=float(cfg.get("max_df", 1.0)),
        lowercase=bool(cfg.get("lowercase", True)),
        sublinear_tf=bool(cfg.get("sublinear_tf", True)),
        C=float(cfg.get("C", 1.0)),
        max_iter=int(cfg.get("max_iter", 1000)),
        class_weight=cfg.get("class_weight", None),
        solver=str(cfg.get("solver", "liblinear")),
        random_state=int(cfg.get("random_state", 42)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model/pce_mlp.yaml")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["gsm8k", "strategyqa", "hotpotqa"],
    )
    parser.add_argument(
        "--level",
        default="step",
        choices=["path", "step", "atom", "path_level", "step_level", "atom_level"],
    )
    parser.add_argument("--label_base_dir", default="data/processed/labels")
    parser.add_argument("--train_splits", default="train")
    parser.add_argument("--val_splits", default="validation,val")
    parser.add_argument("--test_splits", default="test")
    parser.add_argument("--fallback_random_split", action="store_true")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--out", default="outputs/checkpoints/pce.pkl")
    parser.add_argument(
        "--metrics_out",
        default="outputs/metrics/pce_train_metrics.json",
    )
    parser.add_argument(
        "--val_pred_out",
        default="outputs/predictions/pce_val_predictions.jsonl",
    )
    parser.add_argument(
        "--test_pred_out",
        default="outputs/predictions/pce_test_predictions.jsonl",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = read_yaml(args.config)

    train_rows = load_pce_training_rows(
        dataset=args.dataset,
        level=args.level,
        splits=parse_csv_splits(args.train_splits),
        base_dir=args.label_base_dir,
    )
    val_rows = load_pce_training_rows(
        dataset=args.dataset,
        level=args.level,
        splits=parse_csv_splits(args.val_splits),
        base_dir=args.label_base_dir,
    )
    test_rows = load_pce_training_rows(
        dataset=args.dataset,
        level=args.level,
        splits=parse_csv_splits(args.test_splits),
        base_dir=args.label_base_dir,
    )

    if not train_rows:
        raise ValueError("No training rows found.")

    # 构造文本
    train_texts, train_labels, train_metas = build_text_label_meta(
    train_rows,
    include_question=False,
    include_answer=False,
    )
    val_texts, val_labels, val_metas = (
    build_text_label_meta(val_rows, include_question=False,include_answer=False) if val_rows else ([], [], [])
    )
    test_texts, test_labels, test_metas = (
    build_text_label_meta(test_rows, include_question=False,include_answer=False) if test_rows else ([], [], [])
    )

    # 如果没现成 val/test，就从 train 随机切
    if args.fallback_random_split and (not val_texts and not test_texts):
        split_pack = maybe_split_train_val_test(
            texts=train_texts,
            labels=train_labels,
            metas=train_metas,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        train_texts, train_labels, train_metas = split_pack["train"]
        val_texts, val_labels, val_metas = split_pack["val"]
        test_texts, test_labels, test_metas = split_pack["test"]

    if len(set(train_labels)) < 2:
        raise ValueError(
            f"Training labels only contain one class: {set(train_labels)}. "
            "PCE needs both positive and negative examples."
        )

    # 建模
    pce = build_pce_from_config(cfg)
    pce.fit(train_texts, train_labels)

    summary: Dict[str, Any] = {
        "dataset": args.dataset,
        "level": args.level,
        "n_train": len(train_texts),
        "n_val": len(val_texts),
        "n_test": len(test_texts),
        "train_positive_rate": float(np.mean(train_labels)) if train_labels else None,
        "val_positive_rate": float(np.mean(val_labels)) if val_labels else None,
        "test_positive_rate": float(np.mean(test_labels)) if test_labels else None,
        "seed": args.seed,
        "config": cfg,
        "model_info": pce.info() if hasattr(pce, "info") else {},
    }

    # val 评估
    if val_texts:
        val_probs = predict_success_probs(pce, val_texts)
        val_metrics = compute_metrics(val_labels, val_probs)
        summary["val"] = val_metrics

        val_pred_rows = build_prediction_rows(val_metas, val_labels, val_probs)
        save_jsonl(Path(args.val_pred_out), val_pred_rows)
    else:
        summary["val"] = None

    # test 评估
    if test_texts:
        test_probs = predict_success_probs(pce, test_texts)
        test_metrics = compute_metrics(test_labels, test_probs)
        summary["test"] = test_metrics

        test_pred_rows = build_prediction_rows(test_metas, test_labels, test_probs)
        save_jsonl(Path(args.test_pred_out), test_pred_rows)
    else:
        summary["test"] = None

    # 保存模型
    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    with out_path.open("wb") as f:
        pickle.dump(pce, f)

    save_json(Path(args.metrics_out), summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved PCE checkpoint to {out_path}")


if __name__ == "__main__":
    main()
