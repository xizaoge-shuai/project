from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from utils.io import read_jsonl


EPS = 1e-6


def clip_probs(probs: np.ndarray) -> np.ndarray:
    return np.clip(probs, EPS, 1.0 - EPS)


def probs_to_logits(probs: np.ndarray) -> np.ndarray:
    probs = clip_probs(probs)
    return np.log(probs / (1.0 - probs))


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def nll_binary(y_true: np.ndarray, probs: np.ndarray) -> float:
    probs = clip_probs(probs)
    return float(
        -np.mean(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs))
    )


def temperature_scale_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = probs_to_logits(probs)
    scaled_logits = logits / max(float(temperature), EPS)
    return logits_to_probs(scaled_logits)


def fit_temperature_from_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    grid_min: float = 0.05,
    grid_max: float = 10.0,
    num_grid: int = 400,
) -> Tuple[float, float]:
    """
    在 validation set 上通过网格搜索最小化 NLL，拟合 temperature。
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)

    grid = np.exp(np.linspace(np.log(grid_min), np.log(grid_max), num_grid))
    best_t = 1.0
    best_nll = float("inf")

    for t in grid:
        scaled = temperature_scale_probs(probs, t)
        loss = nll_binary(labels, scaled)
        if loss < best_nll:
            best_nll = loss
            best_t = float(t)

    return best_t, best_nll


def fit_platt_from_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    C: float = 1e6,
) -> Dict[str, float]:
    """
    用 validation raw probs 拟合 Platt scaling:
        p_cal = sigmoid(a * logit(p_raw) + b)
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)

    x = probs_to_logits(probs).reshape(-1, 1)
    clf = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=1000,
    )
    clf.fit(x, labels)

    a = float(clf.coef_[0, 0])
    b = float(clf.intercept_[0])
    return {"a": a, "b": b}


def apply_platt_to_probs(probs: np.ndarray, a: float, b: float) -> np.ndarray:
    logits = probs_to_logits(probs)
    calibrated_logits = a * logits + b
    return logits_to_probs(calibrated_logits)


def load_predictions(path: str) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    rows = read_jsonl(path)
    y_true = np.asarray([int(r["label_success"]) for r in rows], dtype=int)
    y_prob = np.asarray([float(r["success_prob"]) for r in rows], dtype=float)
    return rows, y_true, y_prob


def save_json(path: str, obj: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def fit_temperature_cli(pred_path: str, out_path: str) -> None:
    _, y_true, y_prob = load_predictions(pred_path)
    best_t, best_nll = fit_temperature_from_probs(y_prob, y_true)

    payload = {
        "method": "temperature",
        "temperature": best_t,
        "validation_nll": best_nll,
        "n": int(len(y_true)),
    }
    save_json(out_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def fit_platt_cli(pred_path: str, out_path: str) -> None:
    _, y_true, y_prob = load_predictions(pred_path)
    params = fit_platt_from_probs(y_prob, y_true)

    calibrated = apply_platt_to_probs(y_prob, params["a"], params["b"])
    payload = {
        "method": "platt",
        "a": params["a"],
        "b": params["b"],
        "validation_nll": nll_binary(y_true.astype(float), calibrated),
        "n": int(len(y_true)),
    }
    save_json(out_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def apply_calibrator_cli(
    pred_path: str,
    calibrator_path: str,
    out_path: str,
) -> None:
    rows, _, y_prob = load_predictions(pred_path)

    with open(calibrator_path, "r", encoding="utf-8") as f:
        calib = json.load(f)

    method = calib["method"]
    if method == "temperature":
        calibrated = temperature_scale_probs(y_prob, float(calib["temperature"]))
    elif method == "platt":
        calibrated = apply_platt_to_probs(
            y_prob,
            float(calib["a"]),
            float(calib["b"]),
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

    out_rows: List[Dict[str, Any]] = []
    for r, p_raw, p_cal in zip(rows, y_prob.tolist(), calibrated.tolist()):
        rr = dict(r)
        rr["raw_success_prob"] = float(p_raw)
        rr["success_prob"] = float(p_cal)  # 覆盖成校准后的，方便 pce.evaluate 直接读
        out_rows.append(rr)

    save_jsonl(out_path, out_rows)
    print(f"Saved calibrated predictions to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=["temperature", "platt"],
    )
    parser.add_argument("--fit_predictions", default="")
    parser.add_argument("--apply_predictions", default="")
    parser.add_argument("--calibrator", default="")
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    do_fit = bool(args.fit_predictions)
    do_apply = bool(args.apply_predictions)

    if do_fit == do_apply:
        raise ValueError(
            "Exactly one of --fit_predictions or --apply_predictions must be provided."
        )

    if do_fit:
        if args.method == "temperature":
            fit_temperature_cli(args.fit_predictions, args.out)
        elif args.method == "platt":
            fit_platt_cli(args.fit_predictions, args.out)
        else:
            raise ValueError(f"Unsupported method: {args.method}")

    if do_apply:
        if not args.calibrator:
            raise ValueError("--calibrator is required when using --apply_predictions")
        apply_calibrator_cli(
            pred_path=args.apply_predictions,
            calibrator_path=args.calibrator,
            out_path=args.out,
        )


if __name__ == "__main__":
    main()
