#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import json
from pathlib import Path

OUT_DIR = Path("outputs/metrics/baseline_14b_compare")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MD = OUT_DIR / "ds14b_correction_quality.md"
OUT_CSV = OUT_DIR / "ds14b_correction_quality.csv"

CASES = [
    (
        "GSM8K",
        Path(
            "outputs/metrics/model_ablation_14b/"
            "gsm8k_deepseek14b_total3_seed1_margin1.json"
        ),
    ),
    (
        "SVAMP",
        Path(
            "outputs/metrics/model_ablation_14b/"
            "svamp_deepseek14b_total2_seed2_margin0.json"
        ),
    ),
    (
        "ASDiv",
        Path(
            "outputs/metrics/model_ablation_14b/"
            "asdiv_deepseek14b_total2_seed2_margin0.json"
        ),
    ),
    (
        "MathQA",
        Path(
            "outputs/metrics/model_ablation_mathqa_optionmap_14b/"
            "mathqa_deepseek14b_optionmap_total2_seed1_margin2.json"
        ),
    ),
    (
        "MATH500",
        Path(
            "outputs/metrics/model_ablation_14b/"
            "math500_deepseek14b_long1024_total2_seed1_margin0.json"
        ),
    ),
]


def div(a, b):
    return a / b if b else 0.0


def fmt(x):
    return f"{float(x):.3f}"


def load_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_current_ok(row):
    for key in [
        "current_ok",
        "base_ok",
        "majority_ok",
        "orig_ok",
        "initial_ok",
        "before_ok",
    ]:
        if key in row and row[key] is not None:
            return int(row[key])

    # final_ok = current_ok + fixed - broken
    if all(k in row for k in ["final_ok", "fixed", "broken"]):
        value = (
            int(row["final_ok"])
            - int(row["fixed"])
            + int(row["broken"])
        )
        if value in (0, 1):
            return value

    raise KeyError(
        "无法确定 current_ok，现有字段为："
        + ", ".join(sorted(row.keys()))
    )


results = []

for dataset, metric_path in CASES:
    if not metric_path.exists():
        raise FileNotFoundError(
            f"{dataset} metric 不存在：{metric_path}"
        )

    metric = json.loads(metric_path.read_text(encoding="utf-8"))

    n_eval = int(metric["n_eval"])
    changed = int(metric["changed"])
    fixed = int(metric["fixed"])
    broken = int(metric["broken"])
    net = int(metric["net"])

    # GSM8K、SVAMP、ASDiv、MATH500 直接由 metric 计算。
    if "current_acc_on_eval" in metric:
        current_correct = round(
            n_eval * float(metric["current_acc_on_eval"])
        )
        target_wrong = n_eval - current_correct
        target_source = "current_acc_on_eval"

    # MathQA 读取 metric 中明确给出的 prediction_file。
    else:
        pred_value = metric.get("prediction_file")
        if not pred_value:
            raise KeyError(
                f"{dataset} 没有 current_acc_on_eval，"
                "也没有 prediction_file"
            )

        pred_path = Path(pred_value)
        if not pred_path.exists():
            raise FileNotFoundError(
                f"{dataset} prediction 不存在：{pred_path}"
            )

        rows = load_jsonl(pred_path)
        current_ok = [get_current_ok(row) for row in rows]
        target_wrong = sum(value == 0 for value in current_ok)
        current_correct = sum(value == 1 for value in current_ok)
        target_source = str(pred_path)

        if len(rows) != n_eval:
            print(
                f"[WARN] {dataset}: metric n_eval={n_eval}, "
                f"prediction rows={len(rows)}"
            )

    precision = div(fixed, changed)
    recall = div(fixed, target_wrong)
    f1 = div(
        2 * precision * recall,
        precision + recall,
    )
    safe_p = div(fixed, fixed + broken)
    harm = div(broken, changed)

    base_acc = float(metric["base_acc"])
    final_acc = float(
        metric.get(
            "estimated_global_acc",
            metric.get("final_acc"),
        )
    )

    result = {
        "dataset": dataset,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "safe_p": safe_p,
        "harm": harm,
        "target_wrong": target_wrong,
        "current_correct": current_correct,
        "changed": changed,
        "fixed": fixed,
        "broken": broken,
        "net": net,
        "n_eval": n_eval,
        "base_acc": base_acc,
        "final_acc": final_acc,
        "metric_file": str(metric_path),
        "target_wrong_source": target_source,
    }
    results.append(result)

    print("=" * 100)
    print("Dataset      :", dataset)
    print("Metric       :", metric_path)
    print("n_eval       :", n_eval)
    print("Target-Wrong :", target_wrong)
    print("Changed      :", changed)
    print("Fixed        :", fixed)
    print("Broken       :", broken)
    print("Net          :", net)
    print(
        "P/R/F1/Safe-P/Harm:",
        fmt(precision),
        fmt(recall),
        fmt(f1),
        fmt(safe_p),
        fmt(harm),
    )


fields = [
    "dataset",
    "precision",
    "recall",
    "f1",
    "safe_p",
    "harm",
    "target_wrong",
    "current_correct",
    "changed",
    "fixed",
    "broken",
    "net",
    "n_eval",
    "base_acc",
    "final_acc",
    "metric_file",
    "target_wrong_source",
]

with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

lines = [
    "# DS14B Correction Quality",
    "",
    "| Dataset | Precision | Recall | F1 | Safe-P | Harm |",
    "|---|---:|---:|---:|---:|---:|",
]

for r in results:
    lines.append(
        f"| {r['dataset']} "
        f"| {fmt(r['precision'])} "
        f"| {fmt(r['recall'])} "
        f"| {fmt(r['f1'])} "
        f"| {fmt(r['safe_p'])} "
        f"| {fmt(r['harm'])} |"
    )

lines += [
    "",
    "## Detailed Counts",
    "",
    "| Dataset | Target-Wrong | Changed | Fixed | Broken | Net | n_eval | Base Acc | Final Acc |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]

for r in results:
    lines.append(
        f"| {r['dataset']} "
        f"| {r['target_wrong']} "
        f"| {r['changed']} "
        f"| {r['fixed']} "
        f"| {r['broken']} "
        f"| {r['net']} "
        f"| {r['n_eval']} "
        f"| {r['base_acc']:.4f} "
        f"| {r['final_acc']:.4f} |"
    )

lines += [
    "",
    "## Source Files",
    "",
]

for r in results:
    lines.append(
        f"- **{r['dataset']}**: `{r['metric_file']}`; "
        f"Target-Wrong source: `{r['target_wrong_source']}`"
    )

OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("\nSaved:", OUT_MD)
print("Saved:", OUT_CSV)
print()
print(OUT_MD.read_text(encoding="utf-8"))
