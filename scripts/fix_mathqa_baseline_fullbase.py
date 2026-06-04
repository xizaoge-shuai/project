#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
from pathlib import Path

CASES = [
    {
        "name": "mathqa_ds7b_optionmap_fair",
        "in_csv": "outputs/metrics/baseline_tts_compare_mathqa/mathqa_ds7b_optionmap_fair_baseline_compare.csv",
        "out_csv": "outputs/metrics/baseline_tts_compare_mathqa/mathqa_ds7b_optionmap_fair_fullbase_fixed_baseline_compare.csv",
        "out_md": "outputs/metrics/baseline_tts_compare_mathqa/mathqa_ds7b_optionmap_fair_fullbase_fixed_baseline_compare.md",
        "full_base": 0.4900,
        "n_samples": 500,
    },
    {
        "name": "mathqa_qwen3b_optionmap_fair",
        "in_csv": "outputs/metrics/baseline_tts_compare_mathqa/mathqa_qwen3b_optionmap_fair_baseline_compare.csv",
        "out_csv": "outputs/metrics/baseline_tts_compare_mathqa/mathqa_qwen3b_optionmap_fair_fullbase_fixed_baseline_compare.csv",
        "out_md": "outputs/metrics/baseline_tts_compare_mathqa/mathqa_qwen3b_optionmap_fair_fullbase_fixed_baseline_compare.md",
        "full_base": 0.4680,
        "n_samples": 500,
    },
]

FIELDS = [
    "method", "base_acc", "final_acc", "gain", "n_eval", "changed",
    "fixed", "broken", "net", "extra_per_target", "extra_per_sample",
    "repair_precision", "harm_rate"
]

for case in CASES:
    in_csv = Path(case["in_csv"])
    out_csv = Path(case["out_csv"])
    out_md = Path(case["out_md"])

    rows = list(csv.DictReader(in_csv.open(encoding="utf-8")))
    fixed_rows = []

    for r in rows:
        net = int(float(r["net"]))
        gain = net / case["n_samples"]
        final = case["full_base"] + gain

        r["base_acc"] = f"{case['full_base']:.4f}"
        r["gain"] = f"{gain:.4f}"
        r["final_acc"] = f"{final:.4f}"
        fixed_rows.append(r)

    fixed_rows.sort(
        key=lambda x: (float(x["final_acc"]), -float(x["extra_per_sample"]), -float(x["harm_rate"])),
        reverse=True
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in fixed_rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})

    lines = []
    lines.append(f"# MathQA OptionMap Baselines Full-base Fixed: {case['name']}")
    lines.append("")
    lines.append("| Method | Base Acc | Final Acc | ΔAcc | n_eval | Changed | Fixed | Broken | Net | Extra/Target | Extra/Sample | Repair-P | Harm |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in fixed_rows:
        lines.append(
            f"| {r['method']} | {r['base_acc']} | {r['final_acc']} | {r['gain']} | "
            f"{r['n_eval']} | {r['changed']} | {r['fixed']} | {r['broken']} | {r['net']} | "
            f"{r['extra_per_target']} | {r['extra_per_sample']} | {r['repair_precision']} | {r['harm_rate']} |"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("saved:", out_csv)
    print("saved:", out_md)
