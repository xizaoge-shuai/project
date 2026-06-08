#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import json
from pathlib import Path

in_csv = Path("outputs/metrics/baseline_tts_compare_mathqa_14b/mathqa_ds14b_optionmap_fair_baseline_compare.csv")
out_csv = Path("outputs/metrics/baseline_tts_compare_mathqa_14b/mathqa_ds14b_optionmap_fair_fullbase_fixed_baseline_compare.csv")
out_md = Path("outputs/metrics/baseline_tts_compare_mathqa_14b/mathqa_ds14b_optionmap_fair_fullbase_fixed_baseline_compare.md")

base_json = Path("outputs/metrics/model_ablation_mathqa_optionmap_14b/mathqa_deepseek14b_base.json")
if base_json.exists():
    bj = json.load(open(base_json, encoding="utf-8"))
    full_base = float(bj.get("base_acc", 0.7120))
    n_samples = int(bj.get("n_samples", 500))
else:
    full_base = 0.7120
    n_samples = 500

rows = list(csv.DictReader(in_csv.open(encoding="utf-8")))
fields = list(rows[0].keys())

for r in rows:
    net = int(float(r["net"]))
    gain = net / n_samples
    final = full_base + gain

    r["base_acc"] = f"{full_base:.4f}"
    r["gain"] = f"{gain:.4f}"
    r["final_acc"] = f"{final:.4f}"

rows.sort(
    key=lambda r: (
        float(r["final_acc"]),
        float(r["gain"]) / float(r["extra_per_sample"]) if float(r["extra_per_sample"]) > 0 else -999
    ),
    reverse=True
)

with out_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow(r)

lines = []
lines.append("# DS14B MathQA OptionMap Baselines Full-base Fixed")
lines.append("")
lines.append("| Method | Base Acc | Final Acc | ΔAcc | n_eval | Changed | Fixed | Broken | Net | Extra/Target | Extra/Sample | Cost-Acc | Repair-P | Harm |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for r in rows:
    extra = float(r["extra_per_sample"])
    gain = float(r["gain"])
    cost_acc = gain / extra if extra > 0 else None
    cost_acc_s = "NA" if cost_acc is None else f"{cost_acc:.4f}"

    lines.append(
        f"| {r['method']} | {r['base_acc']} | {r['final_acc']} | {r['gain']} | "
        f"{r['n_eval']} | {r['changed']} | {r['fixed']} | {r['broken']} | {r['net']} | "
        f"{r['extra_per_target']} | {r['extra_per_sample']} | {cost_acc_s} | "
        f"{r['repair_precision']} | {r['harm_rate']} |"
    )

out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("saved:", out_csv)
print("saved:", out_md)
print(out_md.read_text(encoding="utf-8"))
