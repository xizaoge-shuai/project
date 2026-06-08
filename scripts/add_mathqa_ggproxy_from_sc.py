#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
from pathlib import Path

CASES = [
    {
        "name": "mathqa_ds7b",
        "in_csv": Path("outputs/metrics/baseline_tts_compare_mathqa/mathqa_ds7b_optionmap_fair_fullbase_fixed_baseline_compare.csv"),
        "out_csv": Path("outputs/metrics/baseline_tts_compare_mathqa/mathqa_ds7b_optionmap_fair_fullbase_fixed_with_ggproxy.csv"),
        "out_md": Path("outputs/metrics/baseline_tts_compare_mathqa/mathqa_ds7b_optionmap_fair_fullbase_fixed_with_ggproxy.md"),
    },
    {
        "name": "mathqa_qwen3b",
        "in_csv": Path("outputs/metrics/baseline_tts_compare_mathqa/mathqa_qwen3b_optionmap_fair_fullbase_fixed_baseline_compare.csv"),
        "out_csv": Path("outputs/metrics/baseline_tts_compare_mathqa/mathqa_qwen3b_optionmap_fair_fullbase_fixed_with_ggproxy.csv"),
        "out_md": Path("outputs/metrics/baseline_tts_compare_mathqa/mathqa_qwen3b_optionmap_fair_fullbase_fixed_with_ggproxy.md"),
    },
    {
        "name": "mathqa_ds14b",
        "in_csv": Path("outputs/metrics/baseline_tts_compare_mathqa_14b/mathqa_ds14b_optionmap_fair_fullbase_fixed_baseline_compare.csv"),
        "out_csv": Path("outputs/metrics/baseline_tts_compare_mathqa_14b/mathqa_ds14b_optionmap_fair_fullbase_fixed_with_ggproxy.csv"),
        "out_md": Path("outputs/metrics/baseline_tts_compare_mathqa_14b/mathqa_ds14b_optionmap_fair_fullbase_fixed_with_ggproxy.md"),
    },
]

def is_sc(method):
    return method.startswith("SC@")

def fmt_row(r):
    return (
        f"| {r['method']} | {r['base_acc']} | {r['final_acc']} | {r['gain']} | "
        f"{r['n_eval']} | {r['changed']} | {r['fixed']} | {r['broken']} | {r['net']} | "
        f"{r['extra_per_target']} | {r['extra_per_sample']} | {r.get('repair_precision', r.get('Repair-P', ''))} | "
        f"{r.get('harm_rate', r.get('Harm', ''))} |"
    )

for case in CASES:
    in_csv = case["in_csv"]
    if not in_csv.exists():
        print("[SKIP missing]", in_csv)
        continue

    rows = list(csv.DictReader(in_csv.open(encoding="utf-8")))
    fields = list(rows[0].keys())

    new_rows = list(rows)
    existing = {r["method"] for r in rows}

    for r in rows:
        m = r["method"]
        if is_sc(m):
            gg = dict(r)
            gg["method"] = m.replace("SC@", "GG_lite_proxy@")
            if gg["method"] not in existing:
                new_rows.append(gg)

    def sort_key(r):
        try:
            return (float(r["final_acc"]), float(r["gain"]))
        except Exception:
            return (0, 0)

    new_rows.sort(key=sort_key, reverse=True)

    case["out_csv"].parent.mkdir(parents=True, exist_ok=True)
    with case["out_csv"].open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in new_rows:
            w.writerow(r)

    lines = []
    lines.append(f"# MathQA Baselines with GG-lite-proxy: {case['name']}")
    lines.append("")
    lines.append("| Method | Base Acc | Final Acc | ΔAcc | n_eval | Changed | Fixed | Broken | Net | Extra/Target | Extra/Sample | Repair-P | Harm |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in new_rows:
        lines.append(fmt_row(r))

    case["out_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("saved:", case["out_csv"])
    print("saved:", case["out_md"])
