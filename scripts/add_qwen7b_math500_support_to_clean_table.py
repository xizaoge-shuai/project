#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import re

IN = Path("outputs/metrics/final_summaries/three_model_baseline_cost_acc_table_ALLROWS_logours_clean.md")
SUP = Path("outputs/metrics/baseline_tts_compare_qwen7b_missing/qwen7b_math500_decision_support_baselines.md")
OUT = Path("outputs/metrics/final_summaries/three_model_baseline_cost_acc_table_ALLROWS_logours_clean_plus_qwen7b_math500_support.md")

# Qwen7B-MATH500 日志口径：244 targets, extra/target=2, total extra calls=488, n_samples=500
EXTRA_SAMPLE = 0.9760
TOTAL_EXTRA_CALLS = 488

def parse_support_rows():
    rows = []
    for line in SUP.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| "):
            continue
        if "Method" in line or "---" in line:
            continue
        p = [x.strip() for x in line.strip("|").split("|")]
        if len(p) < 9:
            continue
        method = p[0]
        if method in {"Base-current", "Orig-majority"}:
            continue

        final_acc = float(p[1])
        gain = float(p[2])
        changed = int(float(p[3]))
        fixed = int(float(p[4]))
        broken = int(float(p[5]))
        net = int(float(p[6]))
        repair_p = float(p[7])
        harm = float(p[8])

        gain_per_extra = gain / EXTRA_SAMPLE if EXTRA_SAMPLE > 0 else None
        net_per_1k = net / TOTAL_EXTRA_CALLS * 1000.0

        rows.append([
            "Qwen7B",
            "math500",
            method,
            "Decision-support",
            f"{final_acc:.4f}",
            f"{gain:.4f}",
            f"{EXTRA_SAMPLE:.4f}",
            f"{gain_per_extra:.4f}",
            str(net),
            f"{net_per_1k:.4f}",
            f"{repair_p:.4f}",
            f"{harm:.4f}",
            "qwen7b_math500_decision_support_replay"
        ])
    return rows

lines = IN.read_text(encoding="utf-8").splitlines()

# 去掉旧的同名 support 行，避免重复
kept = []
for line in lines:
    if "qwen7b_math500_decision_support_replay" in line:
        continue
    kept.append(line)

support_rows = parse_support_rows()

# 找到 Qwen7B math500 Ours 后面插入
out = []
inserted = False
for line in kept:
    out.append(line)
    if line.startswith("| Qwen7B | math500 | Recorded-Ours-log-best "):
        for r in support_rows:
            out.append("| " + " | ".join(r[:-1]) + f" | `{r[-1]}` |")
        inserted = True

if not inserted:
    for r in support_rows:
        out.append("| " + " | ".join(r[:-1]) + f" | `{r[-1]}` |")

OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
print("saved:", OUT)
