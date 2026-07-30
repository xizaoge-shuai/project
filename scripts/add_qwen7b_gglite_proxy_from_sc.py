#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

IN = Path("outputs/metrics/final_summaries/three_model_baseline_cost_acc_table_ALLROWS_logours_clean_plus_qwen7b_math500_support.md")
OUT = Path("outputs/metrics/final_summaries/three_model_baseline_cost_acc_table_ALLROWS_logours_clean_plus_qwen7b_math500_support_ggproxy.md")

lines = IN.read_text(encoding="utf-8").splitlines()
out = []
added = 0

for line in lines:
    out.append(line)
    if not line.startswith("| Qwen7B |"):
        continue
    if "| SC@" not in line:
        continue

    parts = [x.strip() for x in line.strip("|").split("|")]
    if len(parts) < 13:
        continue

    # clone SC row as GG-lite-proxy
    parts[2] = parts[2].replace("SC@", "GG_lite_proxy@")
    parts[3] = "GG-lite-proxy"
    parts[12] = parts[12].replace("`", "") + "_gglite_proxy"
    out.append("| " + " | ".join(parts) + " |")
    added += 1

OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
print("saved:", OUT)
print("added:", added)
