#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import math

IN = Path("outputs/metrics/final_summaries/three_model_baseline_cost_acc_table_ALLROWS.md")
OUT = Path("outputs/metrics/final_summaries/three_model_baseline_family_summary_ALLROWS.md")

def fnum(x):
    try:
        x = x.strip()
        if x in {"NA", ""}:
            return None
        return float(x)
    except Exception:
        return None

def family(method):
    m = method.lower()
    if "recorded-ours" in m or "confirm" in m or "ours" in m:
        return "Ours"
    if "ptrue" in m:
        return "CISC-PTrue"
    if "cisc" in m:
        return "CISC-support"
    if m.startswith("esc"):
        return "ESC"
    if m.startswith("sc"):
        return "SC"
    if "gg" in m:
        return "GG-lite"
    if "base" in m or "cot" in m:
        return "Base/CoT"
    return "Other"

rows = []
for line in IN.read_text(encoding="utf-8").splitlines():
    if not line.startswith("| "):
        continue
    if "Model | Dataset" in line or "---" in line:
        continue
    parts = [x.strip().strip("`") for x in line.strip("|").split("|")]
    if len(parts) < 13:
        continue
    model, dataset, method = parts[0], parts[1], parts[2]
    r = {
        "model": model,
        "dataset": dataset,
        "method": method,
        "family": family(method),
        "acc": fnum(parts[3]),
        "gain": fnum(parts[4]),
        "extra": fnum(parts[5]),
        "gain_per_extra": fnum(parts[6]),
        "net": fnum(parts[7]),
        "net_per_1k": fnum(parts[8]),
        "repair_p": fnum(parts[9]),
        "harm": fnum(parts[10]),
        "utility": fnum(parts[11]),
        "case": parts[12],
    }
    rows.append(r)

def fmt(x):
    if x is None:
        return "NA"
    return f"{x:.4f}"

lines = []
lines.append("# Three-model Baseline Family Summary: All Rows")
lines.append("")
lines.append("| Model | Dataset | Family | #Rows | Best Acc | Best Cost-Acc | Best Net/1K | Best Repair-P | Lowest Harm | Representative Method |")
lines.append("|---|---|---|---:|---|---|---|---|---|---|")

for key in sorted(set((r["model"], r["dataset"], r["family"]) for r in rows)):
    cr = [r for r in rows if (r["model"], r["dataset"], r["family"]) == key]
    best_acc = max(cr, key=lambda r: (r["acc"] if r["acc"] is not None else -1e9))
    valid_cost = [r for r in cr if r["gain_per_extra"] is not None]
    best_cost = max(valid_cost, key=lambda r: r["gain_per_extra"]) if valid_cost else None
    valid_net = [r for r in cr if r["net_per_1k"] is not None]
    best_net = max(valid_net, key=lambda r: r["net_per_1k"]) if valid_net else None
    best_rp = max(cr, key=lambda r: (r["repair_p"] if r["repair_p"] is not None else -1e9))
    valid_harm = [r for r in cr if r["harm"] is not None]
    best_harm = min(valid_harm, key=lambda r: r["harm"]) if valid_harm else None

    rep = best_acc["method"]

    lines.append(
        f"| {key[0]} | {key[1]} | {key[2]} | {len(cr)} | "
        f"{best_acc['method']} ({fmt(best_acc['acc'])}) | "
        f"{best_cost['method'] + ' (' + fmt(best_cost['gain_per_extra']) + ')' if best_cost else 'NA'} | "
        f"{best_net['method'] + ' (' + fmt(best_net['net_per_1k']) + ')' if best_net else 'NA'} | "
        f"{best_rp['method']} ({fmt(best_rp['repair_p'])}) | "
        f"{best_harm['method'] + ' (' + fmt(best_harm['harm']) + ')' if best_harm else 'NA'} | "
        f"{rep} |"
    )

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("saved:", OUT)
