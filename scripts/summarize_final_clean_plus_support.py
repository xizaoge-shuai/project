#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

IN = Path("outputs/metrics/final_summaries/three_model_baseline_cost_acc_table_ALLROWS_logours_clean_plus_qwen7b_math500_support.md")
OUT_BEST = Path("outputs/metrics/final_summaries/final_best_by_view_clean_plus_support.md")
OUT_FAMILY = Path("outputs/metrics/final_summaries/final_family_summary_clean_plus_support.md")

def fnum(x):
    try:
        x = str(x).strip().strip("`")
        if x in {"", "NA"}:
            return None
        return float(x)
    except Exception:
        return None

def fmt(x):
    return "NA" if x is None else f"{x:.4f}"

def is_base(method, family):
    return family == "Base/CoT" or "base" in method.lower() or "cot@1" in method.lower()

rows = []
for line in IN.read_text(encoding="utf-8").splitlines():
    if not line.startswith("| "):
        continue
    if "Model | Dataset" in line or "---" in line:
        continue
    p = [x.strip().strip("`") for x in line.strip("|").split("|")]
    if len(p) < 13:
        continue
    rows.append({
        "model": p[0],
        "dataset": p[1],
        "method": p[2],
        "family": p[3],
        "acc": fnum(p[4]),
        "gain": fnum(p[5]),
        "extra": fnum(p[6]),
        "gain_per_extra": fnum(p[7]),
        "net": fnum(p[8]),
        "net_per_1k": fnum(p[9]),
        "repair_p": fnum(p[10]),
        "harm": fnum(p[11]),
        "case": p[12],
    })

# best by view
lines = []
lines.append("# Final Best by View: Clean + Qwen7B-MATH500 Support Replay")
lines.append("")
lines.append("For Lowest Harm and Repair-P, Base/CoT and non-positive-gain methods are excluded.")
lines.append("")
lines.append("| Model | Dataset | Best Acc | Best Cost-Acc | Best Net/1K | Best Repair-P | Lowest Harm |")
lines.append("|---|---|---|---|---|---|---|")

for key in sorted(set((r["model"], r["dataset"]) for r in rows)):
    cr = [r for r in rows if (r["model"], r["dataset"]) == key]
    pos = [r for r in cr if (r["gain"] or 0) > 0 and not is_base(r["method"], r["family"])]

    best_acc = max(cr, key=lambda r: r["acc"] if r["acc"] is not None else -1e18)

    cost_pool = [r for r in pos if r["gain_per_extra"] is not None]
    best_cost = max(cost_pool, key=lambda r: r["gain_per_extra"]) if cost_pool else None

    net_pool = [r for r in pos if r["net_per_1k"] is not None]
    best_net = max(net_pool, key=lambda r: r["net_per_1k"]) if net_pool else None

    repair_pool = [r for r in pos if r["repair_p"] is not None]
    best_repair = max(repair_pool, key=lambda r: r["repair_p"]) if repair_pool else None

    harm_pool = [r for r in pos if r["harm"] is not None]
    best_harm = min(harm_pool, key=lambda r: r["harm"]) if harm_pool else None

    lines.append(
        f"| {key[0]} | {key[1]} | "
        f"{best_acc['method']} ({fmt(best_acc['acc'])}) | "
        f"{best_cost['method'] + ' (' + fmt(best_cost['gain_per_extra']) + ')' if best_cost else 'NA'} | "
        f"{best_net['method'] + ' (' + fmt(best_net['net_per_1k']) + ')' if best_net else 'NA'} | "
        f"{best_repair['method'] + ' (' + fmt(best_repair['repair_p']) + ')' if best_repair else 'NA'} | "
        f"{best_harm['method'] + ' (' + fmt(best_harm['harm']) + ')' if best_harm else 'NA'} |"
    )

OUT_BEST.write_text("\n".join(lines) + "\n", encoding="utf-8")

# family summary
lines = []
lines.append("# Final Family Summary: Clean + Qwen7B-MATH500 Support Replay")
lines.append("")
lines.append("| Model | Dataset | Family | #Rows | Best Acc | Best Cost-Acc | Best Net/1K | Best Repair-P | Lowest Harm |")
lines.append("|---|---|---|---:|---|---|---|---|---|")

for key in sorted(set((r["model"], r["dataset"], r["family"]) for r in rows)):
    cr = [r for r in rows if (r["model"], r["dataset"], r["family"]) == key]
    pos = [r for r in cr if (r["gain"] or 0) > 0 and not is_base(r["method"], r["family"])]

    best_acc = max(cr, key=lambda r: r["acc"] if r["acc"] is not None else -1e18)

    cost_pool = [r for r in pos if r["gain_per_extra"] is not None]
    best_cost = max(cost_pool, key=lambda r: r["gain_per_extra"]) if cost_pool else None

    net_pool = [r for r in pos if r["net_per_1k"] is not None]
    best_net = max(net_pool, key=lambda r: r["net_per_1k"]) if net_pool else None

    repair_pool = [r for r in pos if r["repair_p"] is not None]
    best_repair = max(repair_pool, key=lambda r: r["repair_p"]) if repair_pool else None

    harm_pool = [r for r in pos if r["harm"] is not None]
    best_harm = min(harm_pool, key=lambda r: r["harm"]) if harm_pool else None

    lines.append(
        f"| {key[0]} | {key[1]} | {key[2]} | {len(cr)} | "
        f"{best_acc['method']} ({fmt(best_acc['acc'])}) | "
        f"{best_cost['method'] + ' (' + fmt(best_cost['gain_per_extra']) + ')' if best_cost else 'NA'} | "
        f"{best_net['method'] + ' (' + fmt(best_net['net_per_1k']) + ')' if best_net else 'NA'} | "
        f"{best_repair['method'] + ' (' + fmt(best_repair['repair_p']) + ')' if best_repair else 'NA'} | "
        f"{best_harm['method'] + ' (' + fmt(best_harm['harm']) + ')' if best_harm else 'NA'} |"
    )

OUT_FAMILY.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("saved:", OUT_BEST)
print("saved:", OUT_FAMILY)
