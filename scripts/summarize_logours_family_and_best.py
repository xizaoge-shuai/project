#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

IN = Path("outputs/metrics/final_summaries/three_model_baseline_cost_acc_table_ALLROWS_logours.md")
OUT_FAMILY = Path("outputs/metrics/final_summaries/three_model_baseline_family_summary_ALLROWS_logours.md")
OUT_BEST = Path("outputs/metrics/final_summaries/three_model_baseline_best_by_view_logours.md")

def fnum(x):
    try:
        x = str(x).strip()
        if x in {"NA", ""}:
            return None
        return float(x)
    except Exception:
        return None

def fmt(x):
    if x is None:
        return "NA"
    return f"{x:.4f}"

def family(method):
    m = method.lower()
    if "ours" in m or ("confirm" in m and "cisc" not in m and "ptrue" not in m):
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
    p = [x.strip().strip("`") for x in line.strip("|").split("|")]
    if len(p) < 13:
        continue
    rows.append({
        "model": p[0],
        "dataset": p[1],
        "method": p[2],
        "family": family(p[2]),
        "acc": fnum(p[3]),
        "gain": fnum(p[4]),
        "extra": fnum(p[5]),
        "gain_per_extra": fnum(p[6]),
        "net": fnum(p[7]),
        "net_per_1k": fnum(p[8]),
        "repair_p": fnum(p[9]),
        "harm": fnum(p[10]),
        "utility": fnum(p[11]),
        "case": p[12],
    })

# family summary
lines = []
lines.append("# Three-model Baseline Family Summary: ALLROWS + Log-canonical Ours")
lines.append("")
lines.append("| Model | Dataset | Family | #Rows | Best Acc | Best Cost-Acc | Best Net/1K | Best Repair-P | Lowest Harm | Representative Method |")
lines.append("|---|---|---|---:|---|---|---|---|---|---|")

for key in sorted(set((r["model"], r["dataset"], r["family"]) for r in rows)):
    cr = [r for r in rows if (r["model"], r["dataset"], r["family"]) == key]
    best_acc = max(cr, key=lambda r: r["acc"] if r["acc"] is not None else -1e18)
    vc = [r for r in cr if r["gain_per_extra"] is not None]
    best_cost = max(vc, key=lambda r: r["gain_per_extra"]) if vc else None
    vn = [r for r in cr if r["net_per_1k"] is not None]
    best_net = max(vn, key=lambda r: r["net_per_1k"]) if vn else None
    vr = [r for r in cr if r["repair_p"] is not None]
    best_rp = max(vr, key=lambda r: r["repair_p"]) if vr else None
    vh = [r for r in cr if r["harm"] is not None]
    best_harm = min(vh, key=lambda r: r["harm"]) if vh else None

    lines.append(
        f"| {key[0]} | {key[1]} | {key[2]} | {len(cr)} | "
        f"{best_acc['method']} ({fmt(best_acc['acc'])}) | "
        f"{best_cost['method'] + ' (' + fmt(best_cost['gain_per_extra']) + ')' if best_cost else 'NA'} | "
        f"{best_net['method'] + ' (' + fmt(best_net['net_per_1k']) + ')' if best_net else 'NA'} | "
        f"{best_rp['method'] + ' (' + fmt(best_rp['repair_p']) + ')' if best_rp else 'NA'} | "
        f"{best_harm['method'] + ' (' + fmt(best_harm['harm']) + ')' if best_harm else 'NA'} | "
        f"{best_acc['method']} |"
    )

OUT_FAMILY.write_text("\n".join(lines) + "\n", encoding="utf-8")

# best by view
lines = []
lines.append("# Three-model Baseline Best by View: Log-canonical Ours")
lines.append("")
lines.append("| Model | Dataset | Best Acc | Best Cost-Acc | Best Net/1K | Best Repair-P | Lowest Harm |")
lines.append("|---|---|---|---|---|---|---|")

for key in sorted(set((r["model"], r["dataset"]) for r in rows)):
    cr = [r for r in rows if (r["model"], r["dataset"]) == key]
    best_acc = max(cr, key=lambda r: r["acc"] if r["acc"] is not None else -1e18)
    vc = [r for r in cr if r["gain_per_extra"] is not None]
    best_cost = max(vc, key=lambda r: r["gain_per_extra"]) if vc else None
    vn = [r for r in cr if r["net_per_1k"] is not None]
    best_net = max(vn, key=lambda r: r["net_per_1k"]) if vn else None
    vr = [r for r in cr if r["repair_p"] is not None]
    best_rp = max(vr, key=lambda r: r["repair_p"]) if vr else None
    vh = [r for r in cr if r["harm"] is not None]
    best_harm = min(vh, key=lambda r: r["harm"]) if vh else None

    lines.append(
        f"| {key[0]} | {key[1]} | "
        f"{best_acc['method']} ({fmt(best_acc['acc'])}) | "
        f"{best_cost['method'] + ' (' + fmt(best_cost['gain_per_extra']) + ')' if best_cost else 'NA'} | "
        f"{best_net['method'] + ' (' + fmt(best_net['net_per_1k']) + ')' if best_net else 'NA'} | "
        f"{best_rp['method'] + ' (' + fmt(best_rp['repair_p']) + ')' if best_rp else 'NA'} | "
        f"{best_harm['method'] + ' (' + fmt(best_harm['harm']) + ')' if best_harm else 'NA'} |"
    )

OUT_BEST.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("saved:", OUT_FAMILY)
print("saved:", OUT_BEST)
