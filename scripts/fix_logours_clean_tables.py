#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

IN = Path("outputs/metrics/final_summaries/three_model_baseline_cost_acc_table_ALLROWS_logours.md")
OUT = Path("outputs/metrics/final_summaries/three_model_baseline_cost_acc_table_ALLROWS_logours_clean.md")
BEST = Path("outputs/metrics/final_summaries/three_model_baseline_best_by_view_logours_clean.md")
FAMILY = Path("outputs/metrics/final_summaries/three_model_baseline_family_summary_ALLROWS_logours_clean.md")

# 按日志修正 Ours 的全量 extra calls，用于 Net/1K
EXTRA_CALLS = {
    ("Qwen7B", "gsm8k"): 2520,
    ("Qwen7B", "svamp"): 972,
    ("Qwen7B", "asdiv_numeric"): 11964,
    ("Qwen7B", "mathqa"): 2688,
    ("Qwen7B", "math500"): 488,

    ("DS7B", "asdiv_numeric"): 16320,
    ("DS7B", "gsm8k"): 13272,
    ("DS7B", "svamp"): 2244,
    ("DS7B", "math500_long1024"): 5352,
    ("DS7B", "mathqa"): 10488,
    ("DS7B", "bbh_formal_fallacies"): 390,
    ("DS7B", "bbh_logical_deduction_five_objects"): 426,

    ("Qwen3B", "asdiv_numeric"): 12996,
    ("Qwen3B", "gsm8k"): 12900,
    ("Qwen3B", "svamp"): 1764,
    ("Qwen3B", "math500_long1024"): 5232,
    ("Qwen3B", "mathqa"): 5160,
    ("Qwen3B", "bbh_formal_fallacies"): 108,
    ("Qwen3B", "bbh_logical_deduction_five_objects"): 162,
}

def fnum(x):
    try:
        x = str(x).strip()
        if x in {"", "NA"}:
            return None
        return float(x)
    except Exception:
        return None

def fmt(x):
    if x is None:
        return "NA"
    return f"{x:.4f}"

def is_base(method):
    m = method.lower()
    return "base" in m or "cot@1" in m

def is_ours(method):
    m = method.lower()
    return "ours" in m or ("confirm" in m and "cisc" not in m and "ptrue" not in m)

def family(method):
    m = method.lower()
    if is_ours(method):
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
    if is_base(method):
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

    r = {
        "model": p[0],
        "dataset": p[1],
        "method": p[2],
        "acc": fnum(p[3]),
        "gain": fnum(p[4]),
        "extra_sample": fnum(p[5]),
        "gain_per_extra": fnum(p[6]),
        "net": fnum(p[7]),
        "net_per_1k": fnum(p[8]),
        "repair_p": fnum(p[9]),
        "harm": fnum(p[10]),
        "case": p[12],
    }

    # 修正 Ours 的 Net/1K
    if is_ours(r["method"]):
        key = (r["model"], r["dataset"])
        if key in EXTRA_CALLS and r["net"] is not None:
            r["net_per_1k"] = r["net"] / EXTRA_CALLS[key] * 1000.0

    r["family"] = family(r["method"])
    rows.append(r)

# clean allrows table, 去掉 Utility
lines = []
lines.append("# Three-model Baseline Cost-Acc Table: ALLROWS + Log-canonical Ours Clean")
lines.append("")
lines.append("| Model | Dataset | Method | Family | Final Acc | ΔAcc | Extra/Sample | ΔAcc/Extra | Net | Net/1K | Repair-P | Harm | Case |")
lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for r in rows:
    lines.append(
        f"| {r['model']} | {r['dataset']} | {r['method']} | {r['family']} | "
        f"{fmt(r['acc'])} | {fmt(r['gain'])} | {fmt(r['extra_sample'])} | {fmt(r['gain_per_extra'])} | "
        f"{fmt(r['net'])} | {fmt(r['net_per_1k'])} | {fmt(r['repair_p'])} | {fmt(r['harm'])} | `{r['case']}` |"
    )
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

# best by view clean：Lowest Harm 排除 Base/CoT 和 non-positive gain
lines = []
lines.append("# Three-model Baseline Best by View: Log-canonical Ours Clean")
lines.append("")
lines.append("For Lowest Harm, Base/CoT and non-positive-gain methods are excluded.")
lines.append("")
lines.append("| Model | Dataset | Best Acc | Best Cost-Acc | Best Net/1K | Best Repair-P | Lowest Harm |")
lines.append("|---|---|---|---|---|---|---|")

for key in sorted(set((r["model"], r["dataset"]) for r in rows)):
    cr = [r for r in rows if (r["model"], r["dataset"]) == key]

    best_acc = max(cr, key=lambda r: r["acc"] if r["acc"] is not None else -1e18)

    vc = [r for r in cr if r["gain_per_extra"] is not None and (r["gain"] or 0) > 0]
    best_cost = max(vc, key=lambda r: r["gain_per_extra"]) if vc else None

    vn = [r for r in cr if r["net_per_1k"] is not None and (r["gain"] or 0) > 0]
    best_net = max(vn, key=lambda r: r["net_per_1k"]) if vn else None

    vr = [r for r in cr if r["repair_p"] is not None and not is_base(r["method"]) and (r["gain"] or 0) > 0]
    best_rp = max(vr, key=lambda r: r["repair_p"]) if vr else None

    vh = [r for r in cr if r["harm"] is not None and not is_base(r["method"]) and (r["gain"] or 0) > 0]
    best_harm = min(vh, key=lambda r: r["harm"]) if vh else None

    lines.append(
        f"| {key[0]} | {key[1]} | "
        f"{best_acc['method']} ({fmt(best_acc['acc'])}) | "
        f"{best_cost['method'] + ' (' + fmt(best_cost['gain_per_extra']) + ')' if best_cost else 'NA'} | "
        f"{best_net['method'] + ' (' + fmt(best_net['net_per_1k']) + ')' if best_net else 'NA'} | "
        f"{best_rp['method'] + ' (' + fmt(best_rp['repair_p']) + ')' if best_rp else 'NA'} | "
        f"{best_harm['method'] + ' (' + fmt(best_harm['harm']) + ')' if best_harm else 'NA'} |"
    )
BEST.write_text("\n".join(lines) + "\n", encoding="utf-8")

# family summary clean
lines = []
lines.append("# Three-model Baseline Family Summary: ALLROWS + Log-canonical Ours Clean")
lines.append("")
lines.append("| Model | Dataset | Family | #Rows | Best Acc | Best Cost-Acc | Best Net/1K | Best Repair-P | Lowest Harm |")
lines.append("|---|---|---|---:|---|---|---|---|---|")

for key in sorted(set((r["model"], r["dataset"], r["family"]) for r in rows)):
    cr = [r for r in rows if (r["model"], r["dataset"], r["family"]) == key]

    best_acc = max(cr, key=lambda r: r["acc"] if r["acc"] is not None else -1e18)

    vc = [r for r in cr if r["gain_per_extra"] is not None and (r["gain"] or 0) > 0]
    best_cost = max(vc, key=lambda r: r["gain_per_extra"]) if vc else None

    vn = [r for r in cr if r["net_per_1k"] is not None and (r["gain"] or 0) > 0]
    best_net = max(vn, key=lambda r: r["net_per_1k"]) if vn else None

    vr = [r for r in cr if r["repair_p"] is not None and not is_base(r["method"]) and (r["gain"] or 0) > 0]
    best_rp = max(vr, key=lambda r: r["repair_p"]) if vr else None

    vh = [r for r in cr if r["harm"] is not None and not is_base(r["method"]) and (r["gain"] or 0) > 0]
    best_harm = min(vh, key=lambda r: r["harm"]) if vh else None

    lines.append(
        f"| {key[0]} | {key[1]} | {key[2]} | {len(cr)} | "
        f"{best_acc['method']} ({fmt(best_acc['acc'])}) | "
        f"{best_cost['method'] + ' (' + fmt(best_cost['gain_per_extra']) + ')' if best_cost else 'NA'} | "
        f"{best_net['method'] + ' (' + fmt(best_net['net_per_1k']) + ')' if best_net else 'NA'} | "
        f"{best_rp['method'] + ' (' + fmt(best_rp['repair_p']) + ')' if best_rp else 'NA'} | "
        f"{best_harm['method'] + ' (' + fmt(best_harm['harm']) + ')' if best_harm else 'NA'} |"
    )
FAMILY.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("saved:", OUT)
print("saved:", BEST)
print("saved:", FAMILY)
