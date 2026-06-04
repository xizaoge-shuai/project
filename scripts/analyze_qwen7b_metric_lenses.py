#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import math
from pathlib import Path

IN = Path("outputs/metrics/qwen7b_old_baselines_fixed_v2/qwen7b_clean_selected_baselines_summary.md")
OUT_DIR = Path("outputs/metrics/final_summaries")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# case -> full n_samples，用来估算 total extra calls
N_SAMPLES = {
    "qwen7b_asdiv_original_candidates": 2249,
    "qwen7b_gsm8k": 1319,
    "qwen7b_svamp": 300,
    "qwen7b_mathqa_mathqa_500_total2_seed2_margin0": 500,
    "qwen7b_math500_math500_guard_variant_best": 500,
}

# utility 参数：可以在论文里解释为每个 broken 的惩罚强于 fixed，
# 每个 extra call 的成本也被惩罚
LAMBDA_BROKEN = 3.0
MU_EXTRA = 0.02

def fnum(x, default=0.0):
    try:
        return float(str(x).strip())
    except Exception:
        return default

def parse_md_table(fp):
    rows = []
    lines = fp.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if not line.startswith("| "):
            continue
        if "---" in line or "Case | Method" in line:
            continue
        parts = [x.strip() for x in line.strip("|").split("|")]
        if len(parts) != 12:
            continue
        case, method, final_acc, gain, n_eval, changed, fixed, broken, net, extra_sample, repair_p, harm = parts
        rows.append({
            "case": case,
            "method": method,
            "final_acc": fnum(final_acc),
            "gain": fnum(gain),
            "n_eval": int(fnum(n_eval)),
            "changed": int(fnum(changed)),
            "fixed": int(fnum(fixed)),
            "broken": int(fnum(broken)),
            "net": int(fnum(net)),
            "extra_sample": fnum(extra_sample),
            "repair_p": fnum(repair_p),
            "harm": fnum(harm),
        })
    return rows

rows = parse_md_table(IN)

# 补充指标
for r in rows:
    n = N_SAMPLES.get(r["case"], 0)
    total_extra = r["extra_sample"] * n if n else 0.0
    r["total_extra"] = total_extra
    r["gain_per_extra_sample"] = r["gain"] / r["extra_sample"] if r["extra_sample"] > 1e-12 else float("nan")
    r["net_per_1k_calls"] = r["net"] / total_extra * 1000 if total_extra > 1e-12 else float("nan")
    r["fixed_per_1k_calls"] = r["fixed"] / total_extra * 1000 if total_extra > 1e-12 else float("nan")
    r["trigger_eff_net"] = r["net"] / r["n_eval"] if r["n_eval"] else float("nan")
    r["trigger_eff_fixed"] = r["fixed"] / r["n_eval"] if r["n_eval"] else float("nan")
    r["utility"] = r["fixed"] - LAMBDA_BROKEN * r["broken"] - MU_EXTRA * total_extra

def fmt(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "NA"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)

# 1) metric lens table
metric_fp = OUT_DIR / "qwen7b_metric_lens_table.md"
lines = []
lines.append("# Qwen7B Metric Lens Table")
lines.append("")
lines.append(f"`utility = fixed - {LAMBDA_BROKEN} * broken - {MU_EXTRA} * total_extra_calls`")
lines.append("")
lines.append("| Case | Method | Acc | ΔAcc | Extra/Sample | Net | Repair-P | Harm | ΔAcc/Extra | Net/1KCalls | Fixed/1KCalls | Utility |")
lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    lines.append(
        f"| {r['case']} | {r['method']} | {fmt(r['final_acc'])} | {fmt(r['gain'])} | "
        f"{fmt(r['extra_sample'])} | {r['net']} | {fmt(r['repair_p'])} | {fmt(r['harm'])} | "
        f"{fmt(r['gain_per_extra_sample'])} | {fmt(r['net_per_1k_calls'])} | "
        f"{fmt(r['fixed_per_1k_calls'])} | {fmt(r['utility'])} |"
    )
metric_fp.write_text("\n".join(lines) + "\n", encoding="utf-8")

# 2) best by metric
best_fp = OUT_DIR / "qwen7b_best_by_metric.md"
lines = []
lines.append("# Qwen7B Best Method by Metric")
lines.append("")
lines.append("| Case | Best Acc | Best ΔAcc/Extra | Best Net/1KCalls | Best Repair-P | Lowest Harm | Best Utility |")
lines.append("|---|---|---|---|---|---|---|")
for case in sorted(set(r["case"] for r in rows)):
    cr = [r for r in rows if r["case"] == case]
    def name_val(r, key):
        return f"{r['method']} ({fmt(r[key])})"
    best_acc = max(cr, key=lambda r: (r["final_acc"], -r["extra_sample"], -r["harm"]))
    best_gain_cost = max([r for r in cr if not math.isnan(r["gain_per_extra_sample"])], key=lambda r: r["gain_per_extra_sample"], default=None)
    best_net_cost = max([r for r in cr if not math.isnan(r["net_per_1k_calls"])], key=lambda r: r["net_per_1k_calls"], default=None)
    best_repair = max(cr, key=lambda r: r["repair_p"])
    low_harm = min([r for r in cr if r["changed"] > 0], key=lambda r: r["harm"], default=cr[0])
    best_util = max(cr, key=lambda r: r["utility"])

    lines.append(
        f"| {case} | {name_val(best_acc, 'final_acc')} | "
        f"{name_val(best_gain_cost, 'gain_per_extra_sample') if best_gain_cost else 'NA'} | "
        f"{name_val(best_net_cost, 'net_per_1k_calls') if best_net_cost else 'NA'} | "
        f"{name_val(best_repair, 'repair_p')} | {name_val(low_harm, 'harm')} | "
        f"{name_val(best_util, 'utility')} |"
    )
best_fp.write_text("\n".join(lines) + "\n", encoding="utf-8")

# 3) Ours vs best non-ours baseline
cmp_fp = OUT_DIR / "qwen7b_ours_vs_best_baseline_metric_lens.md"
lines = []
lines.append("# Qwen7B Ours vs Best Baseline under Metric Lenses")
lines.append("")
lines.append("| Case | Ours Acc | Best Baseline | Baseline Acc | Acc Gap | Ours Extra/Sample | Base Extra/Sample | Cost Ratio | Ours Harm | Base Harm | Utility Gap | Diagnosis |")
lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for case in sorted(set(r["case"] for r in rows)):
    cr = [r for r in rows if r["case"] == case]
    ours = next((r for r in cr if r["method"] == "Recorded-Ours"), None)
    bases = [r for r in cr if r["method"] != "Recorded-Ours" and not r["method"].startswith("Base-current")]
    if not ours or not bases:
        continue
    best_base = max(bases, key=lambda r: (r["final_acc"], -r["extra_sample"], -r["harm"]))
    cost_ratio = ours["extra_sample"] / best_base["extra_sample"] if best_base["extra_sample"] > 1e-12 else float("nan")
    acc_gap = ours["final_acc"] - best_base["final_acc"]
    util_gap = ours["utility"] - best_base["utility"]

    if acc_gap > 1e-6:
        diag = "Ours higher accuracy"
    elif abs(acc_gap) <= 1e-6 and cost_ratio <= 1.05:
        diag = "Tie accuracy; comparable cost"
    elif abs(acc_gap) <= 1e-6:
        diag = "Tie accuracy; baseline cheaper"
    elif ours["harm"] < best_base["harm"]:
        diag = "Lower accuracy but safer"
    else:
        diag = "Baseline stronger on this case"

    lines.append(
        f"| {case} | {fmt(ours['final_acc'])} | {best_base['method']} | {fmt(best_base['final_acc'])} | "
        f"{fmt(acc_gap)} | {fmt(ours['extra_sample'])} | {fmt(best_base['extra_sample'])} | "
        f"{fmt(cost_ratio)} | {fmt(ours['harm'])} | {fmt(best_base['harm'])} | {fmt(util_gap)} | {diag} |"
    )
cmp_fp.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("saved:", metric_fp)
print("saved:", best_fp)
print("saved:", cmp_fp)
