#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import json
import math
import re
from pathlib import Path

OUT_DIR = Path("outputs/metrics/final_summaries")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 数据集总样本数，用来算 total extra calls / Net per 1K calls
N_SAMPLES = {
    "gsm8k": 1319,
    "svamp": 300,
    "asdiv": 2249,
    "asdiv_numeric": 2249,
    "mathqa": 500,
    "math500": 500,
    "math500_short": 500,
    "math500_long1024": 500,
    "bbh_formal_fallacies": 100,
    "bbh_logical_deduction_five_objects": 100,
}

# Qwen7B 只保留已经确认过的 clean case，避免混入 cross_asdiv 小 subset / decision-only 假 replay
QWEN7B_KEEP = {
    "qwen7b_asdiv_original_candidates",
    "qwen7b_mathqa_mathqa_500_total2_seed2_margin0",
    "qwen7b_svamp",
    "qwen7b_gsm8k",
    "qwen7b_math500_math500_guard_variant_best",
}

def fnum(x, default=None):
    if x is None:
        return default
    try:
        s = str(x).strip()
        if not s or s.upper() in {"NA", "N/A", "NONE"}:
            return default
        return float(s)
    except Exception:
        return default

def inum(x, default=0):
    v = fnum(x, None)
    if v is None:
        return default
    return int(round(v))

def pick(d, keys, default=None):
    for k in keys:
        if k in d and d[k] not in [None, ""]:
            return d[k]
    return default

def infer_model(case, fp):
    s = (case + " " + str(fp)).lower()
    if "qwen7b" in s:
        return "Qwen7B"
    if "qwen3b" in s:
        return "Qwen3B"
    if "ds7b" in s or "deepseek7b" in s or "deepseek7b" in s:
        return "DS7B"
    if "deepseek14b" in s or "14b" in s:
        return "DS14B"
    return "UNKNOWN"

def infer_dataset(case):
    s = case.lower()
    if "bbh_formal_fallacies" in s:
        return "bbh_formal_fallacies"
    if "bbh_logical_deduction" in s:
        return "bbh_logical_deduction_five_objects"
    if "gsm8k" in s:
        return "gsm8k"
    if "svamp" in s:
        return "svamp"
    if "asdiv" in s:
        return "asdiv_numeric" if "numeric" in s or "asdiv" in s else "asdiv"
    if "mathqa" in s:
        return "mathqa"
    if "math500" in s:
        if "long1024" in s:
            return "math500_long1024"
        if "short" in s:
            return "math500_short"
        return "math500"
    return case

def method_family(method):
    m = method.lower()
    if "recorded-ours" in m or m.startswith("confirm") or "escconfirm" in m or "esc-confirm" in m or "ours" in m:
        return "Ours"
    if "ptrue" in m:
        return "CISC-PTrue"
    if "cisc" in m:
        return "CISC-support"
    if m.startswith("esc"):
        return "ESC"
    if m.startswith("sc"):
        return "SC"
    if "cot" in m or "base" in m:
        return "Base/CoT"
    if "gg" in m:
        return "GG-lite"
    return "Other"

def normalize_row(raw, case, fp, source):
    method = pick(raw, ["method", "Method", "name", "Name"], "")
    if not method:
        return None

    model = infer_model(case, fp)
    if model not in {"Qwen7B", "DS7B", "Qwen3B"}:
        return None

    dataset = infer_dataset(case)

    final_acc = fnum(pick(raw, ["final_acc", "Final Acc", "Best Final Acc", "accuracy", "acc"]))
    base_acc = fnum(pick(raw, ["base_acc", "Base Acc"]))
    gain = fnum(pick(raw, ["gain", "ΔAcc", "acc_gain", "delta_acc"]))
    if gain is None and final_acc is not None and base_acc is not None:
        gain = final_acc - base_acc

    if final_acc is None:
        return None

    changed = inum(pick(raw, ["changed", "Changed"]), 0)
    fixed = inum(pick(raw, ["fixed", "Fixed"]), 0)
    broken = inum(pick(raw, ["broken", "Broken"]), 0)
    net = fnum(pick(raw, ["net", "Net"]), None)
    if net is None:
        net = fixed - broken

    n_eval = inum(pick(raw, ["n_eval", "Eval Samples", "Triggered Targets", "target_n"]), 0)

    extra_per_sample = fnum(pick(raw, [
        "extra_per_sample", "Extra/Sample", "Extra Calls/Sample",
        "extra_calls_per_sample", "total_per_sample", "Total/Sample"
    ]), None)

    extra_per_target = fnum(pick(raw, [
        "extra_per_target", "Extra/Target", "extra_calls_per_target"
    ]), None)

    precision = fnum(pick(raw, [
        "repair_precision", "PRF-P", "precision", "Precision", "safe_precision"
    ]), None)

    harm = fnum(pick(raw, [
        "harm_rate", "Harm", "harm"
    ]), None)

    # 如果没有 repair precision/harm，用 fixed/broken/changed 补
    if precision is None:
        precision = fixed / max(fixed + broken, 1)
    if harm is None:
        harm = broken / max(changed, 1)

    # 过滤明显坏行：除了 Recorded-Ours 的旧 decision 文件，其它 fixed/broken 不应超过 changed
    if final_acc > 1.000001 or final_acc < -0.000001:
        return None
    if method != "Recorded-Ours":
        if changed >= 0 and (fixed > max(changed, 0) or broken > max(changed, 0)):
            return None
        if harm > 1.000001 or precision > 1.000001:
            return None

    n_samples = N_SAMPLES.get(dataset, 0)
    total_extra = None
    if extra_per_sample is not None and n_samples:
        total_extra = extra_per_sample * n_samples

    gain_per_extra = None
    if gain is not None and extra_per_sample is not None and extra_per_sample > 1e-12:
        gain_per_extra = gain / extra_per_sample

    net_per_1k_calls = None
    if total_extra is not None and total_extra > 1e-12:
        net_per_1k_calls = net / total_extra * 1000.0

    # utility: 惩罚 broken 和 extra calls，主要用于 cost-safety 视角
    # 这里参数固定，表里会注明
    lam_broken = 3.0
    mu_extra = 0.02
    utility = None
    if total_extra is not None:
        utility = fixed - lam_broken * broken - mu_extra * total_extra
    else:
        utility = fixed - lam_broken * broken

    return {
        "model": model,
        "dataset": dataset,
        "case": case,
        "method": method,
        "family": method_family(method),
        "base_acc": base_acc,
        "final_acc": final_acc,
        "gain": gain,
        "n_eval": n_eval,
        "changed": changed,
        "fixed": fixed,
        "broken": broken,
        "net": net,
        "extra_per_target": extra_per_target,
        "extra_per_sample": extra_per_sample,
        "precision": precision,
        "harm": harm,
        "gain_per_extra": gain_per_extra,
        "net_per_1k_calls": net_per_1k_calls,
        "utility": utility,
        "source": source,
        "file": str(fp),
    }

def read_csv_rows(fp, case=None, source="csv"):
    rows = []
    try:
        with fp.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                c = case or fp.name
                nr = normalize_row(r, c, fp, source)
                if nr:
                    rows.append(nr)
    except Exception:
        pass
    return rows

def collect_rows():
    rows = []

    # 1) Qwen7B old baseline CSVs
    for root in [
        Path("outputs/metrics/qwen7b_old_baselines"),
        Path("outputs/metrics/qwen7b_old_baselines_fixed"),
        Path("outputs/metrics/qwen7b_old_baselines_fixed_v2"),
    ]:
        if not root.exists():
            continue
        for fp in root.glob("*_old_candidate_baselines.csv"):
            case = fp.name.replace("_old_candidate_baselines.csv", "")
            if case not in QWEN7B_KEEP:
                continue
            rows.extend(read_csv_rows(fp, case, "qwen7b_old_replay"))

    # 2) 标准 baseline compare CSVs：DS7B / Qwen3B
    patterns = [
        "outputs/metrics/baseline_tts_compare/**/*.csv",
        "outputs/metrics/baseline_tts_compare_mathqa/**/*.csv",
        "outputs/metrics/fair_baselines/**/*.csv",
        "outputs/metrics/cisc_ptrue_compare/**/*.csv",
        "outputs/metrics/hybrid_controller_all/**/*.csv",
        "outputs/metrics/hybrid_controller/**/*.csv",
        "outputs/metrics/esc_confirm/**/*.csv",
    ]

    for pat in patterns:
        for fp in Path(".").glob(pat):
            name = fp.name
            # 跳过已经专门处理过的 qwen7b old
            if "qwen7b" in str(fp).lower() and "old_candidate" in name:
                continue
            # 只读看起来像方法对比的 CSV
            if not any(x in name.lower() for x in [
                "baseline", "compare", "summary", "ptrue", "hybrid", "esc", "cisc"
            ]):
                continue
            case = name
            for suf in [
                "_baseline_compare.csv", "_ptrue_cisc_compare.csv",
                "_old_candidate_baselines.csv", "_summary.csv", ".csv"
            ]:
                case = case.replace(suf, "")
            if "baseline_tts_compare_mathqa" in str(fp) and "fullbase_fixed" not in fp.name:
                continue
            # For MathQA, use only full-base-fixed fair optionmap comparison in the main baseline table.
            if "baseline_tts_compare_mathqa" in str(fp) and "fullbase_fixed" not in fp.name:
                continue
            rows.extend(read_csv_rows(fp, case, str(fp.parent)))

    # 去重：同一 model/dataset/case/method/source 可能重复，保留 final_acc 高且 cost 低的一条
    best = {}
    for r in rows:
        key = (r["model"], r["dataset"], r["case"], r["method"])
        old = best.get(key)
        if old is None:
            best[key] = r
        else:
            old_cost = old["extra_per_sample"] if old["extra_per_sample"] is not None else 1e18
            new_cost = r["extra_per_sample"] if r["extra_per_sample"] is not None else 1e18
            if (r["final_acc"], -new_cost, -r["harm"]) > (old["final_acc"], -old_cost, -old["harm"]):
                best[key] = r

    return list(best.values())

def fmt(x, nd=4):
    if x is None:
        return "NA"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "NA"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)

rows = collect_rows()

# coverage
coverage = {}
for r in rows:
    coverage.setdefault((r["model"], r["dataset"]), 0)
    coverage[(r["model"], r["dataset"])] += 1

coverage_fp = OUT_DIR / "three_model_baseline_coverage_ALLROWS.md"
lines = []
lines.append("# Three-model Baseline Coverage")
lines.append("")
lines.append("| Model | Dataset | #Rows | Status |")
lines.append("|---|---|---:|---|")
for model in ["Qwen7B", "DS7B", "Qwen3B"]:
    for dataset in sorted({d for (m, d) in coverage if m == model}):
        n = coverage.get((model, dataset), 0)
        status = "OK" if n > 0 else "MISSING"
        lines.append(f"| {model} | {dataset} | {n} | {status} |")
coverage_fp.write_text("\n".join(lines) + "\n", encoding="utf-8")

# 1) acc table: 每个 model/dataset 按 final acc 排 top 15
acc_fp = OUT_DIR / "three_model_baseline_acc_table_ALLROWS.md"
lines = []
lines.append("# Three-model Baseline Accuracy Table")
lines.append("")
lines.append("| Model | Dataset | Method | Family | Final Acc | ΔAcc | Base Acc | Fixed | Broken | Net | Source Case |")
lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---|")
for model in ["Qwen7B", "DS7B", "Qwen3B"]:
    md_rows = [r for r in rows if r["model"] == model]
    for dataset in sorted(set(r["dataset"] for r in md_rows)):
        cr = [r for r in md_rows if r["dataset"] == dataset]
        cr.sort(key=lambda r: (r["final_acc"], -(r["extra_per_sample"] or 1e18), -r["harm"]), reverse=True)
        for r in cr:
            lines.append(
                f"| {r['model']} | {r['dataset']} | {r['method']} | {r['family']} | "
                f"{fmt(r['final_acc'])} | {fmt(r['gain'])} | {fmt(r['base_acc'])} | "
                f"{r['fixed']} | {r['broken']} | {fmt(r['net'], 0)} | `{r['case']}` |"
            )
acc_fp.write_text("\n".join(lines) + "\n", encoding="utf-8")

# 2) cost table
cost_fp = OUT_DIR / "three_model_baseline_cost_table_ALLROWS.md"
lines = []
lines.append("# Three-model Baseline Cost Table")
lines.append("")
lines.append("| Model | Dataset | Method | Final Acc | Extra/Target | Extra/Sample | n_eval | Changed | Fixed | Broken | Net | Repair-P | Harm |")
lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for model in ["Qwen7B", "DS7B", "Qwen3B"]:
    md_rows = [r for r in rows if r["model"] == model]
    for dataset in sorted(set(r["dataset"] for r in md_rows)):
        cr = [r for r in md_rows if r["dataset"] == dataset]
        cr.sort(key=lambda r: ((r["extra_per_sample"] if r["extra_per_sample"] is not None else 1e18), -r["final_acc"], r["harm"]))
        for r in cr:
            lines.append(
                f"| {r['model']} | {r['dataset']} | {r['method']} | {fmt(r['final_acc'])} | "
                f"{fmt(r['extra_per_target'])} | {fmt(r['extra_per_sample'])} | {r['n_eval']} | "
                f"{r['changed']} | {r['fixed']} | {r['broken']} | {fmt(r['net'], 0)} | "
                f"{fmt(r['precision'])} | {fmt(r['harm'])} |"
            )
cost_fp.write_text("\n".join(lines) + "\n", encoding="utf-8")

# 3) cost-acc table
trade_fp = OUT_DIR / "three_model_baseline_cost_acc_table_ALLROWS.md"
lines = []
lines.append("# Three-model Baseline Cost-Accuracy Table")
lines.append("")
lines.append("Utility is defined as `fixed - 3 * broken - 0.02 * total_extra_calls`.")
lines.append("")
lines.append("| Model | Dataset | Method | Final Acc | ΔAcc | Extra/Sample | ΔAcc/Extra | Net | Net/1K Calls | Repair-P | Harm | Utility | Source Case |")
lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for model in ["Qwen7B", "DS7B", "Qwen3B"]:
    md_rows = [r for r in rows if r["model"] == model]
    for dataset in sorted(set(r["dataset"] for r in md_rows)):
        cr = [r for r in md_rows if r["dataset"] == dataset]
        cr.sort(key=lambda r: (
            r["gain_per_extra"] if r["gain_per_extra"] is not None else -1e18,
            r["final_acc"],
            -r["harm"],
        ), reverse=True)
        for r in cr:
            lines.append(
                f"| {r['model']} | {r['dataset']} | {r['method']} | "
                f"{fmt(r['final_acc'])} | {fmt(r['gain'])} | {fmt(r['extra_per_sample'])} | "
                f"{fmt(r['gain_per_extra'])} | {fmt(r['net'], 0)} | {fmt(r['net_per_1k_calls'])} | "
                f"{fmt(r['precision'])} | {fmt(r['harm'])} | {fmt(r['utility'])} | `{r['case']}` |"
            )
trade_fp.write_text("\n".join(lines) + "\n", encoding="utf-8")

# 4) 每个 model/dataset 的 best-by-view
best_fp = OUT_DIR / "three_model_baseline_best_by_view_ALLROWS.md"
lines = []
lines.append("# Three-model Baseline Best by View")
lines.append("")
lines.append("| Model | Dataset | Best Acc | Best Cost-Acc | Best Net/1K | Best Repair-P | Lowest Harm | Best Utility |")
lines.append("|---|---|---|---|---|---|---|---|")

def nameval(r, k):
    return f"{r['method']} ({fmt(r[k])})"

for model in ["Qwen7B", "DS7B", "Qwen3B"]:
    md_rows = [r for r in rows if r["model"] == model]
    for dataset in sorted(set(r["dataset"] for r in md_rows)):
        cr = [r for r in md_rows if r["dataset"] == dataset]

        best_acc = max(cr, key=lambda r: (r["final_acc"], -(r["extra_per_sample"] or 1e18), -r["harm"]))

        valid_costacc = [r for r in cr if r["gain_per_extra"] is not None]
        best_costacc = max(valid_costacc, key=lambda r: r["gain_per_extra"]) if valid_costacc else None

        valid_net1k = [r for r in cr if r["net_per_1k_calls"] is not None]
        best_net1k = max(valid_net1k, key=lambda r: r["net_per_1k_calls"]) if valid_net1k else None

        best_repair = max(cr, key=lambda r: r["precision"])
        valid_harm = [r for r in cr if r["changed"] > 0]
        lowest_harm = min(valid_harm, key=lambda r: r["harm"]) if valid_harm else best_acc
        best_util = max(cr, key=lambda r: r["utility"] if r["utility"] is not None else -1e18)

        lines.append(
            f"| {model} | {dataset} | {nameval(best_acc, 'final_acc')} | "
            f"{nameval(best_costacc, 'gain_per_extra') if best_costacc else 'NA'} | "
            f"{nameval(best_net1k, 'net_per_1k_calls') if best_net1k else 'NA'} | "
            f"{nameval(best_repair, 'precision')} | {nameval(lowest_harm, 'harm')} | "
            f"{nameval(best_util, 'utility')} |"
        )

best_fp.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("saved:", coverage_fp)
print("saved:", acc_fp)
print("saved:", cost_fp)
print("saved:", trade_fp)
print("saved:", best_fp)
print("rows:", len(rows))
