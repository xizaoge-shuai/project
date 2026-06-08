#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path

ROOT = Path("outputs/metrics/model_ablation_14b")
OUT = ROOT / "ds14b_main_result_summary.md"

def pick(d, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def to_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def fmt(x):
    x = to_float(x)
    if x is None:
        return "NA"
    return f"{x:.4f}"

def infer_dataset(name):
    if name.startswith("math500_deepseek14b_long1024"):
        return "math500-long1024"
    if name.startswith("gsm8k_deepseek14b"):
        return "gsm8k"
    if name.startswith("svamp_deepseek14b"):
        return "svamp"
    if name.startswith("asdiv_deepseek14b"):
        return "asdiv"
    return name.split("_deepseek14b")[0]

groups = {}
skipped = []

for fp in ROOT.glob("*deepseek14b*total*_seed*_margin*.json"):
    try:
        d = json.load(open(fp, encoding="utf-8"))
    except Exception as e:
        skipped.append((str(fp), f"bad json: {e}"))
        continue

    name = fp.name
    ds = infer_dataset(name)

    base = pick(d, [
        "base_acc", "base_accuracy", "majority_acc", "current_acc"
    ])

    final = pick(d, [
        "final_acc", "final_accuracy", "accuracy", "acc", "numeric_acc", "corrected_acc"
    ])

    gain = pick(d, [
        "gain", "acc_gain", "delta_acc", "improvement"
    ])

    base_f = to_float(base)
    final_f = to_float(final)
    gain_f = to_float(gain)

    # 兜底：如果没有 final，但有 base/net/n_samples，则推 final
    if final_f is None:
        net = to_float(pick(d, ["net"]))
        n_samples = to_float(pick(d, ["n_samples", "total_samples", "eval_samples"]))
        if base_f is not None and net is not None and n_samples not in (None, 0):
            final_f = base_f + net / n_samples

    # 兜底：如果没有 gain，用 final-base
    if gain_f is None and base_f is not None and final_f is not None:
        gain_f = final_f - base_f

    if final_f is None:
        skipped.append((str(fp), f"missing final acc keys; keys={sorted(d.keys())}"))
        continue

    item = {
        "dataset": ds,
        "file": str(fp),
        "base": base_f,
        "final": final_f,
        "gain": gain_f if gain_f is not None else 0.0,
        "n_eval": pick(d, ["n_eval", "target_n", "num_targets", "triggered"]),
        "changed": pick(d, ["changed"]),
        "fixed": pick(d, ["fixed"]),
        "broken": pick(d, ["broken"]),
        "net": pick(d, ["net"]),
    }

    old = groups.get(ds)
    if old is None or (item["final"], item["gain"]) > (old["final"], old["gain"]):
        groups[ds] = item

lines = []
lines.append("# DS14B Main Correction Results")
lines.append("")
lines.append("| Dataset | Base Acc | Best Final Acc | ΔAcc | n_eval | Changed | Fixed | Broken | Net | best_file |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")

for ds in sorted(groups):
    r = groups[ds]
    lines.append(
        f"| {ds} | {fmt(r['base'])} | {fmt(r['final'])} | {fmt(r['gain'])} | "
        f"{r['n_eval']} | {r['changed']} | {r['fixed']} | {r['broken']} | {r['net']} | `{r['file']}` |"
    )

if skipped:
    lines.append("")
    lines.append("## Skipped files")
    lines.append("")
    lines.append("| file | reason |")
    lines.append("|---|---|")
    for fp, reason in skipped:
        lines.append(f"| `{fp}` | {reason} |")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("saved:", OUT)
print(OUT.read_text(encoding="utf-8"))
