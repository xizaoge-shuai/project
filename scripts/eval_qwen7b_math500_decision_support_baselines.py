#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, re
from pathlib import Path
from collections import Counter

IN = Path("outputs/predictions/math500_confirm_clean_v3_all/math500_guard_variant_v2_best.jsonl")
OUT = Path("outputs/metrics/baseline_tts_compare_qwen7b_missing/qwen7b_math500_decision_support_baselines.md")
OUT.parent.mkdir(parents=True, exist_ok=True)

BASE_ACC = 0.6520
N = 500

def norm(x):
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = s.replace("\\boxed", "").replace("{", "").replace("}", "")
    s = re.sub(r"\s+", "", s)
    return s

rows = []
data = [json.loads(x) for x in IN.open(encoding="utf-8") if x.strip()]

methods = []

def eval_method(name, choose_func):
    fixed = broken = changed = net = 0
    for r in data:
        cur = norm(r.get("current_answer"))
        gold = norm(r.get("gold_answer"))
        cur_ok = int(r.get("current_ok", cur == gold))
        pred = norm(choose_func(r))
        if not pred:
            pred = cur
        final_ok = int(pred == gold)
        chg = int(pred != cur)
        fx = int(chg and (not cur_ok) and final_ok)
        br = int(chg and cur_ok and (not final_ok))
        fixed += fx
        broken += br
        changed += chg
    net = fixed - broken
    final_acc = BASE_ACC + net / N
    repair_p = fixed / (fixed + broken) if fixed + broken else 0.0
    harm = broken / changed if changed else 0.0
    methods.append((name, final_acc, final_acc-BASE_ACC, changed, fixed, broken, net, repair_p, harm))

def current(r):
    return r.get("current_answer")

def top_answer(r):
    return r.get("top_answer") or r.get("variant_top_answer") or r.get("final_answer")

def variant_final(r):
    return r.get("variant_final_answer") or r.get("final_answer")

def orig_majority(r):
    xs = [norm(x) for x in r.get("orig_answers_norm") or r.get("orig_answers") or [] if norm(x)]
    if not xs:
        return current(r)
    return Counter(xs).most_common(1)[0][0]

def support_top_total2(r):
    top = norm(top_answer(r))
    total = int(r.get("top_total_support") or r.get("variant_top_total") or 0)
    return top if total >= 2 else current(r)

def support_top_seed2(r):
    top = norm(top_answer(r))
    seed = int(r.get("top_seed_support") or r.get("variant_top_seed") or 0)
    return top if seed >= 2 else current(r)

def support_variant_best(r):
    return variant_final(r)

eval_method("Base-current", current)
eval_method("Orig-majority", orig_majority)
eval_method("Support-top-total2", support_top_total2)
eval_method("Support-top-seed2", support_top_seed2)
eval_method("Decision-variant-best", support_variant_best)

methods.sort(key=lambda x: x[1], reverse=True)

lines = []
lines.append("# Qwen7B MATH500 Decision Support Baselines")
lines.append("")
lines.append(f"Input: `{IN}`")
lines.append("")
lines.append("| Method | Final Acc | ΔAcc | Changed | Fixed | Broken | Net | Repair-P | Harm |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for m in methods:
    lines.append(f"| {m[0]} | {m[1]:.4f} | {m[2]:.4f} | {m[3]} | {m[4]} | {m[5]} | {m[6]} | {m[7]:.4f} | {m[8]:.4f} |")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("saved:", OUT)
