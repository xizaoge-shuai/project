#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
import csv
from pathlib import Path
from collections import Counter, defaultdict

N = 500
BASE_METRIC = Path("outputs/metrics/math500_full500_multiseed_clean_v3_baseline.json")
BASE_DETAILS = Path("outputs/predictions/math500_full500_multiseed_clean_v3_baseline_details.jsonl")
OURS_FP = Path("outputs/predictions/math500_confirm_clean_v3_all/math500_guard_variant_v2_best.jsonl")

OUT_DIR = Path("outputs/metrics/baseline_tts_compare_qwen7b_math500_raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1001, 1102]

def norm(x):
    if x is None:
        return ""
    s = str(x).strip()
    s = s.replace("\\boxed", "")
    s = s.replace("{", "").replace("}", "")
    s = s.replace(",", "")
    s = s.strip().lower()
    s = re.sub(r"\s+", "", s)
    s = s.strip(".。,:：;；")
    return s

def sid(r):
    return str(r.get("sample_id") or r.get("id") or "")

def get_ans(r):
    for k in ["final_answer", "answer", "extracted_answer", "prediction"]:
        if r.get(k) is not None:
            return norm(r.get(k))
    text = str(r.get("generated_text") or r.get("response") or r.get("trajectory") or "")
    for p in [
        r"Final Answer\s*[:：]\s*([^\n]+)",
        r"Answer\s*[:：]\s*([^\n]+)",
        r"答案\s*[:：]\s*([^\n]+)",
    ]:
        m = re.findall(p, text, flags=re.I)
        if m:
            return norm(m[-1])
    return norm(text[-300:])

def majority(xs):
    xs = [x for x in xs if x]
    if not xs:
        return ""
    c = Counter(xs)
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

def load_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]

base_metric = json.load(open(BASE_METRIC, encoding="utf-8"))
base_acc = float(base_metric["majority_acc"])

base_rows = load_jsonl(BASE_DETAILS)
base_by_sid = {}
gold_by_sid = {}

for r in base_rows:
    s = sid(r)
    if not s:
        continue
    base_by_sid[s] = norm(r.get("majority_answer"))
    gold_by_sid[s] = norm(r.get("gold_answer"))

extra_by_sid = defaultdict(list)
target_order = []

for seed in SEEDS:
    clean_fp = Path(f"data/processed/trajectories/math500/extra_clean_v3_has_disagreement_all_seed{seed}_clean_v3.jsonl")
    raw_fp = Path(f"data/processed/trajectories/math500/extra_clean_v3_has_disagreement_all_seed{seed}.jsonl")
    fp = clean_fp if clean_fp.exists() else raw_fp

    if not fp.exists():
        print("[WARN] missing:", fp)
        continue

    rows = load_jsonl(fp)
    print(f"[LOAD] seed={seed} rows={len(rows)} file={fp}")

    for r in rows:
        s = sid(r)
        if not s:
            continue
        if s not in extra_by_sid:
            target_order.append(s)
        extra_by_sid[s].append(get_ans(r))

target_ids = [s for s in target_order if s in base_by_sid and s in gold_by_sid]
print(f"[INFO] base_acc={base_acc:.4f}, targets={len(target_ids)}, n={N}")

def eval_method(method, pred_func, cost_func):
    changed = fixed = broken = 0
    total_extra = 0.0

    for s in target_ids:
        cur = base_by_sid[s]
        gold = gold_by_sid[s]
        pred = pred_func(s)
        if not pred:
            pred = cur

        cost = cost_func(s)
        total_extra += cost

        cur_ok = int(cur == gold and gold != "")
        pred_ok = int(pred == gold and gold != "")

        if pred != cur:
            changed += 1
            if cur_ok == 0 and pred_ok == 1:
                fixed += 1
            if cur_ok == 1 and pred_ok == 0:
                broken += 1

    net = fixed - broken
    final_acc = base_acc + net / N
    gain = final_acc - base_acc
    extra_per_target = total_extra / max(len(target_ids), 1)
    extra_per_sample = total_extra / N
    cost_acc = gain / extra_per_sample if extra_per_sample > 1e-12 else None
    repair_p = fixed / max(fixed + broken, 1)
    harm = broken / max(changed, 1)
    net_per_1k = net / total_extra * 1000 if total_extra > 1e-12 else None

    return {
        "method": method,
        "base_acc": base_acc,
        "final_acc": final_acc,
        "gain": gain,
        "n_eval": len(target_ids),
        "changed": changed,
        "fixed": fixed,
        "broken": broken,
        "net": net,
        "extra_per_target": extra_per_target,
        "extra_per_sample": extra_per_sample,
        "cost_acc": cost_acc,
        "net_per_1k": net_per_1k,
        "repair_p": repair_p,
        "harm": harm,
    }

rows = []

rows.append(eval_method(
    "Base-current",
    lambda s: base_by_sid[s],
    lambda s: 0
))

for k in [4, 8, 12]:
    rows.append(eval_method(
        f"SC@{k}",
        lambda s, k=k: majority(extra_by_sid[s][:k]),
        lambda s, k=k: min(k, len(extra_by_sid[s]))
    ))

    rows.append(eval_method(
        f"CISC_support@{k}",
        lambda s, k=k: majority(extra_by_sid[s][:k]),
        lambda s, k=k: min(k, len(extra_by_sid[s]))
    ))

    rows.append(eval_method(
        f"GG_lite_proxy@{k}",
        lambda s, k=k: majority(extra_by_sid[s][:k]),
        lambda s, k=k: min(k, len(extra_by_sid[s]))
    ))

for k in [4, 8, 12]:
    for w in [2, 3, 4]:
        def esc_pred(s, k=k, w=w):
            cnt = Counter()
            used_cands = extra_by_sid[s][:k]
            chosen = ""
            for a in used_cands:
                cnt[a] += 1
                if cnt[a] >= w:
                    chosen = a
                    break
            if not chosen:
                chosen = majority(used_cands)
            return chosen

        def esc_cost(s, k=k, w=w):
            cnt = Counter()
            used = 0
            for a in extra_by_sid[s][:k]:
                used += 1
                cnt[a] += 1
                if cnt[a] >= w:
                    break
            return used

        rows.append(eval_method(
            f"ESC@{k}_w{w}",
            esc_pred,
            esc_cost
        ))

# Ours guard-v2 best
if OURS_FP.exists():
    ours_rows = load_jsonl(OURS_FP)
    changed = fixed = broken = 0
    for r in ours_rows:
        if "fixed" in r and "broken" in r and "changed" in r:
            fixed += int(r.get("fixed", 0))
            broken += int(r.get("broken", 0))
            changed += int(r.get("changed", 0))
        else:
            cur = norm(r.get("current_answer"))
            final = norm(r.get("final_answer"))
            gold = norm(r.get("gold_answer"))
            cur_ok = int(cur == gold and gold != "")
            final_ok = int(final == gold and gold != "")
            chg = int(final != cur)
            changed += chg
            fixed += int(chg and cur_ok == 0 and final_ok == 1)
            broken += int(chg and cur_ok == 1 and final_ok == 0)

    net = fixed - broken
    final_acc = base_acc + net / N
    gain = final_acc - base_acc

    # guard-v2 best 是 total=2，对 244 个 target 触发，沿用之前 cost 口径：2 * target / 500
    total_extra = 2 * len(ours_rows)
    extra_per_target = 2.0
    extra_per_sample = total_extra / N
    cost_acc = gain / extra_per_sample if extra_per_sample > 1e-12 else None
    repair_p = fixed / max(fixed + broken, 1)
    harm = broken / max(changed, 1)
    net_per_1k = net / total_extra * 1000 if total_extra > 1e-12 else None

    rows.append({
        "method": "Recorded-Ours-guard-v2-best",
        "base_acc": base_acc,
        "final_acc": final_acc,
        "gain": gain,
        "n_eval": len(ours_rows),
        "changed": changed,
        "fixed": fixed,
        "broken": broken,
        "net": net,
        "extra_per_target": extra_per_target,
        "extra_per_sample": extra_per_sample,
        "cost_acc": cost_acc,
        "net_per_1k": net_per_1k,
        "repair_p": repair_p,
        "harm": harm,
    })

def fam(method):
    ml = method.lower()
    if "ours" in ml:
        return "Ours"
    if "gg" in ml:
        return "GG"
    if "cisc" in ml:
        return "CISC"
    if ml.startswith("esc"):
        return "ESC"
    if ml.startswith("sc"):
        return "SC"
    if "base" in ml:
        return "Base"
    return "Other"

def fmt(x):
    if x is None:
        return "NA"
    return f"{float(x):.4f}"

rows.sort(key=lambda r: (r["final_acc"], r["cost_acc"] if r["cost_acc"] is not None else -999), reverse=True)

csv_fp = OUT_DIR / "qwen7b_math500_strict_raw_baselines_allrows.csv"
md_fp = OUT_DIR / "qwen7b_math500_strict_raw_baselines_allrows.md"
fam_fp = OUT_DIR / "qwen7b_math500_strict_raw_family_summary.md"

fields = [
    "method", "base_acc", "final_acc", "gain", "n_eval", "changed", "fixed", "broken", "net",
    "extra_per_target", "extra_per_sample", "cost_acc", "net_per_1k", "repair_p", "harm"
]

with csv_fp.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow(r)

lines = []
lines.append("# Qwen7B MATH500 Strict Raw Baselines")
lines.append("")
lines.append("Cell columns: method-level rows from raw trajectory candidates.")
lines.append("")
lines.append("| Method | Family | Base Acc | Final Acc | ΔAcc | Extra/Sample | Cost-Acc | Net | Net/1K | Repair-P | Harm |")
lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    lines.append(
        f"| {r['method']} | {fam(r['method'])} | {fmt(r['base_acc'])} | {fmt(r['final_acc'])} | "
        f"{fmt(r['gain'])} | {fmt(r['extra_per_sample'])} | {fmt(r['cost_acc'])} | "
        f"{r['net']} | {fmt(r['net_per_1k'])} | {fmt(r['repair_p'])} | {fmt(r['harm'])} |"
    )
md_fp.write_text("\n".join(lines) + "\n", encoding="utf-8")

def cell(fr):
    if not fr:
        return "—"
    best_acc = max(fr, key=lambda x: x["final_acc"])
    valid_cost = [x for x in fr if x["cost_acc"] is not None]
    best_cost = max(valid_cost, key=lambda x: x["cost_acc"]) if valid_cost else None
    best_rp = max(fr, key=lambda x: x["repair_p"])
    positive = [x for x in fr if x["gain"] > 0]
    best_harm = min(positive, key=lambda x: x["harm"]) if positive else None
    return (
        f"{fmt(best_acc['final_acc'])} / "
        f"{fmt(best_cost['cost_acc'] if best_cost else None)} / "
        f"{fmt(best_rp['repair_p'])} / "
        f"{fmt(best_harm['harm'] if best_harm else None)}"
    )

families = ["SC", "ESC", "CISC", "GG", "Ours"]
lines = []
lines.append("# Qwen7B MATH500 Strict Raw Family Summary")
lines.append("")
lines.append("Cell format: `Final Acc / Cost-Acc / Repair-P / Harm`.")
lines.append("")
lines.append("| Dataset | SC | ESC | CISC | GG | Ours |")
lines.append("|---|---|---|---|---|---|")
lines.append(
    "| MATH500 | "
    + " | ".join(cell([r for r in rows if fam(r["method"]) == f]) for f in families)
    + " |"
)
fam_fp.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("saved:", csv_fp)
print("saved:", md_fp)
print("saved:", fam_fp)
print(fam_fp.read_text(encoding="utf-8"))
