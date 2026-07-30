#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, re, csv
from pathlib import Path
from collections import Counter, defaultdict

BASE_DETAILS = Path("outputs/predictions/math500_full500_multiseed_clean_v3_baseline_details.jsonl")
BASE_METRIC = Path("outputs/metrics/math500_full500_multiseed_clean_v3_baseline.json")
OUT_DIR = Path("outputs/metrics/own_generation_baselines")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def norm(x):
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = s.replace("\\boxed", "")
    s = s.replace("{", "").replace("}", "")
    s = s.replace(",", "")
    s = re.sub(r"\s+", "", s)
    return s.strip(".。,:：;；")

def sid(r):
    return str(r.get("sample_id") or r.get("id") or r.get("qid") or "")

def ans(r):
    for k in ["final_answer", "answer", "extracted_answer", "prediction"]:
        if r.get(k) is not None:
            return norm(r.get(k))
    txt = str(r.get("generated_text") or r.get("response") or r.get("trajectory") or r.get("text") or "")
    for p in [
        r"Final Answer\s*[:：]\s*([^\n]+)",
        r"Answer\s*[:：]\s*([^\n]+)",
        r"答案\s*[:：]\s*([^\n]+)",
    ]:
        m = re.findall(p, txt, flags=re.I)
        if m:
            return norm(m[-1])
    return norm(txt[-300:])

def majority(xs):
    xs = [x for x in xs if x]
    if not xs:
        return ""
    c = Counter(xs)
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

base_metric = json.load(open(BASE_METRIC, encoding="utf-8"))
base_acc = float(base_metric["majority_acc"])
N = int(base_metric["n_samples"])

base_by_sid = {}
gold_by_sid = {}

for line in BASE_DETAILS.open(encoding="utf-8"):
    if not line.strip():
        continue
    r = json.loads(line)
    s = sid(r)
    if not s:
        continue
    base_by_sid[s] = norm(r.get("majority_answer"))
    gold_by_sid[s] = norm(r.get("gold_answer"))

# 自动找 independent 生成出来的候选 jsonl
cands = []
for root in [Path("data/processed/trajectories/own_generation_baselines"), Path("outputs")]:
    if not root.exists():
        continue
    for fp in root.rglob("*.jsonl"):
        name = str(fp).lower()
        if "qwen7b" in name and "math500" in name and ("independent" in name or "owngen" in name):
            if any(bad in name for bad in ["baseline_details", "prediction", "final", "metric"]):
                continue
            cands.append(fp)

cands = sorted(set(cands))
print("[candidate files]")
for fp in cands:
    print(fp)

extra_by_sid = defaultdict(list)

for fp in cands:
    try:
        rows = [json.loads(x) for x in fp.open(encoding="utf-8") if x.strip()]
    except Exception:
        continue
    raw_like = False
    for r in rows[:5]:
        if any(k in r for k in ["generated_text", "response", "trajectory", "text", "final_answer", "answer"]):
            raw_like = True
    if not raw_like:
        continue

    used = 0
    for r in rows:
        s = sid(r)
        if not s:
            continue
        a = ans(r)
        if a:
            extra_by_sid[s].append(a)
            used += 1
    print(f"[load] {fp} rows={len(rows)} used={used}")

target_ids = [s for s in base_by_sid if s in gold_by_sid]
inter = sum(1 for s in target_ids if s in extra_by_sid)
print(f"[info] base_acc={base_acc:.4f}, N={N}, base_rows={len(base_by_sid)}, ids_with_extra={inter}")

if inter == 0:
    print("[ERROR] 没有匹配到 candidate 文件或 sample_id 对不上。先运行：")
    print("find data/processed/trajectories/own_generation_baselines outputs -type f -iname '*qwen7b*math500*independent*.jsonl' | sort")
    raise SystemExit(1)

def eval_method(name, pred_func, cost_func):
    changed = fixed = broken = correct = 0
    total_extra = 0

    for s in target_ids:
        cur = base_by_sid[s]
        gold = gold_by_sid[s]
        pred = pred_func(s) or cur

        cur_ok = int(cur == gold and gold != "")
        pred_ok = int(pred == gold and gold != "")

        correct += pred_ok
        total_extra += cost_func(s)

        if pred != cur:
            changed += 1
            if cur_ok == 0 and pred_ok == 1:
                fixed += 1
            elif cur_ok == 1 and pred_ok == 0:
                broken += 1

    net = fixed - broken
    final_by_net = base_acc + net / N
    final_direct = correct / N

    extra_per_sample = total_extra / N
    gain = final_by_net - base_acc
    cost_acc = gain / extra_per_sample if extra_per_sample > 0 else None
    repair_p = fixed / (fixed + broken) if fixed + broken else 0.0
    harm = broken / changed if changed else 0.0

    # 两种算法应一致；不一致说明 base/pred/gold 归一化还有问题
    mismatch = abs(final_by_net - final_direct)

    return {
        "method": name,
        "base_acc": base_acc,
        "final_acc": final_by_net,
        "final_direct": final_direct,
        "mismatch": mismatch,
        "gain": gain,
        "changed": changed,
        "fixed": fixed,
        "broken": broken,
        "net": net,
        "extra_per_sample": extra_per_sample,
        "cost_acc": cost_acc,
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
        lambda s, k=k: majority(extra_by_sid.get(s, [])[:k]),
        lambda s, k=k: min(k, len(extra_by_sid.get(s, [])))
    ))
    rows.append(eval_method(
        f"CISC_support_proxy@{k}",
        lambda s, k=k: majority(extra_by_sid.get(s, [])[:k]),
        lambda s, k=k: min(k, len(extra_by_sid.get(s, [])))
    ))
    rows.append(eval_method(
        f"GG_lite_proxy@{k}",
        lambda s, k=k: majority(extra_by_sid.get(s, [])[:k]),
        lambda s, k=k: min(k, len(extra_by_sid.get(s, [])))
    ))

for k in [4, 8, 12]:
    for w in [2, 3, 4]:
        def esc_pred(s, k=k, w=w):
            cnt = Counter()
            chosen = ""
            for a in extra_by_sid.get(s, [])[:k]:
                cnt[a] += 1
                if cnt[a] >= w:
                    chosen = a
                    break
            return chosen or majority(extra_by_sid.get(s, [])[:k])

        def esc_cost(s, k=k, w=w):
            cnt = Counter()
            used = 0
            for a in extra_by_sid.get(s, [])[:k]:
                used += 1
                cnt[a] += 1
                if cnt[a] >= w:
                    break
            return used

        rows.append(eval_method(f"ESC@{k}_w{w}", esc_pred, esc_cost))

rows.sort(key=lambda r: (r["final_acc"], r["cost_acc"] if r["cost_acc"] is not None else -999), reverse=True)

def fmt(x):
    if x is None:
        return "NA"
    return f"{float(x):.4f}"

out_md = OUT_DIR / "qwen7b_math500_independent_owngen_fixed_eval.md"
out_csv = OUT_DIR / "qwen7b_math500_independent_owngen_fixed_eval.csv"

fields = [
    "method", "base_acc", "final_acc", "final_direct", "mismatch", "gain",
    "changed", "fixed", "broken", "net", "extra_per_sample",
    "cost_acc", "repair_p", "harm"
]

with out_csv.open("w", encoding="utf-8", newline="") as f:
    wcsv = csv.DictWriter(f, fieldnames=fields)
    wcsv.writeheader()
    for r in rows:
        wcsv.writerow(r)

lines = []
lines.append("# Qwen7B MATH500 Independent-generation Baselines Fixed Eval")
lines.append("")
lines.append("| Method | Base Acc | Final Acc | Direct Acc | Mismatch | ΔAcc | Changed | Fixed | Broken | Net | Extra/Sample | Cost-Acc | Repair-P | Harm |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    lines.append(
        f"| {r['method']} | {fmt(r['base_acc'])} | {fmt(r['final_acc'])} | {fmt(r['final_direct'])} | "
        f"{fmt(r['mismatch'])} | {fmt(r['gain'])} | {r['changed']} | {r['fixed']} | {r['broken']} | {r['net']} | "
        f"{fmt(r['extra_per_sample'])} | {fmt(r['cost_acc'])} | {fmt(r['repair_p'])} | {fmt(r['harm'])} |"
    )

out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("saved:", out_md)
print("saved:", out_csv)
print(out_md.read_text(encoding="utf-8"))
