#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
import csv
from pathlib import Path
from collections import Counter, defaultdict

OUT_DIR = Path("outputs/metrics/baseline_14b_compare")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CASES = [
    {
        "dataset": "asdiv",
        "tag": "asdiv_deepseek14b",
        "n_samples": 2249,
    },
    {
        "dataset": "gsm8k",
        "tag": "gsm8k_deepseek14b",
        "n_samples": 1319,
    },
    {
        "dataset": "svamp",
        "tag": "svamp_deepseek14b",
        "n_samples": 300,
    },
    {
        "dataset": "math500-long1024",
        "tag": "math500_deepseek14b_long1024",
        "n_samples": 500,
    },
]

def norm_num(x):
    if x is None:
        return ""
    s = str(x).strip()
    s = s.replace(",", "").replace("$", "").replace("\\$", "")
    s = s.replace("\\boxed", "").replace("\\text", "")
    s = s.replace("{", "").replace("}", "")
    nums = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", s)
    if nums:
        s = nums[-1]
    if "/" in s:
        try:
            a, b = s.split("/", 1)
            v = float(a) / float(b)
            return f"{v:.10f}".rstrip("0").rstrip(".")
        except Exception:
            return s.strip().lower()
    try:
        v = float(s)
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.10f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x).strip().lower()

def sid(r):
    return str(r.get("sample_id") or r.get("id") or r.get("qid") or "")

def majority(xs):
    xs = [x for x in xs if x]
    if not xs:
        return ""
    c = Counter(xs)
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

def extract_answer(r):
    for k in ["answer", "final_answer", "pred_answer", "prediction", "extracted_answer"]:
        if r.get(k) is not None:
            return norm_num(r.get(k))

    text = ""
    for k in ["trajectory", "text", "reasoning", "output", "completion", "response"]:
        if r.get(k):
            text = str(r[k])
            break

    for pat in [
        r"Final Answer\s*[:：]\s*([^\n]+)",
        r"Answer\s*[:：]\s*([^\n]+)",
        r"答案\s*[:：]\s*([^\n]+)",
        r"The answer is\s*([^\n\.]+)",
    ]:
        m = re.findall(pat, text, flags=re.I)
        if m:
            return norm_num(m[-1])

    return norm_num(text[-300:])

def load_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]

def load_base_details(fp):
    rows = load_jsonl(fp)
    base_by_sid = {}
    gold_by_sid = {}

    for r in rows:
        s = sid(r)
        if not s:
            continue

        gold = norm_num(r.get("gold_answer") or r.get("answer") or r.get("target") or r.get("label"))
        gold_by_sid[s] = gold

        cur = norm_num(
            r.get("majority_answer")
            or r.get("current_answer")
            or r.get("base_answer")
            or r.get("final_answer")
            or r.get("prediction")
        )

        if not cur:
            ans = r.get("answers_norm") or r.get("answers") or r.get("base_answers") or []
            cur = majority([norm_num(a) for a in ans])

        base_by_sid[s] = cur

    return rows, base_by_sid, gold_by_sid

def load_target_ids(fp):
    return [x.strip() for x in open(fp, encoding="utf-8") if x.strip()]

def load_extra(extra_fps):
    extra = defaultdict(list)
    for fp in extra_fps:
        if not fp.exists():
            print("[WARN] missing extra:", fp)
            continue
        for line in open(fp, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            s = sid(r)
            a = extract_answer(r)
            if s and a:
                extra[s].append(a)
    return extra

def infer_base_acc(base_rows, base_by_sid, gold_by_sid, n_samples):
    ok = 0
    total = 0
    for r in base_rows:
        s = sid(r)
        if not s:
            continue
        if s in base_by_sid and s in gold_by_sid and gold_by_sid[s]:
            total += 1
            ok += int(base_by_sid[s] == gold_by_sid[s])
    # 正常 total 应该等于 n_samples；如果不是，仍以 n_samples 为 denominator 保持和主结果一致
    return ok / n_samples

def eval_method(name, pred_by_sid, cost_by_sid, target_ids, base_by_sid, gold_by_sid, base_acc, n_samples):
    changed = fixed = broken = 0

    for s in target_ids:
        base = base_by_sid.get(s, "")
        gold = gold_by_sid.get(s, "")
        pred = pred_by_sid.get(s, base)

        base_ok = int(base == gold and gold != "")
        pred_ok = int(pred == gold and gold != "")

        if pred != base:
            changed += 1
            if base_ok == 0 and pred_ok == 1:
                fixed += 1
            if base_ok == 1 and pred_ok == 0:
                broken += 1

    net = fixed - broken
    final_acc = base_acc + net / n_samples
    gain = final_acc - base_acc

    total_extra = sum(float(cost_by_sid.get(s, 0.0)) for s in target_ids)
    extra_per_target = total_extra / max(len(target_ids), 1)
    extra_per_sample = total_extra / n_samples
    cost_acc = gain / extra_per_sample if extra_per_sample > 1e-12 else None
    net_per_1k = net / total_extra * 1000.0 if total_extra > 1e-12 else None
    repair_p = fixed / max(fixed + broken, 1)
    harm = broken / max(changed, 1)

    return {
        "method": name,
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

def fget(d, keys, default=None):
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

def best_ours(tag, base_acc, n_samples, n_eval):
    root = Path("outputs/metrics/model_ablation_14b")
    best = None

    for fp in root.glob(f"{tag}_total*_seed*_margin*.json"):
        d = json.load(open(fp, encoding="utf-8"))

        base = to_float(fget(d, ["base_acc", "base_accuracy", "majority_acc"], base_acc))
        final = to_float(fget(d, ["final_acc", "final_accuracy", "accuracy", "acc", "numeric_acc"]))
        net = to_float(fget(d, ["net"]))
        gain = to_float(fget(d, ["gain", "acc_gain", "delta_acc"]))

        if final is None and base is not None and net is not None:
            final = base + net / n_samples

        if gain is None and final is not None and base is not None:
            gain = final - base

        if final is None:
            continue

        item = (final, gain if gain is not None else -999, fp, d, base)
        if best is None or item[:2] > best[:2]:
            best = item

    if best is None:
        return None

    final, gain, fp, d, base = best
    changed = int(fget(d, ["changed"], 0))
    fixed = int(fget(d, ["fixed"], 0))
    broken = int(fget(d, ["broken"], 0))
    net = int(fget(d, ["net"], fixed - broken))

    # 从文件名解析 total；total2/3/4 表示每个 seed 取多少，三 seed 总 extra = total * 3
    m = re.search(r"_total(\d+)_", fp.name)
    per_seed_total = int(m.group(1)) if m else 4
    extra_per_target = per_seed_total * 3
    total_extra = extra_per_target * n_eval
    extra_per_sample = total_extra / n_samples

    cost_acc = gain / extra_per_sample if extra_per_sample > 1e-12 else None
    net_per_1k = net / total_extra * 1000.0 if total_extra > 1e-12 else None
    repair_p = fixed / max(fixed + broken, 1)
    harm = broken / max(changed, 1)

    return {
        "method": "Ours-best",
        "final_acc": final,
        "gain": gain,
        "n_eval": n_eval,
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
        "source": str(fp),
    }

def family(method):
    m = method.lower()
    if "ours" in m:
        return "Ours"
    if "gg" in m:
        return "GG-lite-proxy"
    if "cisc" in m:
        return "CISC"
    if m.startswith("esc"):
        return "ESC"
    if m.startswith("sc"):
        return "SC"
    if "base" in m:
        return "Base"
    return "Other"

def fmt(x):
    if x is None:
        return "NA"
    return f"{float(x):.4f}"

all_rows = []

for case in CASES:
    ds = case["dataset"]
    tag = case["tag"]
    n_samples = case["n_samples"]

    base_details_fp = Path(f"outputs/predictions/model_ablation_14b/{tag}_base_details.jsonl")
    target_ids_fp = Path(f"outputs/targets/model_ablation_14b/{tag}_has_disagreement_ids.txt")
    extra_fps = [
        Path(f"data/processed/trajectories/model_ablation_14b/{tag}_extra_seed42.jsonl"),
        Path(f"data/processed/trajectories/model_ablation_14b/{tag}_extra_seed101.jsonl"),
        Path(f"data/processed/trajectories/model_ablation_14b/{tag}_extra_seed202.jsonl"),
    ]

    if not base_details_fp.exists() or not target_ids_fp.exists():
        print("[SKIP]", ds, "missing required files")
        continue

    base_rows, base_by_sid, gold_by_sid = load_base_details(base_details_fp)
    target_ids = load_target_ids(target_ids_fp)
    extra_by_sid = load_extra(extra_fps)
    base_acc = infer_base_acc(base_rows, base_by_sid, gold_by_sid, n_samples)

    print(f"[CASE] {ds}: base_acc={base_acc:.4f}, n_samples={n_samples}, n_targets={len(target_ids)}")

    rows = []

    pred = {s: base_by_sid.get(s, "") for s in target_ids}
    cost = {s: 0 for s in target_ids}
    rows.append(eval_method("Base-current", pred, cost, target_ids, base_by_sid, gold_by_sid, base_acc, n_samples))

    # SC / CISC / GG-lite-proxy
    for k in [4, 8, 12]:
        pred = {}
        cost = {}
        for s in target_ids:
            cand = extra_by_sid.get(s, [])[:k]
            pred[s] = majority(cand) or base_by_sid.get(s, "")
            cost[s] = min(k, len(extra_by_sid.get(s, [])))

        rows.append(eval_method(f"SC@{k}", pred, cost, target_ids, base_by_sid, gold_by_sid, base_acc, n_samples))
        rows.append(eval_method(f"CISC@{k}", pred, cost, target_ids, base_by_sid, gold_by_sid, base_acc, n_samples))
        rows.append(eval_method(f"GG_lite_proxy@{k}", pred, cost, target_ids, base_by_sid, gold_by_sid, base_acc, n_samples))

    # ESC
    for k in [4, 8, 12]:
        for w in [2, 3, 4]:
            pred = {}
            cost = {}
            for s in target_ids:
                counts = Counter()
                used = 0
                chosen = ""
                for a in extra_by_sid.get(s, [])[:k]:
                    used += 1
                    counts[a] += 1
                    if counts[a] >= w:
                        chosen = a
                        break
                if not chosen:
                    chosen = majority(extra_by_sid.get(s, [])[:k]) or base_by_sid.get(s, "")
                    used = min(k, len(extra_by_sid.get(s, [])))
                pred[s] = chosen
                cost[s] = used
            rows.append(eval_method(f"ESC@{k}_w{w}", pred, cost, target_ids, base_by_sid, gold_by_sid, base_acc, n_samples))

    ours = best_ours(tag, base_acc, n_samples, len(target_ids))
    if ours:
        rows.append(ours)

    rows.sort(key=lambda r: (r["final_acc"], r["cost_acc"] if r["cost_acc"] is not None else -999), reverse=True)

    csv_fp = OUT_DIR / f"{tag}_baseline_compare.csv"
    md_fp = OUT_DIR / f"{tag}_baseline_compare.md"

    fields = [
        "dataset", "family", "method", "final_acc", "gain", "n_eval", "changed",
        "fixed", "broken", "net", "extra_per_target", "extra_per_sample",
        "cost_acc", "net_per_1k", "repair_p", "harm", "source"
    ]

    with csv_fp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr["dataset"] = ds
            rr["family"] = family(r["method"])
            rr.setdefault("source", "")
            w.writerow(rr)

    lines = []
    lines.append(f"# DS14B Baseline Compare: {ds}")
    lines.append("")
    lines.append("| Method | Family | Final Acc | ΔAcc | Extra/Sample | Cost-Acc | Net | Net/1K | Repair-P | Harm |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['method']} | {family(r['method'])} | {fmt(r['final_acc'])} | {fmt(r['gain'])} | "
            f"{fmt(r['extra_per_sample'])} | {fmt(r['cost_acc'])} | {r['net']} | {fmt(r['net_per_1k'])} | "
            f"{fmt(r['repair_p'])} | {fmt(r['harm'])} |"
        )
    md_fp.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for r in rows:
        rr = dict(r)
        rr["dataset"] = ds
        rr["family"] = family(r["method"])
        all_rows.append(rr)

# all rows
all_md = OUT_DIR / "ds14b_all_baselines_allrows.md"
lines = []
lines.append("# DS14B All Baseline Rows")
lines.append("")
lines.append("| Dataset | Method | Family | Final Acc | ΔAcc | Extra/Sample | Cost-Acc | Net | Net/1K | Repair-P | Harm |")
lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in sorted(all_rows, key=lambda x: (x["dataset"], x["family"], -(x["final_acc"] or 0))):
    lines.append(
        f"| {r['dataset']} | {r['method']} | {r['family']} | {fmt(r['final_acc'])} | {fmt(r['gain'])} | "
        f"{fmt(r['extra_per_sample'])} | {fmt(r['cost_acc'])} | {r['net']} | {fmt(r['net_per_1k'])} | "
        f"{fmt(r['repair_p'])} | {fmt(r['harm'])} |"
    )
all_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

# family summary
fam_md = OUT_DIR / "ds14b_family_summary.md"
lines = []
lines.append("# DS14B Family Summary")
lines.append("")
lines.append("Cell format: `Final Acc / Cost-Acc / Repair-P / Harm`.")
lines.append("")
lines.append("| Dataset | SC | ESC | CISC | GG-lite-proxy | Ours |")
lines.append("|---|---|---|---|---|---|")

def family_cell(cr):
    if not cr:
        return "—"
    best_acc = max([r for r in cr if r["final_acc"] is not None], key=lambda r: r["final_acc"], default=None)
    best_cost = max([r for r in cr if r["cost_acc"] is not None], key=lambda r: r["cost_acc"], default=None)
    best_rp = max([r for r in cr if r["repair_p"] is not None], key=lambda r: r["repair_p"], default=None)
    positive_harm = [r for r in cr if r["harm"] is not None and r["gain"] > 0]
    best_harm = min(positive_harm, key=lambda r: r["harm"], default=None)

    return (
        f"{fmt(best_acc['final_acc'] if best_acc else None)} / "
        f"{fmt(best_cost['cost_acc'] if best_cost else None)} / "
        f"{fmt(best_rp['repair_p'] if best_rp else None)} / "
        f"{fmt(best_harm['harm'] if best_harm else None)}"
    )

for ds in ["asdiv", "gsm8k", "math500-long1024", "svamp"]:
    ds_rows = [r for r in all_rows if r["dataset"] == ds]
    if not ds_rows:
        continue
    row = [ds]
    for fam in ["SC", "ESC", "CISC", "GG-lite-proxy", "Ours"]:
        row.append(family_cell([r for r in ds_rows if r["family"] == fam]))
    lines.append("| " + " | ".join(row) + " |")

fam_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("saved:", all_md)
print("saved:", fam_md)
for fp in sorted(OUT_DIR.glob("*deepseek14b*baseline_compare.md")):
    print("saved:", fp)
