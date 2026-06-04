#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import re
from pathlib import Path
from collections import defaultdict, Counter


def read_jsonl(fp):
    fp = Path(fp)
    rows = []
    if not fp.exists():
        return rows
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_ids(fp):
    fp = Path(fp)
    if not fp.exists():
        return None
    return set(x.strip() for x in fp.read_text(encoding="utf-8").splitlines() if x.strip())


def get_sid(r):
    for k in ["sample_id", "id", "qid", "question_id"]:
        if k in r and r[k] is not None:
            return str(r[k])
    return None


def normalize_numeric(x):
    if x is None:
        return ""
    s = str(x).strip()
    s = s.replace(",", "").replace("$", "").replace("\\$", "")
    nums = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", s)
    if nums:
        s = nums[-1]
    if "/" in s:
        try:
            a, b = s.split("/", 1)
            return str(round(float(a) / float(b), 10)).rstrip("0").rstrip(".")
        except Exception:
            pass
    try:
        v = float(s)
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.10f}".rstrip("0").rstrip(".")
    except Exception:
        return re.sub(r"\s+", "", s.lower())


def normalize_choice(x):
    if x is None:
        return ""
    s = str(x).strip().lower()
    m = re.search(r"\b([abcde])\b", s)
    if m:
        return m.group(1)
    m = re.search(r"\(([abcde])\)", s)
    if m:
        return m.group(1)
    return s[:1]


def norm_answer(x, task_type):
    return normalize_choice(x) if task_type == "choice" else normalize_numeric(x)


def build_base_info(baseline_details, target_ids, task_type):
    base_info = {}

    for r in read_jsonl(baseline_details):
        sid = get_sid(r)
        if sid is None:
            continue
        if target_ids is not None and sid not in target_ids:
            continue

        gold = ""
        for k in ["gold_norm", "gold_answer", "answer", "label"]:
            if k in r and r[k] is not None:
                gold = norm_answer(r[k], task_type)
                break

        base_pred = ""
        for k in ["majority_answer", "pred_answer", "prediction", "first_answer"]:
            if k in r and r[k] is not None:
                base_pred = norm_answer(r[k], task_type)
                break

        if not gold:
            continue

        if "majority_ok" in r:
            base_ok = int(r["majority_ok"])
        elif "base_ok" in r:
            base_ok = int(r["base_ok"])
        else:
            base_ok = int(base_pred == gold)

        base_info[sid] = {
            "gold": gold,
            "base_pred": base_pred,
            "base_ok": base_ok,
        }

    return base_info


def softmax(xs, temp):
    temp = max(float(temp), 1e-8)
    if not xs:
        return []
    m = max(xs)
    es = [math.exp((x - m) / temp) for x in xs]
    z = sum(es)
    if z <= 0:
        return [1.0 / len(xs)] * len(xs)
    return [e / z for e in es]


def weighted_vote(cands, temp):
    cands = [c for c in cands if c["answer"]]
    if not cands:
        return ""

    scores = [float(c.get("p_true", 0.5)) for c in cands]
    weights = softmax(scores, temp)

    vote = defaultdict(float)
    for c, w in zip(cands, weights):
        vote[c["answer"]] += w

    return sorted(vote.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def top1_vote(cands):
    cands = [c for c in cands if c["answer"]]
    if not cands:
        return ""
    cands = sorted(cands, key=lambda x: (-float(x.get("p_true", 0.5)), int(x.get("candidate_index", 0))))
    return cands[0]["answer"]


def eval_pred(base_info, pred_by_sid, gen_cost_by_sid, score_cost_by_sid, base_acc, n_samples):
    changed = fixed = broken = 0
    gen_calls = 0
    score_calls = 0

    for sid, b in base_info.items():
        base_pred = b["base_pred"]
        base_ok = int(b["base_ok"])
        gold = b["gold"]
        pred = pred_by_sid.get(sid, base_pred) or base_pred

        pred_ok = int(pred == gold)
        gen_calls += gen_cost_by_sid.get(sid, 0)
        score_calls += score_cost_by_sid.get(sid, 0)

        if pred != base_pred:
            changed += 1
            if base_ok == 0 and pred_ok == 1:
                fixed += 1
            elif base_ok == 1 and pred_ok == 0:
                broken += 1

    net = fixed - broken
    final_acc = float(base_acc) + net / float(n_samples)

    base_wrong = max(int(round((1.0 - float(base_acc)) * int(n_samples))), 1)
    precision = fixed / max(fixed + broken, 1)
    recall = fixed / base_wrong
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    harm = broken / max(changed, 1)

    n_eval = len(base_info)
    return {
        "final_acc": final_acc,
        "gain": final_acc - float(base_acc),
        "n_eval": n_eval,
        "changed": changed,
        "fixed": fixed,
        "broken": broken,
        "net": net,
        "gen_calls": gen_calls,
        "score_calls": score_calls,
        "total_calls": gen_calls + score_calls,
        "gen_per_target": gen_calls / max(n_eval, 1),
        "score_per_target": score_calls / max(n_eval, 1),
        "total_per_target": (gen_calls + score_calls) / max(n_eval, 1),
        "total_per_sample": (gen_calls + score_calls) / max(n_samples, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "harm_rate": harm,
    }


def load_scores(fp, target_ids, task_type):
    by_sid = defaultdict(list)
    for r in read_jsonl(fp):
        sid = str(r.get("sample_id"))
        if target_ids is not None and sid not in target_ids:
            continue
        ans = norm_answer(r.get("answer"), task_type)
        if not ans:
            continue
        by_sid[sid].append({
            "answer": ans,
            "p_true": float(r.get("p_true", 0.5)),
            "candidate_index": int(r.get("candidate_index", len(by_sid[sid]))),
        })

    for sid in by_sid:
        by_sid[sid].sort(key=lambda x: x["candidate_index"])
    return by_sid


def write_outputs(rows, out_dir, prefix):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_fp = out_dir / f"{prefix}_ptrue_cisc_compare.csv"
    md_fp = out_dir / f"{prefix}_ptrue_cisc_compare.md"
    json_fp = out_dir / f"{prefix}_ptrue_cisc_compare.json"

    keys = [
        "method", "base_acc", "final_acc", "gain", "n_eval",
        "changed", "fixed", "broken", "net",
        "gen_per_target", "score_per_target", "total_per_target", "total_per_sample",
        "precision", "recall", "f1", "harm_rate"
    ]

    with csv_fp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    json_fp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    def fmt(x):
        try:
            return f"{float(x):.4f}"
        except Exception:
            return str(x)

    lines = []
    lines.append(f"# P(True)-CISC comparison: {prefix}")
    lines.append("")
    lines.append("| Method | Base Acc | Final Acc | ΔAcc | Net | Gen/Target | Score/Target | Total/Target | Total/Sample | PRF-P | PRF-R | F1 | Harm |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in sorted(rows, key=lambda x: (-float(x["final_acc"]), float(x["total_per_sample"]))):
        lines.append(
            f"| {r['method']} | {fmt(r['base_acc'])} | {fmt(r['final_acc'])} | {fmt(r['gain'])} | "
            f"{r['net']} | {fmt(r['gen_per_target'])} | {fmt(r['score_per_target'])} | "
            f"{fmt(r['total_per_target'])} | {fmt(r['total_per_sample'])} | "
            f"{fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} | {fmt(r['harm_rate'])} |"
        )

    md_fp.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[SAVE]", csv_fp)
    print("[SAVE]", md_fp)
    print("[SAVE]", json_fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_details", required=True)
    ap.add_argument("--ptrue_jsonl", required=True)
    ap.add_argument("--target_ids", required=True)
    ap.add_argument("--task_type", choices=["numeric", "choice"], default="numeric")
    ap.add_argument("--base_acc", type=float, required=True)
    ap.add_argument("--n_samples", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[4, 8, 12])
    ap.add_argument("--temps", nargs="+", type=float, default=[0.2, 0.5, 1.0, 2.0])
    ap.add_argument("--ours_json", default=None)
    ap.add_argument("--ours_name", default="Confirm")
    args = ap.parse_args()

    target_ids = load_ids(args.target_ids)
    base_info = build_base_info(args.baseline_details, target_ids, args.task_type)
    by_sid = load_scores(args.ptrue_jsonl, target_ids, args.task_type)

    rows = []

    def add(method, pred_by_sid, k_by_sid):
        gen_cost = {sid: k_by_sid.get(sid, 0) for sid in base_info}
        score_cost = {sid: k_by_sid.get(sid, 0) for sid in base_info}
        r = eval_pred(base_info, pred_by_sid, gen_cost, score_cost, args.base_acc, args.n_samples)
        r["method"] = method
        r["base_acc"] = args.base_acc
        rows.append(r)

    for K in args.ks:
        for T in args.temps:
            pred = {}
            k_used = {}
            for sid, b in base_info.items():
                cs = by_sid.get(sid, [])[:K]
                pred[sid] = weighted_vote(cs, T) if cs else b["base_pred"]
                k_used[sid] = len(cs)
            add(f"CISC_PTrue@{K}_T{T}", pred, k_used)

        pred = {}
        k_used = {}
        for sid, b in base_info.items():
            cs = by_sid.get(sid, [])[:K]
            pred[sid] = top1_vote(cs) if cs else b["base_pred"]
            k_used[sid] = len(cs)
        add(f"PTrue_top1@{K}", pred, k_used)

    if args.ours_json and Path(args.ours_json).exists():
        d = json.load(open(args.ours_json, encoding="utf-8"))
        final = d.get("estimated_global_acc", d.get("final_acc", d.get("acc")))
        fixed = int(d.get("fixed", 0))
        broken = int(d.get("broken", 0))
        changed = int(d.get("changed", 0))
        net = int(d.get("net", fixed - broken))
        n_eval = int(d.get("n_eval", len(base_info)))

        base_wrong = max(int(round((1.0 - float(args.base_acc)) * int(args.n_samples))), 1)
        precision = fixed / max(fixed + broken, 1)
        recall = fixed / base_wrong
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        harm = broken / max(changed, 1)

        rows.append({
            "method": args.ours_name,
            "base_acc": args.base_acc,
            "final_acc": float(final),
            "gain": float(final) - args.base_acc,
            "n_eval": n_eval,
            "changed": changed,
            "fixed": fixed,
            "broken": broken,
            "net": net,
            "gen_calls": 12 * n_eval,
            "score_calls": 0,
            "total_calls": 12 * n_eval,
            "gen_per_target": 12.0,
            "score_per_target": 0.0,
            "total_per_target": 12.0,
            "total_per_sample": (12 * n_eval) / max(args.n_samples, 1),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "harm_rate": harm,
        })

    write_outputs(rows, args.out_dir, args.prefix)


if __name__ == "__main__":
    main()
