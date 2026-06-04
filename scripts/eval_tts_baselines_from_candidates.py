#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import re
from collections import defaultdict, Counter
from pathlib import Path


def read_jsonl(fp):
    fp = Path(fp)
    if not fp.exists():
        return []
    rows = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
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
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.replace("\\$", "")
    s = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", s)
    s = re.sub(r"boxed\{([^{}]+)\}", r"\1", s)
    s = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", s)
    s = s.strip()

    # 抽取最后一个数字
    nums = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", s)
    if nums:
        s = nums[-1]

    # 处理简单分数
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
    if task_type == "choice":
        return normalize_choice(x)
    return normalize_numeric(x)


def extract_text(r):
    for k in ["trajectory", "text", "reasoning", "output", "completion", "response"]:
        if k in r and r[k]:
            return str(r[k])
    return ""


def extract_answer(r, task_type):
    for k in [
        "answer", "final_answer", "pred_answer", "prediction",
        "majority_answer", "extracted_answer", "response_answer"
    ]:
        if k in r and r[k] is not None:
            a = norm_answer(r[k], task_type)
            if a:
                return a

    text = extract_text(r)
    if not text:
        return ""

    patterns = [
        r"Final Answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*is\s*([^\n]+)",
        r"Proposed answer\s*[:：]\s*([^\n]+)",
        r"Answer\s*[:：]\s*([^\n]+)",
        r"答案\s*[:：]\s*([^\n]+)",
    ]
    for p in patterns:
        m = re.findall(p, text, flags=re.I)
        if m:
            return norm_answer(m[-1], task_type)

    return norm_answer(text[-300:], task_type)


def majority_vote(ans_list):
    ans_list = [a for a in ans_list if a]
    if not ans_list:
        return ""
    c = Counter(ans_list)
    # 按出现次数降序；并列时按第一次出现顺序
    first_pos = {}
    for i, a in enumerate(ans_list):
        if a not in first_pos:
            first_pos[a] = i
    return sorted(c.items(), key=lambda kv: (-kv[1], first_pos[kv[0]]))[0][0]


def softmax(xs, temp):
    if not xs:
        return []
    temp = max(float(temp), 1e-8)
    m = max(xs)
    es = [math.exp((x - m) / temp) for x in xs]
    z = sum(es)
    if z <= 0:
        return [1.0 / len(xs)] * len(xs)
    return [e / z for e in es]


def extract_confidence(r):
    # 这些字段如果存在，优先用
    for k in [
        "p_true", "confidence", "self_confidence", "score",
        "avg_logprob", "mean_logprob", "length_norm_logprob",
        "response_prob", "sequence_prob"
    ]:
        if k in r and r[k] is not None:
            try:
                return float(r[k]), "field:" + k
            except Exception:
                pass

    # 如果有 token_logprobs，则取平均 logprob
    for k in ["token_logprobs", "logprobs"]:
        if k in r and isinstance(r[k], list) and len(r[k]) > 0:
            vals = []
            for v in r[k]:
                try:
                    vals.append(float(v))
                except Exception:
                    pass
            if vals:
                return sum(vals) / len(vals), "field:" + k

    return None, "none"


def novelty_score(text, seen_tokens):
    toks = re.findall(r"[A-Za-z0-9_]+", str(text).lower())
    if not toks:
        return 0.0
    new = sum(1 for t in toks if t not in seen_tokens)
    return new / max(len(toks), 1)


def support_confidences(cands):
    cnt = Counter(c["answer"] for c in cands if c["answer"])
    n = max(len(cands), 1)
    return [cnt.get(c["answer"], 0) / n for c in cands]


def cisc_weighted_vote(cands, temp=1.0, use_support_fallback=True):
    cands = [c for c in cands if c["answer"]]
    if not cands:
        return ""

    raw_scores = []
    all_have_conf = True
    for c in cands:
        if c["confidence"] is None:
            all_have_conf = False
            break
        raw_scores.append(float(c["confidence"]))

    if not all_have_conf:
        if use_support_fallback:
            raw_scores = support_confidences(cands)
        else:
            raw_scores = [1.0] * len(cands)

    weights = softmax(raw_scores, temp)

    vote = defaultdict(float)
    for c, w in zip(cands, weights):
        vote[c["answer"]] += w

    return sorted(vote.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def gg_lite_select(cands, lambda_c=1.0, lambda_n=0.5):
    cands = [c for c in cands if c["answer"]]
    if not cands:
        return ""

    # confidence fallback: answer support
    support_scores = support_confidences(cands)
    seen = set()
    scored = []

    for i, c in enumerate(cands):
        conf = c["confidence"]
        if conf is None:
            conf = support_scores[i]
        try:
            conf = float(conf)
        except Exception:
            conf = support_scores[i]

        nov = novelty_score(c["text"], seen)
        toks = re.findall(r"[A-Za-z0-9_]+", str(c["text"]).lower())
        seen.update(toks)

        score = float(lambda_c) * conf + float(lambda_n) * nov
        scored.append((score, i, c["answer"]))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][2]


def eval_prediction(base_info, pred_by_sid, used_cost_by_sid, base_acc, n_samples, task_type):
    changed = fixed = broken = kept = 0
    n_eval = len(base_info)
    used_total = 0

    for sid, b in base_info.items():
        base_pred = b["base_pred"]
        base_ok = b["base_ok"]
        gold = b["gold"]
        pred = pred_by_sid.get(sid, base_pred)

        if not pred:
            pred = base_pred

        pred_ok = int(pred == gold)
        used_total += used_cost_by_sid.get(sid, 0)

        if pred != base_pred:
            changed += 1
            if base_ok == 0 and pred_ok == 1:
                fixed += 1
            elif base_ok == 1 and pred_ok == 0:
                broken += 1
        else:
            kept += 1

    net = fixed - broken
    final_acc = float(base_acc) + net / float(n_samples)

    base_wrong_global = max(int(round((1.0 - float(base_acc)) * int(n_samples))), 1)
    precision = fixed / max(fixed + broken, 1)
    recall = fixed / base_wrong_global
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    harm_rate = broken / max(changed, 1)

    return {
        "final_acc": final_acc,
        "gain": final_acc - float(base_acc),
        "n_eval": n_eval,
        "changed": changed,
        "fixed": fixed,
        "broken": broken,
        "net": net,
        "extra_calls": used_total,
        "extra_per_target": used_total / max(n_eval, 1),
        "extra_per_sample": used_total / max(int(n_samples), 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "harm_rate": harm_rate,
    }


def build_base_info(baseline_details, target_ids, task_type):
    rows = read_jsonl(baseline_details)
    base_info = {}
    for r in rows:
        sid = get_sid(r)
        if sid is None:
            continue
        if target_ids is not None and sid not in target_ids:
            continue

        gold = None
        for k in ["gold_norm", "gold_answer", "answer", "label"]:
            if k in r and r[k] is not None:
                gold = norm_answer(r[k], task_type)
                break

        base_pred = None
        for k in ["majority_answer", "pred_answer", "prediction", "first_answer"]:
            if k in r and r[k] is not None:
                base_pred = norm_answer(r[k], task_type)
                break

        if base_pred is None:
            base_pred = extract_answer(r, task_type)

        if not gold:
            continue

        if "majority_ok" in r:
            try:
                base_ok = int(r["majority_ok"])
            except Exception:
                base_ok = int(base_pred == gold)
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


def build_candidates(extra_jsonls, target_ids, task_type):
    by_sid = defaultdict(list)
    for fp in extra_jsonls:
        for r in read_jsonl(fp):
            sid = get_sid(r)
            if sid is None:
                continue
            if target_ids is not None and sid not in target_ids:
                continue

            ans = extract_answer(r, task_type)
            conf, conf_src = extract_confidence(r)
            text = extract_text(r)

            by_sid[sid].append({
                "answer": ans,
                "confidence": conf,
                "confidence_source": conf_src,
                "text": text,
                "source": str(fp),
            })

    return by_sid


def write_outputs(rows, out_dir, prefix):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_fp = out_dir / f"{prefix}_baseline_compare.csv"
    md_fp = out_dir / f"{prefix}_baseline_compare.md"
    json_fp = out_dir / f"{prefix}_baseline_compare.json"

    fieldnames = [
        "method", "base_acc", "final_acc", "gain", "n_eval",
        "changed", "fixed", "broken", "net",
        "extra_calls", "extra_per_target", "extra_per_sample",
        "precision", "recall", "f1", "harm_rate"
    ]

    with csv_fp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    with json_fp.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    def fmt(x):
        try:
            return f"{float(x):.4f}"
        except Exception:
            return str(x)

    lines = []
    lines.append(f"# Baseline comparison: {prefix}")
    lines.append("")
    lines.append("| Method | Base Acc | Final Acc | ΔAcc | n_eval | Changed | Fixed | Broken | Net | Extra/Target | Extra/Sample | PRF-P | PRF-R | PRF-F1 | Harm/Changed |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda x: (-float(x["final_acc"]), float(x["extra_per_sample"]))):
        lines.append(
            f"| {r['method']} | {fmt(r['base_acc'])} | {fmt(r['final_acc'])} | {fmt(r['gain'])} | "
            f"{r['n_eval']} | {r['changed']} | {r['fixed']} | {r['broken']} | {r['net']} | "
            f"{fmt(r['extra_per_target'])} | {fmt(r['extra_per_sample'])} | "
            f"{fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} | {fmt(r['harm_rate'])} |"
        )

    md_fp.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[SAVE]", csv_fp)
    print("[SAVE]", md_fp)
    print("[SAVE]", json_fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_details", required=True)
    ap.add_argument("--extra_jsonls", nargs="+", required=True)
    ap.add_argument("--target_ids", required=True)
    ap.add_argument("--task_type", choices=["numeric", "choice"], required=True)
    ap.add_argument("--base_acc", type=float, required=True)
    ap.add_argument("--n_samples", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--max_candidates", nargs="+", type=int, default=[4, 8, 12])
    ap.add_argument("--esc_windows", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--cisc_temps", nargs="+", type=float, default=[0.2, 0.5, 1.0, 2.0])
    ap.add_argument("--gg_lambdas", nargs="+", default=["1,0", "1,0.5", "1,1"])
    ap.add_argument("--ours_json", default=None)
    ap.add_argument("--ours_name", default="ours_confirm")
    args = ap.parse_args()

    target_ids = load_ids(args.target_ids)
    base_info = build_base_info(args.baseline_details, target_ids, args.task_type)
    candidates = build_candidates(args.extra_jsonls, target_ids, args.task_type)

    if not base_info:
        raise RuntimeError("No base_info loaded. Check baseline_details / target_ids / task_type.")

    rows = []

    def add_row(method, pred_by_sid, cost_by_sid):
        r = eval_prediction(
            base_info=base_info,
            pred_by_sid=pred_by_sid,
            used_cost_by_sid=cost_by_sid,
            base_acc=args.base_acc,
            n_samples=args.n_samples,
            task_type=args.task_type,
        )
        r["method"] = method
        r["base_acc"] = args.base_acc
        rows.append(r)

    # SC@K: vanilla majority over first K candidates
    for K in args.max_candidates:
        pred = {}
        cost = {}
        for sid, b in base_info.items():
            cs = candidates.get(sid, [])[:K]
            pred[sid] = majority_vote([c["answer"] for c in cs]) if cs else b["base_pred"]
            cost[sid] = len(cs)
        add_row(f"SC@{K}", pred, cost)

    # ESC: window early-stop, max L = K
    for K in args.max_candidates:
        for w in args.esc_windows:
            if w > K:
                continue
            pred = {}
            cost = {}
            for sid, b in base_info.items():
                cs_all = candidates.get(sid, [])[:K]
                observed = []
                used = 0
                for i in range(0, len(cs_all), w):
                    window = cs_all[i:i+w]
                    if not window:
                        break
                    observed.extend(window)
                    used += len(window)

                    ans_window = [c["answer"] for c in window if c["answer"]]
                    if len(ans_window) == w and len(set(ans_window)) == 1:
                        break

                pred[sid] = majority_vote([c["answer"] for c in observed]) if observed else b["base_pred"]
                cost[sid] = used
            add_row(f"ESC@{K}_w{w}", pred, cost)

    # CISC-like weighted vote
    for K in args.max_candidates:
        for temp in args.cisc_temps:
            pred = {}
            cost = {}
            for sid, b in base_info.items():
                cs = candidates.get(sid, [])[:K]
                pred[sid] = cisc_weighted_vote(cs, temp=temp, use_support_fallback=True) if cs else b["base_pred"]
                cost[sid] = len(cs)
            add_row(f"CISC_support_proxy@{K}_T{temp}", pred, cost)

    # GG-lite candidate selector
    for K in args.max_candidates:
        for pair in args.gg_lambdas:
            try:
                lc, ln = [float(x) for x in pair.split(",")]
            except Exception:
                continue

            pred = {}
            cost = {}
            for sid, b in base_info.items():
                cs = candidates.get(sid, [])[:K]
                pred[sid] = gg_lite_select(cs, lambda_c=lc, lambda_n=ln) if cs else b["base_pred"]
                cost[sid] = len(cs)
            add_row(f"GG_lite@{K}_lc{lc}_ln{ln}", pred, cost)

    # Our existing method row, if provided
    if args.ours_json and Path(args.ours_json).exists():
        d = json.load(open(args.ours_json, encoding="utf-8"))
        final = d.get("estimated_global_acc", d.get("final_acc", d.get("acc")))
        if final is not None:
            fixed = int(d.get("fixed", 0))
            broken = int(d.get("broken", 0))
            changed = int(d.get("changed", 0))
            net = d.get("net", fixed - broken)
            n_eval = int(d.get("n_eval", len(base_info)))
            extra_per_target = 12.0
            extra_calls = int(round(extra_per_target * n_eval))
            extra_per_sample = extra_calls / max(args.n_samples, 1)
            base_wrong_global = max(int(round((1.0 - args.base_acc) * args.n_samples)), 1)
            precision = fixed / max(fixed + broken, 1)
            recall = fixed / base_wrong_global
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            harm_rate = broken / max(changed, 1)

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
                "extra_calls": extra_calls,
                "extra_per_target": extra_per_target,
                "extra_per_sample": extra_per_sample,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "harm_rate": harm_rate,
            })

    write_outputs(rows, args.out_dir, args.prefix)


if __name__ == "__main__":
    main()
