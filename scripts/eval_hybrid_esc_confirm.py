#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict, Counter


def read_jsonl(fp):
    rows = []
    fp = Path(fp)
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
    s = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", s)
    s = re.sub(r"boxed\{([^{}]+)\}", r"\1", s)
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


def extract_text(r):
    for k in ["trajectory", "text", "reasoning", "output", "completion", "response"]:
        if k in r and r[k]:
            return str(r[k])
    return ""


def extract_answer(r, task_type):
    for k in ["answer", "final_answer", "pred_answer", "prediction", "majority_answer", "extracted_answer"]:
        if k in r and r[k] is not None:
            a = norm_answer(r[k], task_type)
            if a:
                return a

    text = extract_text(r)
    patterns = [
        r"Final Answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*is\s*([^\n]+)",
        r"Answer\s*[:：]\s*([^\n]+)",
        r"答案\s*[:：]\s*([^\n]+)",
    ]
    for p in patterns:
        m = re.findall(p, text, flags=re.I)
        if m:
            return norm_answer(m[-1], task_type)

    return norm_answer(text[-300:], task_type)


def build_base_info(baseline_details, target_ids, task_type):
    base = {}
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

        base[sid] = {
            "gold": gold,
            "base_pred": base_pred,
            "base_ok": base_ok,
        }
    return base


def build_candidates(extra_jsonls, target_ids, task_type):
    by_sid = defaultdict(list)
    for file_idx, fp in enumerate(extra_jsonls):
        for r in read_jsonl(fp):
            sid = get_sid(r)
            if sid is None:
                continue
            if target_ids is not None and sid not in target_ids:
                continue

            ans = extract_answer(r, task_type)
            by_sid[sid].append({
                "answer": ans,
                "source_file": str(fp),
                "seed_index": file_idx,
            })
    return by_sid


def majority(ans):
    ans = [a for a in ans if a]
    if not ans:
        return ""
    cnt = Counter(ans)
    first = {}
    for i, a in enumerate(ans):
        if a not in first:
            first[a] = i
    return sorted(cnt.items(), key=lambda kv: (-kv[1], first[kv[0]]))[0][0]


def confirm_decision(cands, base_pred, k, min_total, min_seed, min_margin):
    used = cands[:k]
    answers = [c["answer"] for c in used if c["answer"]]
    if not answers:
        return base_pred

    cnt = Counter(answers)
    ranked = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    best_ans, best_count = ranked[0]
    second_count = ranked[1][1] if len(ranked) > 1 else 0
    margin = best_count - second_count

    seed_set = set()
    for c in used:
        if c["answer"] == best_ans:
            seed_set.add(c["seed_index"])

    if best_ans != base_pred and best_count >= min_total and len(seed_set) >= min_seed and margin >= min_margin:
        return best_ans
    return base_pred


def eval_method(base_info, pred_by_sid, cost_by_sid, base_acc, n_samples):
    changed = fixed = broken = 0
    total_cost = 0

    for sid, b in base_info.items():
        base_pred = b["base_pred"]
        gold = b["gold"]
        base_ok = int(b["base_ok"])
        pred = pred_by_sid.get(sid, base_pred) or base_pred

        pred_ok = int(pred == gold)
        total_cost += cost_by_sid.get(sid, 0)

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
        "extra_calls": total_cost,
        "extra_per_target": total_cost / max(n_eval, 1),
        "extra_per_sample": total_cost / max(n_samples, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "harm_rate": harm,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_details", required=True)
    ap.add_argument("--extra_jsonls", nargs="+", required=True)
    ap.add_argument("--target_ids", required=True)
    ap.add_argument("--task_type", choices=["numeric", "choice"], default="numeric")
    ap.add_argument("--base_acc", type=float, required=True)
    ap.add_argument("--n_samples", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--esc_w", type=int, default=3)
    ap.add_argument("--min_total_support", type=int, default=2)
    ap.add_argument("--min_seed_support", type=int, default=1)
    ap.add_argument("--min_margin", type=int, default=0)
    args = ap.parse_args()

    target_ids = load_ids(args.target_ids)
    base_info = build_base_info(args.baseline_details, target_ids, args.task_type)
    cands = build_candidates(args.extra_jsonls, target_ids, args.task_type)

    rows = []

    pred = {}
    cost = {}
    action_counter = Counter()

    for sid, b in base_info.items():
        cs = cands.get(sid, [])[:args.k]
        base_pred = b["base_pred"]

        # Stage 1: ESC early accept
        win = cs[:args.esc_w]
        win_ans = [c["answer"] for c in win if c["answer"]]

        if len(win_ans) == args.esc_w and len(set(win_ans)) == 1:
            pred[sid] = win_ans[0]
            cost[sid] = args.esc_w
            action_counter["esc_accept"] += 1
        else:
            pred[sid] = confirm_decision(
                cs,
                base_pred,
                k=args.k,
                min_total=args.min_total_support,
                min_seed=args.min_seed_support,
                min_margin=args.min_margin,
            )
            cost[sid] = len(cs)
            if pred[sid] != base_pred:
                action_counter["confirm_replace"] += 1
            else:
                action_counter["confirm_keep"] += 1

    r = eval_method(base_info, pred, cost, args.base_acc, args.n_samples)
    r["method"] = f"ESCConfirm_w{args.esc_w}_k{args.k}_total{args.min_total_support}_seed{args.min_seed_support}_margin{args.min_margin}"
    r["base_acc"] = args.base_acc
    r["action_counter"] = dict(action_counter)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"{args.prefix}_esc_confirm.json"
    out_md = out_dir / f"{args.prefix}_esc_confirm.md"

    out_json.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")

    def fmt(x):
        try:
            return f"{float(x):.4f}"
        except Exception:
            return str(x)

    lines = []
    lines.append(f"# ESC-Confirm Hybrid: {args.prefix}")
    lines.append("")
    lines.append("| Method | Base Acc | Final Acc | ΔAcc | n_eval | Changed | Fixed | Broken | Net | Extra/Target | Extra/Sample | PRF-P | PRF-R | F1 | Harm | Actions |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    lines.append(
        f"| {r['method']} | {fmt(r['base_acc'])} | {fmt(r['final_acc'])} | {fmt(r['gain'])} | "
        f"{r['n_eval']} | {r['changed']} | {r['fixed']} | {r['broken']} | {r['net']} | "
        f"{fmt(r['extra_per_target'])} | {fmt(r['extra_per_sample'])} | "
        f"{fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} | {fmt(r['harm_rate'])} | "
        f"`{dict(action_counter)}` |"
    )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[SAVE]", out_json)
    print("[SAVE]", out_md)
    print(out_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
