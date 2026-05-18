import argparse
import json
import re
from pathlib import Path
from collections import Counter


def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]


def write_json(fp, obj):
    p = Path(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(fp, rows):
    p = Path(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def clean(x):
    x = str(x or "").strip().replace(",", "").replace("$", "")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", x)
    if not nums:
        return x
    y = nums[-1]
    if "." in y:
        y = y.rstrip("0").rstrip(".")
    return y


def ok(a, g):
    return clean(a) == clean(g)


def support_dict(r):
    return {clean(k): int(v) for k, v in r.get("seedaware_total_support", {}).items()}


def top_support(r):
    sup = support_dict(r)
    if not sup:
        return "", 0
    return sorted(sup.items(), key=lambda x: x[1], reverse=True)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--orig_majority_threshold", type=int, default=2)
    ap.add_argument("--top_total_threshold", type=int, default=7)
    ap.add_argument("--base_acc", type=float, required=True)
    ap.add_argument("--n_samples", type=int, required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.input_jsonl)

    fixed = broken = changed = correct = 0
    out = []

    for r in rows:
        gold = r["gold_answer"]
        cur = clean(r.get("current_best_answer", ""))

        base_pred = clean(
            r.get(
                "final_guard_answer",
                r.get("seedaware_answer", r.get("current_best_answer", ""))
            )
        )

        pred = base_pred
        reason = r.get("final_guard_reason", r.get("seedaware_reason", "base"))

        orig = [clean(x) for x in r.get("orig_answers", []) if clean(x) != ""]
        orig_cnt = Counter(orig)

        top_ans, top_cnt = top_support(r)

        if orig_cnt.get(cur, 0) >= args.orig_majority_threshold and top_cnt <= args.top_total_threshold:
            pred = cur
            reason = f"selective_orig_majority_keep_top_total<={args.top_total_threshold}"

        cur_ok = ok(cur, gold)
        pred_ok = ok(pred, gold)

        fixed_i = int((not cur_ok) and pred_ok)
        broken_i = int(cur_ok and (not pred_ok))
        changed_i = int(pred != cur)

        fixed += fixed_i
        broken += broken_i
        changed += changed_i
        correct += int(pred_ok)

        rr = dict(r)
        rr["selective_orig_majority_answer"] = pred
        rr["selective_orig_majority_reason"] = reason
        rr["selective_orig_majority_ok"] = int(pred_ok)
        rr["selective_orig_majority_fixed"] = fixed_i
        rr["selective_orig_majority_broken"] = broken_i
        rr["selective_orig_majority_changed"] = changed_i
        rr["orig_majority_count_current"] = orig_cnt.get(cur, 0)
        rr["top_support_answer"] = top_ans
        rr["top_support_total"] = top_cnt
        out.append(rr)

    net = fixed - broken

    summary = {
        "rule": f"selective_orig_majority_current>={args.orig_majority_threshold}_keep_if_top_total<={args.top_total_threshold}",
        "input_jsonl": args.input_jsonl,
        "n_resampled": len(rows),
        "base_acc": args.base_acc,
        "n_samples": args.n_samples,
        "fixed": fixed,
        "broken": broken,
        "net": net,
        "changed": changed,
        "acc_on_resampled": correct / max(1, len(rows)),
        "estimated_global_acc": args.base_acc + net / args.n_samples,
        "estimated_global_gain": net / args.n_samples,
        "out_jsonl": args.out_jsonl,
    }

    write_json(args.out_json, summary)
    write_jsonl(args.out_jsonl, out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
