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
    x = str(x or "").strip()
    x = x.replace(",", "").replace("$", "")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", x)
    if not nums:
        return x
    y = nums[-1]
    if "." in y:
        y = y.rstrip("0").rstrip(".")
    return y


def is_num(x):
    return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", clean(x) or ""))


def ok(ans, gold):
    return clean(ans) == clean(gold)


def choose_extra_any_orig_support(r):
    """
    Non-destructive confirmation rule:
    If any extra answer matches one original candidate answer,
    use that matched answer as resampling-confirmed answer.
    Otherwise keep current_best.
    """
    cur = clean(r.get("current_best_answer", ""))
    orig = [clean(x) for x in r.get("orig_answers", []) if is_num(x)]
    extra = [clean(x) for x in r.get("extra_answers", []) if is_num(x)]

    orig_set = set(orig)

    for x in extra:
        if x in orig_set:
            return x, "extra_any_orig_support"

    return cur, "keep_current"


def choose_extra_consensus_orig_support(r):
    cur = clean(r.get("current_best_answer", ""))
    orig = {clean(x) for x in r.get("orig_answers", []) if is_num(x)}
    extra = [clean(x) for x in r.get("extra_answers", []) if is_num(x)]

    if len(extra) >= 2 and len(set(extra)) == 1 and extra[0] in orig:
        return extra[0], "extra_consensus_orig_support"

    return cur, "keep_current"



def choose_guard_orig_majority_if_current_in_orig(r):
    """
    Guarded confirmation rule:
    first apply extra_any_orig_support, but if the extra-supported answer
    is the original majority answer and the current answer is also among
    original candidates, keep current to avoid being pulled by a repeated
    wrong candidate.
    """
    cur = clean(r.get("current_best_answer", ""))
    orig = [clean(x) for x in r.get("orig_answers", []) if is_num(x)]
    extra = [clean(x) for x in r.get("extra_answers", []) if is_num(x)]

    orig_set = set(orig)
    x = None
    for e in extra:
        if e in orig_set:
            x = e
            break

    if x is None:
        return cur, "keep_current"

    cnt = Counter(orig)
    if cnt:
        max_freq = max(cnt.values())
        if cur in cnt and cnt.get(x, 0) == max_freq and max_freq >= 2:
            return cur, "guard_keep_current_orig_majority"

    return x, "guard_extra_any_orig_support"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resample_jsonl", required=True)
    ap.add_argument("--rule", default="extra_any_orig_support",
                    choices=["extra_any_orig_support", "extra_consensus_orig_support", "guard_orig_majority_if_current_in_orig"])
    ap.add_argument("--base_acc", type=float, default=0.910538286580743)
    ap.add_argument("--n_samples", type=int, default=1319)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.resample_jsonl)

    fixed = 0
    broken = 0
    changed = 0
    correct = 0
    out_rows = []

    if args.rule == "extra_any_orig_support":
        chooser = choose_extra_any_orig_support
    elif args.rule == "extra_consensus_orig_support":
        chooser = choose_extra_consensus_orig_support
    elif args.rule == "guard_orig_majority_if_current_in_orig":
        chooser = choose_guard_orig_majority_if_current_in_orig
    else:
        raise ValueError(f"Unknown rule: {args.rule}")

    for r in rows:
        gold = r["gold_answer"]
        cur = clean(r.get("current_best_answer", ""))
        pred, reason = chooser(r)

        cur_ok = ok(cur, gold)
        pred_ok = ok(pred, gold)

        fixed += int((not cur_ok) and pred_ok)
        broken += int(cur_ok and (not pred_ok))
        changed += int(clean(pred) != clean(cur))
        correct += int(pred_ok)

        rr = dict(r)
        rr["confirm_rule"] = args.rule
        rr["confirm_reason"] = reason
        rr["confirm_answer"] = pred
        rr["confirm_ok"] = int(pred_ok)
        rr["confirm_fixed"] = int((not cur_ok) and pred_ok)
        rr["confirm_broken"] = int(cur_ok and (not pred_ok))
        rr["confirm_changed"] = int(clean(pred) != clean(cur))
        out_rows.append(rr)

    net = fixed - broken
    estimated_acc = args.base_acc + net / args.n_samples

    summary = {
        "rule": args.rule,
        "n_resampled": len(rows),
        "base_acc": args.base_acc,
        "n_samples": args.n_samples,
        "fixed": fixed,
        "broken": broken,
        "net": net,
        "changed": changed,
        "acc_on_resampled": correct / max(1, len(rows)),
        "estimated_global_acc": estimated_acc,
        "estimated_global_gain": net / args.n_samples,
        "out_jsonl": args.out_jsonl,
    }

    write_json(args.out_json, summary)
    write_jsonl(args.out_jsonl, out_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
