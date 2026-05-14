import argparse
import json
import re
from pathlib import Path
from collections import Counter


def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]


def read_ids(fp):
    p = Path(fp)
    if not p.exists():
        return set()
    return {x.strip() for x in p.open("r", encoding="utf-8") if x.strip()}


def clean(x):
    x = str(x or "").strip().replace(",", "").replace("$", "")
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


def extra_supported_answer(r):
    orig = {clean(x) for x in r.get("orig_answers", []) if is_num(x)}
    for x in r.get("extra_answers", []):
        x = clean(x)
        if is_num(x) and x in orig:
            return x
    return None


def extra_consensus_answer(r):
    xs = [clean(x) for x in r.get("extra_answers", []) if is_num(x)]
    if len(xs) >= 2 and len(set(xs)) == 1:
        return xs[0]
    return None


def guarded_margin_rule(r):
    cur = clean(r.get("current_best_answer", ""))
    x = extra_supported_answer(r)
    if x is None:
        return cur, "keep_current"

    orig = [clean(a) for a in r.get("orig_answers", []) if is_num(a)]
    cnt = Counter(orig)
    if cnt:
        max_freq = max(cnt.values())
        if cur in cnt and cnt.get(x, 0) == max_freq and max_freq >= 2:
            return cur, "guard_keep_current_orig_majority"

    return x, "guard_extra_any_orig_support"


def eval_rule(name, rows, margin_ids, fn):
    fixed = broken = changed = correct = 0
    fixed_ids, broken_ids = [], []

    for r in rows:
        gold = r["gold_answer"]
        cur = clean(r.get("current_best_answer", ""))
        pred, reason = fn(r, r["sample_id"] in margin_ids)

        cur_ok = ok(cur, gold)
        pred_ok = ok(pred, gold)

        fixed += int((not cur_ok) and pred_ok)
        broken += int(cur_ok and (not pred_ok))
        changed += int(clean(pred) != cur)
        correct += int(pred_ok)

        if (not cur_ok) and pred_ok:
            fixed_ids.append(r["sample_id"])
        if cur_ok and (not pred_ok):
            broken_ids.append(r["sample_id"])

    print(f"| {name} | {len(rows)} | {correct/len(rows):.4f} | {fixed} | {broken} | {fixed-broken} | {changed} | {fixed_ids} | {broken_ids} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resample_jsonl", required=True)
    ap.add_argument("--margin_ids", default="outputs/targets/gsm8k_full_margin030_107_sample_ids.txt")
    args = ap.parse_args()

    rows = read_jsonl(args.resample_jsonl)
    margin_ids = read_ids(args.margin_ids)

    print("file:", args.resample_jsonl)
    print("rows:", len(rows))
    print("margin_ids:", len(margin_ids))
    print("| Rule | n | acc_on_rows | fixed | broken | net | changed | fixed_ids | broken_ids |")
    print("|---|---:|---:|---:|---:|---:|---:|---|---|")

    # 规则0：当前 guarded rule
    def rule_guarded(r, in_margin):
        return guarded_margin_rule(r)

    # 规则1：只允许 margin030 内改；outside_margin 一律不改
    def rule_margin_only(r, in_margin):
        if not in_margin:
            return clean(r.get("current_best_answer", "")), "outside_margin_keep"
        return guarded_margin_rule(r)

    # 规则2：margin 内用 guarded；outside 只有 extra 两条一致且支持原始候选才改
    def rule_margin_guard_outside_consensus(r, in_margin):
        if in_margin:
            return guarded_margin_rule(r)

        cur = clean(r.get("current_best_answer", ""))
        x = extra_supported_answer(r)
        c = extra_consensus_answer(r)
        if x is not None and c is not None and x == c:
            return x, "outside_extra_consensus_orig_support"
        return cur, "outside_keep_current"

    # 规则3：margin 内用 guarded；outside 需要 extra 两条一致，而且 current 不是原始多数
    def rule_margin_guard_outside_consensus_not_current_majority(r, in_margin):
        if in_margin:
            return guarded_margin_rule(r)

        cur = clean(r.get("current_best_answer", ""))
        x = extra_supported_answer(r)
        c = extra_consensus_answer(r)
        if x is None or c is None or x != c:
            return cur, "outside_keep_current"

        orig = [clean(a) for a in r.get("orig_answers", []) if is_num(a)]
        cnt = Counter(orig)
        if cnt:
            max_freq = max(cnt.values())
            if cur in cnt and cnt[cur] == max_freq and max_freq >= 2:
                return cur, "outside_guard_current_orig_majority"

        return x, "outside_extra_consensus_orig_support"

    # 规则4：outside 只在 current 不在原始候选时允许改
    def rule_margin_guard_outside_current_not_in_orig(r, in_margin):
        if in_margin:
            return guarded_margin_rule(r)

        cur = clean(r.get("current_best_answer", ""))
        orig = {clean(a) for a in r.get("orig_answers", []) if is_num(a)}
        if cur in orig:
            return cur, "outside_current_in_orig_keep"

        x = extra_supported_answer(r)
        if x is not None:
            return x, "outside_current_not_in_orig_extra_support"
        return cur, "outside_keep_current"

    rules = [
        ("guarded_all", rule_guarded),
        ("margin_only", rule_margin_only),
        ("margin_guard_outside_consensus", rule_margin_guard_outside_consensus),
        ("margin_guard_outside_consensus_not_current_majority", rule_margin_guard_outside_consensus_not_current_majority),
        ("margin_guard_outside_current_not_in_orig", rule_margin_guard_outside_current_not_in_orig),
    ]

    for name, fn in rules:
        eval_rule(name, rows, margin_ids, fn)


if __name__ == "__main__":
    main()
