import argparse
import json
import re
from pathlib import Path
from collections import Counter, defaultdict


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


def pick_extra_supported(r):
    orig = {clean(x) for x in r.get("orig_answers", []) if is_num(x)}
    for x in r.get("extra_answers", []):
        x = clean(x)
        if is_num(x) and x in orig:
            return x
    return None


def guarded_pred(r):
    cur = clean(r.get("current_best_answer", ""))
    x = pick_extra_supported(r)
    if x is None:
        return cur, "keep_current"

    orig = [clean(a) for a in r.get("orig_answers", []) if is_num(a)]
    cnt = Counter(orig)
    if cnt:
        max_freq = max(cnt.values())
        if cur in cnt and cnt.get(x, 0) == max_freq and max_freq >= 2:
            return cur, "guard_keep_current_orig_majority"

    return x, "guard_extra_any_orig_support"


def row_features(r, margin_ids, all_ids):
    sid = r["sample_id"]
    cur = clean(r.get("current_best_answer", ""))
    gold = clean(r.get("gold_answer", ""))
    pred, reason = guarded_pred(r)

    orig = [clean(x) for x in r.get("orig_answers", []) if is_num(x)]
    extra = [clean(x) for x in r.get("extra_answers", []) if is_num(x)]
    cnt = Counter(orig)
    x = pick_extra_supported(r)

    cur_ok = ok(cur, gold)
    pred_ok = ok(pred, gold)

    if (not cur_ok) and pred_ok:
        outcome = "fixed"
    elif cur_ok and (not pred_ok):
        outcome = "broken"
    elif clean(pred) != cur:
        outcome = "changed_neutral"
    else:
        outcome = "unchanged"

    if sid in margin_ids:
        trigger_bucket = "in_margin030"
    else:
        trigger_bucket = "outside_margin030"

    if sid in all_ids:
        all_bucket = "in_all_disagree71"
    else:
        all_bucket = "outside_all_disagree71"

    support_count = cnt.get(x, 0) if x is not None else 0
    max_freq = max(cnt.values()) if cnt else 0

    extra_consensus = int(len(extra) >= 2 and len(set(extra)) == 1)
    extra_any_correct = int(any(ok(e, gold) for e in extra))
    extra_new_correct = int((not any(ok(a, gold) for a in orig)) and extra_any_correct)

    return {
        "sample_id": sid,
        "gold": gold,
        "current": cur,
        "pred": pred,
        "reason": reason,
        "cur_ok": int(cur_ok),
        "pred_ok": int(pred_ok),
        "outcome": outcome,
        "trigger_bucket": trigger_bucket,
        "all_bucket": all_bucket,
        "orig": orig,
        "extra": extra,
        "support": x,
        "support_count_in_orig": support_count,
        "orig_max_freq": max_freq,
        "support_is_orig_majority": int(x is not None and support_count == max_freq and max_freq >= 2),
        "current_in_orig": int(cur in cnt),
        "extra_consensus": extra_consensus,
        "extra_any_correct": extra_any_correct,
        "extra_new_correct": extra_new_correct,
        "changed": int(clean(pred) != cur),
    }


def summarize(name, feats):
    n = len(feats)
    if n == 0:
        print(f"\n## {name}: EMPTY")
        return

    fixed = sum(1 for x in feats if x["outcome"] == "fixed")
    broken = sum(1 for x in feats if x["outcome"] == "broken")
    changed = sum(x["changed"] for x in feats)
    pred_correct = sum(x["pred_ok"] for x in feats)
    cur_correct = sum(x["cur_ok"] for x in feats)

    print(f"\n## {name}")
    print("| n | cur_acc | pred_acc | fixed | broken | net | changed |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    print(f"| {n} | {cur_correct/n:.4f} | {pred_correct/n:.4f} | {fixed} | {broken} | {fixed-broken} | {changed} |")

    print("\n### outcome counter")
    print(Counter(x["outcome"] for x in feats))

    print("\n### reason counter")
    print(Counter(x["reason"] for x in feats))

    print("\n### feature buckets")
    keys = [
        "support_is_orig_majority",
        "current_in_orig",
        "extra_consensus",
        "extra_any_correct",
        "extra_new_correct",
    ]
    for k in keys:
        c = Counter(x[k] for x in feats)
        print(k, dict(c))


def print_cases(title, feats, outcome):
    xs = [x for x in feats if x["outcome"] == outcome]
    print(f"\n# {title}: {len(xs)}")
    for x in xs:
        print("=" * 100)
        print("sample_id:", x["sample_id"])
        print("bucket:", x["trigger_bucket"], x["all_bucket"])
        print("gold:", x["gold"])
        print("current:", x["current"], "cur_ok=", x["cur_ok"])
        print("pred:", x["pred"], "pred_ok=", x["pred_ok"], "reason=", x["reason"])
        print("orig:", x["orig"])
        print("extra:", x["extra"])
        print("support:", x["support"])
        print("support_count_in_orig:", x["support_count_in_orig"])
        print("orig_max_freq:", x["orig_max_freq"])
        print("support_is_orig_majority:", x["support_is_orig_majority"])
        print("current_in_orig:", x["current_in_orig"])
        print("extra_consensus:", x["extra_consensus"])
        print("extra_any_correct:", x["extra_any_correct"])
        print("extra_new_correct:", x["extra_new_correct"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resample_jsonl", required=True)
    ap.add_argument("--margin_ids", default="outputs/targets/gsm8k_full_margin030_107_sample_ids.txt")
    ap.add_argument("--all_disagree_ids", default="outputs/targets/gsm8k_full_all_disagree_71_sample_ids.txt")
    args = ap.parse_args()

    rows = read_jsonl(args.resample_jsonl)
    margin_ids = read_ids(args.margin_ids)
    all_ids = read_ids(args.all_disagree_ids)

    feats = [row_features(r, margin_ids, all_ids) for r in rows]

    print("file:", args.resample_jsonl)
    print("rows:", len(rows))
    print("margin_ids:", len(margin_ids))
    print("all_disagree_ids:", len(all_ids))

    summarize("ALL", feats)
    summarize("in_margin030", [x for x in feats if x["trigger_bucket"] == "in_margin030"])
    summarize("outside_margin030", [x for x in feats if x["trigger_bucket"] == "outside_margin030"])
    summarize("in_all_disagree71", [x for x in feats if x["all_bucket"] == "in_all_disagree71"])
    summarize("outside_all_disagree71", [x for x in feats if x["all_bucket"] == "outside_all_disagree71"])

    print_cases("FIXED CASES", feats, "fixed")
    print_cases("BROKEN CASES", feats, "broken")


if __name__ == "__main__":
    main()
