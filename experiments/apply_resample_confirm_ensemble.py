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


def is_num(x):
    return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", clean(x) or ""))


def ok(ans, gold):
    return clean(ans) == clean(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resample_jsonls", nargs="+", required=True)
    ap.add_argument("--min_support", type=int, default=2)
    ap.add_argument("--base_acc", type=float, default=0.910538286580743)
    ap.add_argument("--n_samples", type=int, default=1319)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    by_file = []
    for fp in args.resample_jsonls:
        rows = read_jsonl(fp)
        by_file.append({r["sample_id"]: r for r in rows})

    sids = sorted(set.intersection(*[set(x.keys()) for x in by_file]))

    out_rows = []
    fixed = broken = changed = correct = 0

    for sid in sids:
        base = by_file[0][sid]
        gold = base["gold_answer"]
        cur = clean(base.get("current_best_answer", ""))

        orig = [clean(x) for x in base.get("orig_answers", []) if is_num(x)]
        orig_set = set(orig)
        cnt_orig = Counter(orig)

        support = Counter()
        support_sources = {}

        for run_idx, data in enumerate(by_file):
            r = data[sid]
            for x in r.get("extra_answers", []):
                x = clean(x)
                if is_num(x) and x in orig_set:
                    support[x] += 1
                    support_sources.setdefault(x, []).append(run_idx)

        pred = cur
        reason = "keep_current"
        if support:
            ranked = support.most_common()
            top_ans, top_count = ranked[0]

            # tie -> keep current
            tie = len(ranked) >= 2 and ranked[1][1] == top_count

            if top_count >= args.min_support and not tie:
                # frequency guard: if top is original majority and current also appears in original candidates, keep current
                max_freq = max(cnt_orig.values()) if cnt_orig else 0
                if cur in cnt_orig and cnt_orig.get(top_ans, 0) == max_freq and max_freq >= 2:
                    pred = cur
                    reason = "guard_keep_current_orig_majority"
                else:
                    pred = top_ans
                    reason = f"ensemble_support_{top_count}"

        cur_ok = ok(cur, gold)
        pred_ok = ok(pred, gold)

        fixed += int((not cur_ok) and pred_ok)
        broken += int(cur_ok and (not pred_ok))
        changed += int(pred != cur)
        correct += int(pred_ok)

        rr = dict(base)
        rr["ensemble_min_support"] = args.min_support
        rr["ensemble_support"] = dict(support)
        rr["ensemble_answer"] = pred
        rr["ensemble_reason"] = reason
        rr["ensemble_ok"] = int(pred_ok)
        rr["ensemble_fixed"] = int((not cur_ok) and pred_ok)
        rr["ensemble_broken"] = int(cur_ok and (not pred_ok))
        rr["ensemble_changed"] = int(pred != cur)
        out_rows.append(rr)

    net = fixed - broken
    summary = {
        "rule": "multi_seed_resampling_confirmation",
        "n_runs": len(args.resample_jsonls),
        "n_resampled": len(sids),
        "min_support": args.min_support,
        "base_acc": args.base_acc,
        "n_samples": args.n_samples,
        "fixed": fixed,
        "broken": broken,
        "net": net,
        "changed": changed,
        "acc_on_resampled": correct / max(1, len(sids)),
        "estimated_global_acc": args.base_acc + net / args.n_samples,
        "estimated_global_gain": net / args.n_samples,
        "out_jsonl": args.out_jsonl,
    }

    write_json(args.out_json, summary)
    write_jsonl(args.out_jsonl, out_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
