import argparse
import json
import re
from pathlib import Path
from collections import Counter, defaultdict


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
    ap.add_argument("--min_total_support", type=int, default=3)
    ap.add_argument("--min_seed_support", type=int, default=2)
    ap.add_argument("--base_acc", type=float, default=0.910538286580743)
    ap.add_argument("--n_samples", type=int, default=1319)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    by_file = []
    for fp in args.resample_jsonls:
        rows = read_jsonl(fp)
        by_file.append({r["sample_id"]: r for r in rows})

    common_sids = sorted(set.intersection(*[set(x.keys()) for x in by_file]))

    out_rows = []
    fixed = broken = changed = correct = 0

    for sid in common_sids:
        base = by_file[0][sid]
        gold = base["gold_answer"]
        cur = clean(base.get("current_best_answer", ""))

        orig = [clean(x) for x in base.get("orig_answers", []) if is_num(x)]
        orig_set = set(orig)
        orig_cnt = Counter(orig)

        total_support = Counter()
        seed_support = defaultdict(set)

        for seed_idx, data in enumerate(by_file):
            r = data[sid]
            for a in r.get("extra_answers", []):
                a = clean(a)
                if is_num(a) and a in orig_set:
                    total_support[a] += 1
                    seed_support[a].add(seed_idx)

        pred = cur
        reason = "keep_current"

        candidates = []
        for ans, total_cnt in total_support.items():
            seed_cnt = len(seed_support[ans])
            candidates.append((ans, total_cnt, seed_cnt))

        candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)

        if candidates:
            top_ans, top_total, top_seed = candidates[0]

            tie = False
            if len(candidates) >= 2:
                _, second_total, second_seed = candidates[1]
                tie = (top_total == second_total and top_seed == second_seed)

            if (
                not tie
                and top_total >= args.min_total_support
                and top_seed >= args.min_seed_support
            ):
                max_freq = max(orig_cnt.values()) if orig_cnt else 0

                if cur in orig_cnt and orig_cnt.get(top_ans, 0) == max_freq and max_freq >= 2:
                    pred = cur
                    reason = "guard_keep_current_orig_majority"
                else:
                    pred = top_ans
                    reason = f"seedaware_total{top_total}_seed{top_seed}"

        cur_ok = ok(cur, gold)
        pred_ok = ok(pred, gold)

        fixed += int((not cur_ok) and pred_ok)
        broken += int(cur_ok and (not pred_ok))
        changed += int(pred != cur)
        correct += int(pred_ok)

        rr = dict(base)
        rr["seedaware_min_total_support"] = args.min_total_support
        rr["seedaware_min_seed_support"] = args.min_seed_support
        rr["seedaware_total_support"] = dict(total_support)
        rr["seedaware_seed_support"] = {k: len(v) for k, v in seed_support.items()}
        rr["seedaware_answer"] = pred
        rr["seedaware_reason"] = reason
        rr["seedaware_ok"] = int(pred_ok)
        rr["seedaware_fixed"] = int((not cur_ok) and pred_ok)
        rr["seedaware_broken"] = int(cur_ok and (not pred_ok))
        rr["seedaware_changed"] = int(pred != cur)
        out_rows.append(rr)

    net = fixed - broken

    summary = {
        "rule": "seedaware_multi_seed_confirmation",
        "n_runs": len(args.resample_jsonls),
        "n_resampled": len(common_sids),
        "min_total_support": args.min_total_support,
        "min_seed_support": args.min_seed_support,
        "base_acc": args.base_acc,
        "n_samples": args.n_samples,
        "fixed": fixed,
        "broken": broken,
        "net": net,
        "changed": changed,
        "acc_on_resampled": correct / max(1, len(common_sids)),
        "estimated_global_acc": args.base_acc + net / args.n_samples,
        "estimated_global_gain": net / args.n_samples,
        "out_jsonl": args.out_jsonl,
    }

    write_json(args.out_json, summary)
    write_jsonl(args.out_jsonl, out_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
