import argparse
import json
import re
from pathlib import Path
from collections import Counter, defaultdict


def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]


def clean(x):
    s = str(x or "").strip().replace(",", "").replace("$", "")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    if not nums:
        return s
    y = nums[-1]
    if "." in y:
        y = y.rstrip("0").rstrip(".")
    return y


def ok(a, g):
    return clean(a) == clean(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_details", required=True)
    ap.add_argument("--target_ids", required=True)
    ap.add_argument("--extra_jsonls", nargs="+", required=True)
    ap.add_argument("--per_seed_budget", type=int, required=True)
    ap.add_argument("--min_total_support", type=int, default=2)
    ap.add_argument("--min_seed_support", type=int, default=2)
    ap.add_argument("--min_margin", type=int, default=1)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    base_rows = read_jsonl(args.baseline_details)
    base_by_id = {r["sample_id"]: r for r in base_rows}
    numeric_n = len(base_rows)
    numeric_base_acc = sum(int(r.get("majority_ok", 0)) for r in base_rows) / max(1, numeric_n)

    target_ids = [x.strip() for x in open(args.target_ids, encoding="utf-8") if x.strip()]

    extras = defaultdict(list)
    for seed_idx, fp in enumerate(args.extra_jsonls):
        per_sample_count = defaultdict(int)
        for r in read_jsonl(fp):
            sid = r["sample_id"]
            if sid not in base_by_id:
                continue
            if per_sample_count[sid] >= args.per_seed_budget:
                continue
            ans = clean(r.get("final_answer", ""))
            if ans:
                extras[sid].append((seed_idx, ans))
                per_sample_count[sid] += 1

    fixed = broken = changed = 0
    cur_correct = final_correct = 0
    out_rows = []

    for sid in target_ids:
        b = base_by_id[sid]
        gold = b["gold_answer"]
        current = clean(b.get("majority_answer", ""))
        cur_ok = int(ok(current, gold))

        cnt = Counter(a for _, a in extras.get(sid, []) if a)
        seed_support = defaultdict(set)
        for seed_idx, ans in extras.get(sid, []):
            seed_support[ans].add(seed_idx)

        if cnt:
            top, top_total = cnt.most_common(1)[0]
            runner = cnt.most_common(2)[1][1] if len(cnt) >= 2 else 0
            top_seed = len(seed_support[top])
        else:
            top, top_total, runner, top_seed = "", 0, 0, 0

        margin = top_total - runner
        final = current
        reason = "keep_current"

        if (
            top and top != current
            and top_total >= args.min_total_support
            and top_seed >= args.min_seed_support
            and margin >= args.min_margin
        ):
            final = top
            reason = f"extra_total{top_total}_seed{top_seed}_margin{margin}"

        fin_ok = int(ok(final, gold))

        is_changed = int(final != current)
        is_fixed = int(cur_ok == 0 and fin_ok == 1)
        is_broken = int(cur_ok == 1 and fin_ok == 0)

        cur_correct += cur_ok
        final_correct += fin_ok
        fixed += is_fixed
        broken += is_broken
        changed += is_changed

        out_rows.append({
            "sample_id": sid,
            "gold_answer": gold,
            "current_answer": current,
            "final_answer": final,
            "current_ok": cur_ok,
            "final_ok": fin_ok,
            "fixed": is_fixed,
            "broken": is_broken,
            "changed": is_changed,
            "reason": reason,
            "top": top,
            "top_total": top_total,
            "top_seed": top_seed,
            "runner_total": runner,
            "margin": margin,
            "extra_support": dict(cnt),
            "per_seed_budget": args.per_seed_budget,
        })

    net = fixed - broken
    summary = {
        "dataset": "asdiv_numeric",
        "ablation": "per_seed_extra_budget",
        "per_seed_budget": args.per_seed_budget,
        "n_eval": len(target_ids),
        "numeric_base_acc": numeric_base_acc,
        "numeric_n": numeric_n,
        "min_total_support": args.min_total_support,
        "min_seed_support": args.min_seed_support,
        "min_margin": args.min_margin,
        "current_acc_on_eval": cur_correct / max(1, len(target_ids)),
        "final_acc_on_eval": final_correct / max(1, len(target_ids)),
        "fixed": fixed,
        "broken": broken,
        "net": net,
        "changed": changed,
        "estimated_numeric_acc": numeric_base_acc + net / numeric_n,
        "estimated_numeric_gain": net / numeric_n,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
