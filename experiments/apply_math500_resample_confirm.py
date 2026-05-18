import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

from experiments.eval_math500_baseline import normalize, math_equal


def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]


def write_jsonl(fp, rows):
    p = Path(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(fp, obj):
    p = Path(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def norm_nonempty(x):
    y = normalize(x)
    return y if str(y).strip() else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_details", required=True)
    ap.add_argument("--extra_jsonls", nargs="+", required=True)
    ap.add_argument("--target_ids", required=True)
    ap.add_argument("--min_total_support", type=int, default=3)
    ap.add_argument("--min_seed_support", type=int, default=2)
    ap.add_argument("--base_acc", type=float, required=True)
    ap.add_argument("--n_samples", type=int, required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    target_ids = [x.strip() for x in open(args.target_ids, encoding="utf-8") if x.strip()]
    target_set = set(target_ids)

    base_rows = read_jsonl(args.baseline_details)
    base_by_id = {r["sample_id"]: r for r in base_rows if r["sample_id"] in target_set}

    # extras_by_id[sid] = list of (seed_index, answer_raw, answer_norm)
    extras_by_id = defaultdict(list)
    for seed_idx, fp in enumerate(args.extra_jsonls):
        for r in read_jsonl(fp):
            sid = r["sample_id"]
            if sid not in target_set:
                continue
            raw = str(r.get("final_answer", "")).strip()
            nrm = norm_nonempty(raw)
            if not nrm:
                continue
            extras_by_id[sid].append((seed_idx, raw, nrm))

    out_rows = []
    fixed = broken = changed = 0
    current_correct = final_correct = 0

    for sid in target_ids:
        b = base_by_id[sid]
        gold = b["gold_answer"]

        current_raw = b.get("majority_answer", "")
        current_norm = norm_nonempty(current_raw)

        # baseline_details 里的 majority_answer 有时是 normalize 后的形式，直接用于 math_equal
        cur_ok = int(math_equal(current_raw, gold))

        extras = extras_by_id.get(sid, [])
        cnt = Counter(x[2] for x in extras)

        raw_repr = {}
        seed_support = defaultdict(set)
        for seed_idx, raw, nrm in extras:
            raw_repr.setdefault(nrm, raw)
            seed_support[nrm].add(seed_idx)

        if cnt:
            top_norm, top_total = cnt.most_common(1)[0]
            top_seed = len(seed_support[top_norm])
            top_raw = raw_repr.get(top_norm, top_norm)
        else:
            top_norm, top_total, top_seed, top_raw = "", 0, 0, ""

        final_raw = current_raw
        reason = "keep_current"

        if (
            top_norm
            and top_norm != current_norm
            and top_total >= args.min_total_support
            and top_seed >= args.min_seed_support
        ):
            final_raw = top_raw
            reason = f"extra_support_total{top_total}_seed{top_seed}"

        fin_ok = int(math_equal(final_raw, gold))

        current_correct += cur_ok
        final_correct += fin_ok

        is_changed = int(norm_nonempty(final_raw) != current_norm)
        is_fixed = int(cur_ok == 0 and fin_ok == 1)
        is_broken = int(cur_ok == 1 and fin_ok == 0)

        changed += is_changed
        fixed += is_fixed
        broken += is_broken

        out_rows.append({
            "sample_id": sid,
            "gold_answer": gold,
            "current_answer": current_raw,
            "final_answer": final_raw,
            "current_ok": cur_ok,
            "final_ok": fin_ok,
            "fixed": is_fixed,
            "broken": is_broken,
            "changed": is_changed,
            "reason": reason,
            "top_answer": top_raw,
            "top_answer_norm": top_norm,
            "top_total_support": top_total,
            "top_seed_support": top_seed,
            "extra_support": dict(cnt),
            "extra_seed_support": {k: len(v) for k, v in seed_support.items()},
            "orig_answers": b.get("answers", []),
            "orig_answers_norm": b.get("answers_norm", []),
        })

    n_resampled = len(target_ids)
    net = fixed - broken

    summary = {
        "dataset": "math500",
        "rule": "extra_multiseed_support",
        "n_resampled": n_resampled,
        "base_acc": args.base_acc,
        "n_samples": args.n_samples,
        "min_total_support": args.min_total_support,
        "min_seed_support": args.min_seed_support,
        "current_acc_on_resampled": current_correct / max(1, n_resampled),
        "final_acc_on_resampled": final_correct / max(1, n_resampled),
        "fixed": fixed,
        "broken": broken,
        "net": net,
        "changed": changed,
        "estimated_global_acc": args.base_acc + net / args.n_samples,
        "estimated_global_gain": net / args.n_samples,
        "out_jsonl": args.out_jsonl,
    }

    write_jsonl(args.out_jsonl, out_rows)
    write_json(args.out_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
