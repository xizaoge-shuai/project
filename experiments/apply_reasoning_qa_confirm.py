import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

from experiments.eval_reasoning_qa_baseline import normalize_answer


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


def ok(ans, gold, dataset, choices=None):
    return normalize_answer(ans, dataset, choices or {}) == normalize_answer(gold, dataset, choices or {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_details", required=True)
    ap.add_argument("--dataset", required=True, choices=["strategyqa", "mathqa"])
    ap.add_argument("--min_total_support", type=int, default=2)
    ap.add_argument("--min_margin", type=int, default=1)
    ap.add_argument("--base_acc", type=float, required=True)
    ap.add_argument("--n_samples", type=int, required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.baseline_details)

    fixed = broken = changed = 0
    cur_correct = final_correct = 0
    out_rows = []

    for r in rows:
        sid = r["sample_id"]
        gold = r["gold_answer"]
        choices = r.get("choices", {}) or {}

        answers = r.get("answers", [])
        answers_norm = r.get("answers_norm", [])

        if not answers_norm:
            answers_norm = [normalize_answer(a, args.dataset, choices) for a in answers]

        cur = r.get("majority_answer", "")
        cur_norm = normalize_answer(cur, args.dataset, choices)

        cnt = Counter(a for a in answers_norm if str(a).strip())
        if cnt:
            top, top_total = cnt.most_common(1)[0]
            runner = cnt.most_common(2)[1][1] if len(cnt) >= 2 else 0
        else:
            top, top_total, runner = "", 0, 0

        margin = top_total - runner

        cur_ok = int(ok(cur_norm, gold, args.dataset, choices))
        final = cur_norm
        reason = "keep_current"

        if top and top != cur_norm and top_total >= args.min_total_support and margin >= args.min_margin:
            final = top
            reason = f"top_total{top_total}_margin{margin}"

        fin_ok = int(ok(final, gold, args.dataset, choices))

        is_changed = int(final != cur_norm)
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
            "current_answer": cur_norm,
            "final_answer": final,
            "current_ok": cur_ok,
            "final_ok": fin_ok,
            "fixed": is_fixed,
            "broken": is_broken,
            "changed": is_changed,
            "reason": reason,
            "support": dict(cnt),
            "top": top,
            "top_total": top_total,
            "runner_total": runner,
            "margin": margin,
            "answers": answers,
            "answers_norm": answers_norm,
        })

    n = len(rows)
    net = fixed - broken

    summary = {
        "dataset": args.dataset,
        "rule": "majority_support_replay",
        "n_eval": n,
        "base_acc": args.base_acc,
        "n_samples": args.n_samples,
        "min_total_support": args.min_total_support,
        "min_margin": args.min_margin,
        "current_acc_on_eval": cur_correct / max(1, n),
        "final_acc_on_eval": final_correct / max(1, n),
        "fixed": fixed,
        "broken": broken,
        "net": net,
        "changed": changed,
        "estimated_global_acc": args.base_acc + net / args.n_samples,
        "estimated_global_gain": net / args.n_samples,
        "out_jsonl": args.out_jsonl,
    }

    write_json(args.out_json, summary)
    write_jsonl(args.out_jsonl, out_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
