import argparse
import json
import re
import string
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


def normalize_text(s):
    s = str(s or "").lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s


def normalize_answer(x, dataset, choices=None):
    s = str(x or "").strip()

    if dataset == "strategyqa":
        low = s.lower()
        if "yes" in low and "no" not in low:
            return "yes"
        if "no" in low and "yes" not in low:
            return "no"
        if low in {"true", "1"}:
            return "yes"
        if low in {"false", "0"}:
            return "no"
        return low

    if dataset == "mathqa":
        low = s.lower().strip()
        m = re.search(r"\b([abcde])\b", low)
        if m:
            return m.group(1)
        m = re.search(r"^([abcde])[\)\.:]", low)
        if m:
            return m.group(1)

        if choices:
            ns = normalize_text(low)
            for k, v in choices.items():
                if normalize_text(v) == ns:
                    return k
        return low

    return normalize_text(s)


def ok(ans, gold, dataset, choices=None):
    return normalize_answer(ans, dataset, choices or {}) == normalize_answer(gold, dataset, choices or {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_details", required=True)
    ap.add_argument("--extra_jsonls", nargs="+", required=True)
    ap.add_argument("--target_ids", required=True)
    ap.add_argument("--dataset", required=True, choices=["strategyqa", "mathqa"])
    ap.add_argument("--min_total_support", type=int, default=2)
    ap.add_argument("--min_seed_support", type=int, default=2)
    ap.add_argument("--min_margin", type=int, default=1)
    ap.add_argument("--base_acc", type=float, required=True)
    ap.add_argument("--n_samples", type=int, required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    target_ids = [x.strip() for x in open(args.target_ids, encoding="utf-8") if x.strip()]
    target_set = set(target_ids)

    base_rows = read_jsonl(args.baseline_details)
    base_by_id = {r["sample_id"]: r for r in base_rows if r["sample_id"] in target_set}

    extras = defaultdict(list)
    for seed_idx, fp in enumerate(args.extra_jsonls):
        for r in read_jsonl(fp):
            sid = r["sample_id"]
            if sid not in target_set:
                continue
            choices = r.get("choices", {}) or {}
            ans = normalize_answer(r.get("final_answer", ""), args.dataset, choices)
            if ans:
                extras[sid].append((seed_idx, ans))

    fixed = broken = changed = 0
    cur_correct = final_correct = 0
    out_rows = []

    for sid in target_ids:
        b = base_by_id[sid]
        gold = b["gold_answer"]

        current = b.get("majority_answer", "")
        current_norm = normalize_answer(current, args.dataset, {})

        cur_ok = int(ok(current_norm, gold, args.dataset, {}))

        cnt = Counter(a for _, a in extras.get(sid, []) if a)
        seed_support = defaultdict(set)
        for seed_idx, ans in extras.get(sid, []):
            seed_support[ans].add(seed_idx)

        if cnt:
            top, top_total = cnt.most_common(1)[0]
            runner_total = cnt.most_common(2)[1][1] if len(cnt) >= 2 else 0
            top_seed = len(seed_support[top])
        else:
            top, top_total, runner_total, top_seed = "", 0, 0, 0

        margin = top_total - runner_total

        final = current_norm
        reason = "keep_current"

        if (
            top
            and top != current_norm
            and top_total >= args.min_total_support
            and top_seed >= args.min_seed_support
            and margin >= args.min_margin
        ):
            final = top
            reason = f"extra_total{top_total}_seed{top_seed}_margin{margin}"

        fin_ok = int(ok(final, gold, args.dataset, {}))

        is_changed = int(final != current_norm)
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
            "current_answer": current_norm,
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
            "runner_total": runner_total,
            "margin": margin,
            "extra_support": dict(cnt),
            "extra_seed_support": {k: len(v) for k, v in seed_support.items()},
            "base_answers": b.get("answers", []),
            "base_answers_norm": b.get("answers_norm", []),
        })

    n_eval = len(target_ids)
    net = fixed - broken

    summary = {
        "dataset": args.dataset,
        "rule": "extra_multiseed_confirmation",
        "n_eval": n_eval,
        "base_acc": args.base_acc,
        "n_samples": args.n_samples,
        "min_total_support": args.min_total_support,
        "min_seed_support": args.min_seed_support,
        "min_margin": args.min_margin,
        "current_acc_on_eval": cur_correct / max(1, n_eval),
        "final_acc_on_eval": final_correct / max(1, n_eval),
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
