import argparse
import json
import re
import string
from pathlib import Path
from collections import Counter, defaultdict


def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]


def write_json(fp, obj):
    Path(fp).parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(fp, rows):
    Path(fp).parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def clean_text(s):
    s = str(s or "").lower().strip()
    s = re.sub(r"^(final answer\s*[:：]\s*)", "", s)
    s = re.sub(r"^(the answer is|answer is|answer)\s*[:：]?\s*", "", s)
    s = s.strip().strip(".").strip()
    return s


def extract_option(s):
    raw = str(s or "")
    x = clean_text(raw)

    fa = re.findall(r"final answer\s*[:：]\s*([^\n\|]+)", raw, flags=re.I)
    candidates = fa + [raw]

    for c in reversed(candidates):
        c = str(c or "").strip()

        m = re.search(r"\(([A-Ea-e])\)", c)
        if m:
            return m.group(1).lower()

        m = re.search(r"\boption\s*([A-Ea-e])\b", c, flags=re.I)
        if m:
            return m.group(1).lower()

        m = re.search(r"^\s*([A-Ea-e])[\)\.\:]\s*", c)
        if m:
            return m.group(1).lower()

        m = re.search(r"final answer\s*[:：]\s*([A-Ea-e])\b", c, flags=re.I)
        if m:
            return m.group(1).lower()

    m = re.search(r"^\s*([a-e])\b", x)
    if m:
        return m.group(1)

    if re.search(r"\btrue\b", x) and not re.search(r"\bfalse\b", x):
        return "true"
    if re.search(r"\bfalse\b", x) and not re.search(r"\btrue\b", x):
        return "false"
    if re.search(r"\byes\b", x) and not re.search(r"\bno\b", x):
        return "true"
    if re.search(r"\bno\b", x) and not re.search(r"\byes\b", x):
        return "false"

    x = "".join(ch for ch in x if ch not in string.punctuation)
    x = " ".join(x.split())
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_fixed_details", required=True)
    ap.add_argument("--extra_jsonls", nargs="+", required=True)
    ap.add_argument("--target_ids", required=True)
    ap.add_argument("--subtask", required=True)
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

    base_rows = read_jsonl(args.baseline_fixed_details)
    base_by_id = {r["sample_id"]: r for r in base_rows if r["sample_id"] in target_set}

    extras = defaultdict(list)
    for seed_idx, fp in enumerate(args.extra_jsonls):
        for r in read_jsonl(fp):
            sid = r["sample_id"]
            if sid not in target_set:
                continue
            ans = extract_option(r.get("final_answer", ""))
            if ans:
                extras[sid].append((seed_idx, ans))

    fixed = broken = changed = 0
    cur_correct = final_correct = 0
    out_rows = []

    for sid in target_ids:
        b = base_by_id[sid]
        gold = b.get("gold_norm_fixed") or extract_option(b.get("gold_answer", ""))
        current = b.get("majority_answer_fixed") or extract_option(b.get("majority_answer", ""))

        cur_ok = int(current == gold)

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
        final = current
        reason = "keep_current"

        if (
            top
            and top != current
            and top_total >= args.min_total_support
            and top_seed >= args.min_seed_support
            and margin >= args.min_margin
        ):
            final = top
            reason = f"extra_total{top_total}_seed{top_seed}_margin{margin}"

        fin_ok = int(final == gold)

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
            "subtask": args.subtask,
            "gold_answer": b.get("gold_answer", ""),
            "gold_norm_fixed": gold,
            "current_answer_fixed": current,
            "final_answer_fixed": final,
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
            "base_answers_norm_fixed": b.get("answers_norm_fixed", []),
        })

    n_eval = len(target_ids)
    net = fixed - broken

    summary = {
        "dataset": "bbh_logic",
        "subtask": args.subtask,
        "rule": "fixed_extractor_extra_multiseed_confirmation",
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
