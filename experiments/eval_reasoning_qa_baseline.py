import argparse
import json
import re
import string
from pathlib import Path
from collections import defaultdict, Counter


def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]


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

    if dataset == "hotpotqa":
        return normalize_text(s)

    return normalize_text(s)


def ok(pred, gold, dataset, choices=None):
    p = normalize_answer(pred, dataset, choices)
    g = normalize_answer(gold, dataset, choices)
    return p == g


def majority(vals):
    vals = [v for v in vals if str(v).strip()]
    if not vals:
        return ""
    return Counter(vals).most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--dataset", required=True, choices=["strategyqa", "hotpotqa", "mathqa"])
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.trajectories)
    by = defaultdict(list)
    for r in rows:
        by[r["sample_id"]].append(r)

    first = maj = anyok = 0
    has_dis = all_dis = 0
    details = []

    for sid, rs in sorted(by.items()):
        rs = sorted(rs, key=lambda x: str(x.get("trajectory_id", "")))
        gold = rs[0].get("gold_answer", rs[0].get("answer", ""))
        choices = rs[0].get("choices", {})

        answers_raw = [r.get("final_answer", "") for r in rs]
        answers_norm = [normalize_answer(a, args.dataset, choices) for a in answers_raw]

        first_ans = answers_raw[0] if answers_raw else ""
        maj_norm = majority(answers_norm)

        first_ok = int(ok(first_ans, gold, args.dataset, choices))
        maj_ok = int(maj_norm == normalize_answer(gold, args.dataset, choices))
        any_ok = int(any(ok(a, gold, args.dataset, choices) for a in answers_raw))

        first += first_ok
        maj += maj_ok
        anyok += any_ok

        uniq = set(a for a in answers_norm if a)
        has_dis += int(len(uniq) >= 2)
        all_dis += int(len(uniq) >= 3)

        details.append({
            "sample_id": sid,
            "gold_answer": gold,
            "gold_norm": normalize_answer(gold, args.dataset, choices),
            "answers": answers_raw,
            "answers_norm": answers_norm,
            "first_answer": first_ans,
            "majority_answer": maj_norm,
            "first_ok": first_ok,
            "majority_ok": maj_ok,
            "oracle_any_ok": any_ok,
        })

    n = len(by)
    summary = {
        "dataset": args.dataset,
        "n_samples": n,
        "n_trajectories": len(rows),
        "first_acc": first / max(1, n),
        "majority_acc": maj / max(1, n),
        "oracle_any_acc": anyok / max(1, n),
        "has_disagreement": has_dis,
        "all_disagree": all_dis,
        "out_jsonl": args.out_jsonl,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in details:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
