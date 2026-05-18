import argparse
import json
import re
import string
from pathlib import Path
from collections import defaultdict, Counter


def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]


def norm(s):
    s = str(s or "").lower().strip()
    s = re.sub(r"<([^>]+)>", r"\1", s)
    s = re.sub(r"\b(final answer|answer)\b\s*[:：]?", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s


def token_f1(a, g):
    aa = norm(a).split()
    gg = norm(g).split()
    if not aa or not gg:
        return 0.0
    ca, cg = Counter(aa), Counter(gg)
    inter = sum((ca & cg).values())
    if inter == 0:
        return 0.0
    p = inter / len(aa)
    r = inter / len(gg)
    return 2 * p * r / (p + r)


def contains(a, g):
    a = norm(a)
    g = norm(g)
    return bool(a and g and (a in g or g in a))


def majority(vals):
    vals = [v for v in vals if str(v).strip()]
    return Counter(vals).most_common(1)[0][0] if vals else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.trajectories)
    by = defaultdict(list)
    for r in rows:
        by[r["sample_id"]].append(r)

    first_em = maj_em = any_em = 0
    maj_contains = 0
    maj_f1_05 = maj_f1_08 = 0
    has_dis = all_dis = 0
    details = []

    for sid, rs in sorted(by.items()):
        rs = sorted(rs, key=lambda x: str(x.get("trajectory_id", "")))
        gold = rs[0].get("gold_answer", rs[0].get("answer", ""))

        answers = [r.get("final_answer", "") for r in rs]
        answers_norm = [norm(a) for a in answers]
        maj_norm = majority(answers_norm)

        maj_text = ""
        for a in answers:
            if norm(a) == maj_norm:
                maj_text = a
                break

        first_ok = int(norm(answers[0] if answers else "") == norm(gold))
        maj_ok = int(maj_norm == norm(gold))
        any_ok = int(any(norm(a) == norm(gold) for a in answers))
        contain_ok = int(contains(maj_text, gold))
        f1 = token_f1(maj_text, gold)

        first_em += first_ok
        maj_em += maj_ok
        any_em += any_ok
        maj_contains += contain_ok
        maj_f1_05 += int(f1 >= 0.5)
        maj_f1_08 += int(f1 >= 0.8)

        uniq = set(a for a in answers_norm if a)
        has_dis += int(len(uniq) >= 2)
        all_dis += int(len(uniq) >= 3)

        details.append({
            "sample_id": sid,
            "gold_answer": gold,
            "answers": answers,
            "answers_norm": answers_norm,
            "first_answer": answers[0] if answers else "",
            "majority_answer": maj_text,
            "majority_norm": maj_norm,
            "first_em": first_ok,
            "majority_em": maj_ok,
            "oracle_any_em": any_ok,
            "majority_contains": contain_ok,
            "majority_token_f1": f1,
        })

    n = len(by)
    summary = {
        "dataset": "hotpotqa",
        "n_samples": n,
        "n_trajectories": len(rows),
        "first_em": first_em / max(1, n),
        "majority_em": maj_em / max(1, n),
        "oracle_any_em": any_em / max(1, n),
        "majority_substring": maj_contains / max(1, n),
        "majority_token_f1_ge_05": maj_f1_05 / max(1, n),
        "majority_token_f1_ge_08": maj_f1_08 / max(1, n),
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
