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
    s = re.sub(r"^(final answer\s*[:：]\s*)", "", s)
    s = re.sub(r"^(the answer is|answer is)\s+", "", s)
    s = s.strip().strip(".").strip()

    # true/false 统一
    if s in {"true", "yes"}:
        return "true"
    if s in {"false", "no"}:
        return "false"

    # 去标点
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s


def ok(a, g):
    return norm(a) == norm(g)


def majority(vals):
    vals = [v for v in vals if str(v).strip()]
    if not vals:
        return ""
    return Counter(vals).most_common(1)[0][0]


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

    first = maj = anyok = 0
    has_dis = all_dis = 0
    details = []

    for sid, rs in sorted(by.items()):
        rs = sorted(rs, key=lambda x: str(x.get("trajectory_id", "")))
        gold = rs[0].get("gold_answer", rs[0].get("answer", ""))

        answers_raw = [r.get("final_answer", "") for r in rs]
        answers_norm = [norm(a) for a in answers_raw]
        maj_norm = majority(answers_norm)

        first_ok = int(ok(answers_raw[0] if answers_raw else "", gold))
        maj_ok = int(maj_norm == norm(gold))
        any_ok = int(any(ok(a, gold) for a in answers_raw))

        first += first_ok
        maj += maj_ok
        anyok += any_ok

        uniq = set(a for a in answers_norm if a)
        has_dis += int(len(uniq) >= 2)
        all_dis += int(len(uniq) >= 3)

        details.append({
            "sample_id": sid,
            "subtask": rs[0].get("subtask", ""),
            "gold_answer": gold,
            "gold_norm": norm(gold),
            "answers": answers_raw,
            "answers_norm": answers_norm,
            "first_answer": answers_raw[0] if answers_raw else "",
            "majority_answer": maj_norm,
            "first_ok": first_ok,
            "majority_ok": maj_ok,
            "oracle_any_ok": any_ok,
        })

    n = len(by)
    summary = {
        "dataset": "bbh_logic",
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
