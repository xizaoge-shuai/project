import argparse
import json
import re
import string
from pathlib import Path
from collections import Counter


def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]


def clean_text(s):
    s = str(s or "").lower().strip()
    s = re.sub(r"^(final answer\s*[:：]\s*)", "", s)
    s = re.sub(r"^(the answer is|answer is|answer)\s*[:：]?\s*", "", s)
    s = s.strip().strip(".").strip()
    return s


def extract_option(s):
    raw = str(s or "")
    x = clean_text(raw)

    # 优先抽最后一个 Final Answer 后面的选项
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

    # 如果开头就是 a/b/c + 文本，也按选项处理
    m = re.search(r"^\s*([a-e])\b", x)
    if m:
        return m.group(1)

    # boolean
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


def norm_gold(g):
    return extract_option(g)


def majority(vals):
    vals = [v for v in vals if str(v).strip()]
    return Counter(vals).most_common(1)[0][0] if vals else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.details)

    first = maj = anyok = 0
    has_dis = all_dis = 0
    out_rows = []

    for r in rows:
        gold = r["gold_answer"]
        gold_norm = norm_gold(gold)

        answers = r.get("answers", [])
        answers_norm = [extract_option(a) for a in answers]
        maj_ans = majority(answers_norm)

        first_ok = int((answers_norm[0] if answers_norm else "") == gold_norm)
        maj_ok = int(maj_ans == gold_norm)
        any_ok = int(any(a == gold_norm for a in answers_norm))

        first += first_ok
        maj += maj_ok
        anyok += any_ok

        uniq = set(a for a in answers_norm if a)
        has_dis += int(len(uniq) >= 2)
        all_dis += int(len(uniq) >= 3)

        rr = dict(r)
        rr["gold_norm_fixed"] = gold_norm
        rr["answers_norm_fixed"] = answers_norm
        rr["majority_answer_fixed"] = maj_ans
        rr["first_ok_fixed"] = first_ok
        rr["majority_ok_fixed"] = maj_ok
        rr["oracle_any_ok_fixed"] = any_ok
        out_rows.append(rr)

    n = len(rows)
    summary = {
        "n_samples": n,
        "first_acc_fixed": first / max(1, n),
        "majority_acc_fixed": maj / max(1, n),
        "oracle_any_acc_fixed": anyok / max(1, n),
        "has_disagreement_fixed": has_dis,
        "all_disagree_fixed": all_dis,
        "out_jsonl": args.out_jsonl,
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
