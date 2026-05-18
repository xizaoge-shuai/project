import argparse
import json
import re
import string
from pathlib import Path
from collections import Counter


def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]


def norm(s):
    s = str(s or "").lower().strip()
    s = re.sub(r"<([^>]+)>", r"\1", s)
    s = re.sub(r"\b(final answer|answer)\b\s*[:：]?", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s


def extract_short(s):
    raw = str(s or "").strip()
    if not raw:
        return ""

    parts = re.findall(r"final answer\s*[:：]\s*([^\n\.]+(?:\.[^\n]*)?)", raw, flags=re.I)
    cands = []

    for p in parts:
        p = p.strip()
        p = re.sub(r"<([^>]+)>", r"\1", p)
        p = p.split("Final Answer")[0].strip()
        p = p.strip(" .,:;|")
        if p:
            cands.append(p)

    # 也加入第一句，防止模型先给答案再重复 Final Answer
    first = raw.splitlines()[0].strip()
    first = first.split(". Final Answer")[0].strip()
    first = first.strip(" .,:;|")
    if first:
        cands.append(first)

    # 倾向短 span，过滤明显 prompt 残片
    clean = []
    for c in cands:
        if len(c.split()) <= 8 and "do not" not in c.lower() and "only answer" not in c.lower():
            clean.append(c)

    if clean:
        return min(clean, key=lambda x: (len(x.split()), len(x)))

    return first[:100]


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
    ap.add_argument("--details", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.details)

    em = any_em = contain_acc = f1_05 = f1_08 = 0
    out_rows = []

    for r in rows:
        gold = r["gold_answer"]
        answers_raw = r.get("answers", [])
        short_answers = [extract_short(a) for a in answers_raw]
        maj = majority([norm(a) for a in short_answers])

        # 找回 maj 原文本：用 norm 对应
        maj_text = ""
        for a in short_answers:
            if norm(a) == maj:
                maj_text = a
                break

        em_ok = int(norm(maj_text) == norm(gold))
        any_ok = int(any(norm(a) == norm(gold) for a in short_answers))
        contain_ok = int(contains(maj_text, gold))
        f1 = token_f1(maj_text, gold)

        em += em_ok
        any_em += any_ok
        contain_acc += contain_ok
        f1_05 += int(f1 >= 0.5)
        f1_08 += int(f1 >= 0.8)

        rr = dict(r)
        rr["short_answers"] = short_answers
        rr["majority_short_answer"] = maj_text
        rr["em_ok_short"] = em_ok
        rr["oracle_any_em_short"] = any_ok
        rr["contains_ok_short"] = contain_ok
        rr["token_f1_short"] = f1
        out_rows.append(rr)

    n = len(rows)
    summary = {
        "n_samples": n,
        "em_majority_short": em / max(1, n),
        "oracle_any_em_short": any_em / max(1, n),
        "substring_majority_short": contain_acc / max(1, n),
        "token_f1_ge_05_short": f1_05 / max(1, n),
        "token_f1_ge_08_short": f1_08 / max(1, n),
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
