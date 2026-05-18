import argparse
import json
import re
import string
from pathlib import Path
from collections import Counter, defaultdict


def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]


def norm_basic(s):
    s = str(s or "").lower().strip()
    s = re.sub(r"^(final answer\s*[:：]\s*)", "", s)
    s = re.sub(r"^(the answer is|answer is)\s+", "", s)
    s = s.strip().strip(".").strip()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s


def norm_yesno(s):
    x = str(s or "").lower()
    yes = re.search(r"\byes\b|\btrue\b", x)
    no = re.search(r"\bno\b|\bfalse\b", x)
    if yes and not no:
        return "yes"
    if no and not yes:
        return "no"
    if x.strip() in {"1"}:
        return "yes"
    if x.strip() in {"0"}:
        return "no"
    return norm_basic(s)


def norm_option(s):
    x = str(s or "").lower().strip()
    x = re.sub(r"^(final answer\s*[:：]\s*)", "", x)
    x = re.sub(r"^(the answer is|answer is)\s+", "", x).strip()
    m = re.search(r"\boption\s*([a-e])\b", x)
    if m:
        return m.group(1)
    m = re.search(r"^\(?([a-e])\)?\.?$", x)
    if m:
        return m.group(1)
    m = re.search(r"\(([a-e])\)", x)
    if m:
        return m.group(1)
    return norm_basic(x)


def norm_bool(s):
    x = str(s or "").lower().strip()
    if re.search(r"\btrue\b", x) and not re.search(r"\bfalse\b", x):
        return "true"
    if re.search(r"\bfalse\b", x) and not re.search(r"\btrue\b", x):
        return "false"
    if re.search(r"\byes\b", x) and not re.search(r"\bno\b", x):
        return "true"
    if re.search(r"\bno\b", x) and not re.search(r"\byes\b", x):
        return "false"
    return norm_basic(x)


def token_f1(a, b):
    aa = norm_basic(a).split()
    bb = norm_basic(b).split()
    if not aa or not bb:
        return 0.0
    ca, cb = Counter(aa), Counter(bb)
    inter = sum((ca & cb).values())
    if inter == 0:
        return 0.0
    p = inter / len(aa)
    r = inter / len(bb)
    return 2 * p * r / (p + r)


def contains_match(a, g):
    a = norm_basic(a)
    g = norm_basic(g)
    return bool(a and g and (a in g or g in a))


def majority(vals):
    vals = [v for v in vals if str(v).strip()]
    return Counter(vals).most_common(1)[0][0] if vals else ""


def classify_task(task, subtask=""):
    name = (task + " " + subtask).lower()
    if "strategy" in name:
        return "yesno"
    if "hotpot" in name:
        return "openqa"
    if "boolean" in name:
        return "bool"
    if "formal_fallacies" in name or "logical_deduction" in name or "tracking_shuffled" in name:
        return "option_or_text"
    return "basic"


def normalize_by_kind(s, kind):
    if kind == "yesno":
        return norm_yesno(s)
    if kind == "bool":
        return norm_bool(s)
    if kind == "option_or_text":
        return norm_option(s)
    return norm_basic(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--subtask", default="")
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_cases", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.details)
    kind = classify_task(args.task, args.subtask)

    n = len(rows)
    stats = Counter()
    gold_counter = Counter()
    maj_counter = Counter()
    ans_div = Counter()
    f1_bins = Counter()

    cases = []

    for r in rows:
        gold = r.get("gold_answer", r.get("answer", ""))
        answers = r.get("answers", [])
        if not answers:
            a = r.get("majority_answer", "")
            answers = [a] if a else []

        answers_norm = [normalize_by_kind(a, kind) for a in answers]
        gold_norm = normalize_by_kind(gold, kind)
        maj = r.get("majority_answer", "")
        maj_norm = normalize_by_kind(maj, kind) if maj else majority(answers_norm)

        gold_counter[gold_norm] += 1
        maj_counter[maj_norm] += 1
        ans_div[len(set(x for x in answers_norm if x))] += 1

        exact = int(maj_norm == gold_norm)
        any_exact = int(any(a == gold_norm for a in answers_norm))
        contain = int(contains_match(maj, gold))
        f1 = token_f1(maj, gold)

        if f1 >= 0.8:
            f1_bins[">=0.8"] += 1
        elif f1 >= 0.5:
            f1_bins["0.5-0.8"] += 1
        elif f1 > 0:
            f1_bins["0-0.5"] += 1
        else:
            f1_bins["0"] += 1

        stats["exact"] += exact
        stats["any_exact"] += any_exact
        stats["contains"] += contain
        stats["f1_ge_05"] += int(f1 >= 0.5)
        stats["f1_ge_08"] += int(f1 >= 0.8)

        if not exact:
            reason = []
            if any_exact:
                reason.append("oracle_has_correct_but_majority_wrong")
            if contain:
                reason.append("substring_match_possible")
            if f1 >= 0.5:
                reason.append("token_f1_match_possible")
            if not any_exact and f1 < 0.5:
                reason.append("generation_or_context_failure")
            if len(set(x for x in answers_norm if x)) >= 2:
                reason.append("answer_disagreement")
            if not maj_norm:
                reason.append("empty_majority")

            cases.append({
                "sample_id": r.get("sample_id", r.get("id", "")),
                "gold": gold,
                "gold_norm": gold_norm,
                "majority": maj,
                "majority_norm": maj_norm,
                "answers": answers,
                "answers_norm": answers_norm,
                "f1": round(f1, 4),
                "contains": contain,
                "reason": reason,
            })

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(f"# Diagnosis: {args.task} {args.subtask}\n\n")
        f.write(f"kind: `{kind}`\n\n")
        f.write("| metric | value |\n|---|---:|\n")
        f.write(f"| n | {n} |\n")
        f.write(f"| exact_majority | {stats['exact']/max(1,n):.4f} |\n")
        f.write(f"| oracle_any_exact | {stats['any_exact']/max(1,n):.4f} |\n")
        f.write(f"| substring_possible | {stats['contains']/max(1,n):.4f} |\n")
        f.write(f"| token_f1_ge_0.5 | {stats['f1_ge_05']/max(1,n):.4f} |\n")
        f.write(f"| token_f1_ge_0.8 | {stats['f1_ge_08']/max(1,n):.4f} |\n\n")

        f.write("## answer diversity\n\n")
        f.write("| unique_answers | count |\n|---:|---:|\n")
        for k, v in sorted(ans_div.items()):
            f.write(f"| {k} | {v} |\n")

        f.write("\n## gold top\n\n")
        f.write("| gold_norm | count |\n|---|---:|\n")
        for k, v in gold_counter.most_common(20):
            f.write(f"| {k} | {v} |\n")

        f.write("\n## majority top\n\n")
        f.write("| majority_norm | count |\n|---|---:|\n")
        for k, v in maj_counter.most_common(20):
            f.write(f"| {k} | {v} |\n")

        f.write("\n## token F1 bins\n\n")
        f.write("| bin | count |\n|---|---:|\n")
        for k, v in f1_bins.items():
            f.write(f"| {k} | {v} |\n")

        f.write("\n## first 40 wrong cases\n\n")
        for c in cases[:40]:
            f.write("\n" + "=" * 100 + "\n")
            f.write(json.dumps(c, ensure_ascii=False, indent=2) + "\n")

    with open(args.out_cases, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print("saved:", args.out_md)
    print("wrong_cases:", len(cases))
    print("saved_cases:", args.out_cases)


if __name__ == "__main__":
    main()
