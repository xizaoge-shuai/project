import argparse
import json
import re
from pathlib import Path
from collections import defaultdict, Counter

try:
    import sympy as sp
except Exception:
    sp = None


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


def extract_balanced_boxed(text):
    s = str(text or "")
    key = r"\boxed{"
    idx = s.rfind(key)
    if idx < 0:
        return ""

    i = idx + len(key)
    depth = 1
    out = []

    while i < len(s):
        ch = s[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
            out.append(ch)
        else:
            out.append(ch)
        i += 1

    return ""


def replace_frac(s):
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    for _ in range(20):
        ns = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
        if ns == s:
            break
        s = ns
    return s


def normalize(s):
    s = str(s or "").strip()

    boxed = extract_balanced_boxed(s)
    if boxed:
        s = boxed

    s = s.replace("$", "")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\,", "")
    s = s.replace(r"\!", "")
    s = s.replace(r"\pi", "pi")
    s = replace_frac(s)
    s = s.replace("^", "**")

    # 常见等价写法
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"\s+", "", s)
    s = s.strip().strip(".")
    return s


def sympy_expr(s):
    s = normalize(s)

    # 坐标、集合、区间这类先不强行 sympify，避免误判
    if "," in s:
        return None

    try:
        return sp.sympify(s)
    except Exception:
        return None


def math_equal(a, b):
    na = normalize(a)
    nb = normalize(b)

    if na == nb:
        return True

    # 去掉最外层括号再比较
    if na.startswith("(") and na.endswith(")") and nb.startswith("(") and nb.endswith(")"):
        if na[1:-1] == nb[1:-1]:
            return True

    if sp is None:
        return False

    ea = sympy_expr(na)
    eb = sympy_expr(nb)
    if ea is None or eb is None:
        return False

    try:
        return bool(sp.simplify(ea - eb) == 0)
    except Exception:
        return False


def majority_answer(answers):
    vals = [normalize(a) for a in answers if normalize(a)]
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

    by_sid = defaultdict(list)
    for r in rows:
        by_sid[r["sample_id"]].append(r)

    first = 0
    majority = 0
    oracle_any = 0
    has_disagreement = 0
    all_disagree = 0
    empty_final = 0
    details = []

    for sid, rs in sorted(by_sid.items()):
        rs = sorted(rs, key=lambda x: x.get("trajectory_id", ""))
        gold = rs[0].get("gold_answer", rs[0].get("answer", ""))
        answers = [r.get("final_answer", "") for r in rs]

        empty_final += sum(1 for a in answers if not str(a).strip())

        first_ans = answers[0] if answers else ""
        maj_ans = majority_answer(answers)

        first_ok = int(math_equal(first_ans, gold))
        maj_ok = int(math_equal(maj_ans, gold))
        any_ok = int(any(math_equal(a, gold) for a in answers))

        first += first_ok
        majority += maj_ok
        oracle_any += any_ok

        uniq = set(normalize(a) for a in answers if normalize(a))
        has_disagreement += int(len(uniq) >= 2)
        all_disagree += int(len(uniq) >= 3)

        details.append({
            "sample_id": sid,
            "gold_answer": gold,
            "answers": answers,
            "answers_norm": [normalize(a) for a in answers],
            "first_answer": first_ans,
            "majority_answer": maj_ans,
            "first_ok": first_ok,
            "majority_ok": maj_ok,
            "oracle_any_ok": any_ok,
        })

    n = len(by_sid)
    summary = {
        "dataset": "math500",
        "n_samples": n,
        "n_trajectories": len(rows),
        "first_acc": first / max(1, n),
        "majority_acc": majority / max(1, n),
        "oracle_any_acc": oracle_any / max(1, n),
        "has_disagreement": has_disagreement,
        "all_disagree": all_disagree,
        "empty_final": empty_final,
        "empty_final_rate": empty_final / max(1, len(rows)),
        "out_jsonl": args.out_jsonl,
    }

    write_json(args.out_json, summary)
    write_jsonl(args.out_jsonl, details)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
