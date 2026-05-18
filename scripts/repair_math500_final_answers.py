import argparse
import json
import re
from pathlib import Path


BAD_SET = {"", ":", "：", ";", ".", ","}


def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]


def write_jsonl(fp, rows):
    p = Path(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def is_bad_answer(x):
    s = str(x or "").strip()
    if s in BAD_SET:
        return True
    # 单字符数字/变量可以是合法答案，例如 5、9、x、n
    if len(s) == 1 and (s.isdigit() or s.isalpha()):
        return False
    # 只有纯标点才算坏答案
    if len(s) <= 1:
        return True
    return False


def extract_balanced_boxed(text):
    s = str(text or "")
    key = r"\boxed"
    idx = s.rfind(key)
    if idx < 0:
        return ""

    brace = s.find("{", idx)
    if brace < 0:
        return ""

    i = brace + 1
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


def clean_candidate(x):
    s = str(x or "").strip()
    s = s.strip("$")
    s = s.strip()
    s = s.rstrip(".")
    if s.lower().startswith("answer:"):
        s = s.split(":", 1)[1].strip()
    return s


def extract_from_response(text):
    s = str(text or "")

    boxed = extract_balanced_boxed(s)
    if boxed:
        return clean_candidate(boxed)

    # 从后往前找更接近结尾的答案行
    lines = [x.strip() for x in s.splitlines() if x.strip()]
    for line in reversed(lines):
        if r"\boxed" in line:
            boxed = extract_balanced_boxed(line)
            if boxed:
                return clean_candidate(boxed)

        # The final answer is p - q
        m = re.search(r"(?:final answer|answer)\s*(?:is|=|:)?\s*(.+)$", line, flags=re.I)
        if m:
            cand = clean_candidate(m.group(1))
            if not is_bad_answer(cand) and len(cand) <= 120:
                return cand

    # 最后兜底：如果最后一行很短，而且看起来像答案
    for line in reversed(lines):
        cand = clean_candidate(line)
        if not is_bad_answer(cand) and len(cand) <= 80:
            if any(ch.isdigit() for ch in cand) or "\\" in cand or re.search(r"[a-zA-Z]", cand):
                return cand

    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.input)

    n_changed = 0
    n_bad_before = 0
    n_bad_after = 0

    out = []
    for r in rows:
        rr = dict(r)
        old = str(rr.get("final_answer", "")).strip()
        if is_bad_answer(old):
            n_bad_before += 1
            new = extract_from_response(rr.get("response", rr.get("generated_text", "")))
            if new and new != old:
                rr["final_answer_before_repair"] = old
                rr["final_answer"] = new
                rr["final_answer_repaired"] = True
                n_changed += 1

        if is_bad_answer(rr.get("final_answer", "")):
            n_bad_after += 1

        out.append(rr)

    write_jsonl(args.output, out)

    print({
        "input": args.input,
        "output": args.output,
        "rows": len(rows),
        "bad_before": n_bad_before,
        "changed": n_changed,
        "bad_after": n_bad_after,
    })


if __name__ == "__main__":
    main()
