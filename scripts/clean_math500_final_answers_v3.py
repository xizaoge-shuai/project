import argparse
import json
import re
from pathlib import Path


BAD_SET = {"", ":", "：", ";", ".", ",", "\\[", "\\]", "\\boxed"}


def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]


def write_jsonl(fp, rows):
    p = Path(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def balanced_boxed_all(text):
    s = str(text or "")
    keys = [r"\boxed{", r"\boxed("]
    outs = []

    for key in keys:
        start = 0
        while True:
            idx = s.find(key, start)
            if idx < 0:
                break

            if key.endswith("{"):
                close = "}"
            else:
                close = ")"

            i = idx + len(key)
            depth = 1
            out = []

            while i < len(s):
                ch = s[i]
                if ch == key[-1]:
                    depth += 1
                    out.append(ch)
                elif ch == close:
                    depth -= 1
                    if depth == 0:
                        cand = "".join(out).strip()
                        if cand:
                            outs.append(cand)
                        break
                    out.append(ch)
                else:
                    out.append(ch)
                i += 1

            start = idx + len(key)

    return outs


def clean_candidate(x):
    s = str(x or "").strip()
    s = s.strip("$").strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "")
    s = s.replace("\\!", "")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.rstrip(".。")

    # 去掉常见句子壳
    for pat in [
        r"^final answer\s*[:：]\s*",
        r"^the final answer is\s*",
        r"^answer\s*[:：]\s*",
        r"^therefore,\s*",
        r"^thus,\s*",
    ]:
        s = re.sub(pat, "", s, flags=re.I).strip()

    # 如果整段里还有 boxed，取最后一个 boxed 内容
    boxed = balanced_boxed_all(s)
    if boxed:
        s = boxed[-1].strip()

    # 去掉外层 \( \), \[ \]
    s = re.sub(r"^\\\((.*)\\\)$", r"\1", s).strip()
    s = re.sub(r"^\\\[(.*)\\\]$", r"\1", s).strip()

    # 处理重复 final answer 文本：只留第一个明显答案片段
    split_markers = [
        "isthefinalanswer",
        "is the final answer",
        "is the answer",
        "is the solution",
        "\\boxed",
    ]
    low = s.lower().replace(" ", "")
    for marker in split_markers[:1]:
        pos = low.find(marker)
        if pos > 0:
            # 不直接用 low 切原串，只在过长时触发
            if len(s) > 80:
                s = s[:pos].strip()

    # 过长且包含等号/冒号的，优先从最后的短答案抽
    if len(s) > 120:
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
        if nums:
            return nums[-1]

    return s.strip()


def is_bad(x):
    s = str(x or "").strip()
    if s in BAD_SET:
        return True
    if not s:
        return True
    # 单字符数字/变量是合法答案
    if len(s) == 1 and (s.isdigit() or s.isalpha()):
        return False
    # 太像句子开头，不是答案
    bad_prefixes = [
        "we need to",
        "the equation",
        "the inequality",
        "setting",
        "therefore",
        "thus",
        "now",
        "for ",
        "to get",
    ]
    if any(s.lower().startswith(p) for p in bad_prefixes):
        return True
    return False


def extract_from_response(text):
    s = str(text or "")

    boxed = balanced_boxed_all(s)
    if boxed:
        cand = clean_candidate(boxed[-1])
        if not is_bad(cand):
            return cand

    lines = [x.strip() for x in s.splitlines() if x.strip()]
    for line in reversed(lines):
        boxed = balanced_boxed_all(line)
        if boxed:
            cand = clean_candidate(boxed[-1])
            if not is_bad(cand):
                return cand

        m = re.search(r"(?:final answer|answer)\s*(?:is|=|:)?\s*(.+)$", line, flags=re.I)
        if m:
            cand = clean_candidate(m.group(1))
            if not is_bad(cand) and len(cand) <= 120:
                return cand

    # 兜底：短行
    for line in reversed(lines):
        cand = clean_candidate(line)
        if not is_bad(cand) and len(cand) <= 80:
            if any(ch.isdigit() for ch in cand) or "\\" in cand or re.search(r"[a-zA-Z]", cand):
                return cand

    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    out = []

    changed = 0
    bad_before = 0
    bad_after = 0

    for r in rows:
        rr = dict(r)
        old = clean_candidate(rr.get("final_answer", ""))
        if is_bad(old) or len(old) > 120:
            bad_before += 1
            new = extract_from_response(rr.get("response", rr.get("generated_text", "")))
        else:
            new = old

        if new != rr.get("final_answer", ""):
            rr["final_answer_before_clean_v3"] = rr.get("final_answer", "")
            rr["final_answer"] = new
            rr["final_answer_cleaned_v3"] = True
            changed += 1

        if is_bad(rr.get("final_answer", "")) or len(str(rr.get("final_answer", ""))) > 120:
            bad_after += 1

        out.append(rr)

    write_jsonl(args.output, out)
    print({
        "input": args.input,
        "output": args.output,
        "rows": len(rows),
        "bad_before": bad_before,
        "changed": changed,
        "bad_after": bad_after,
    })


if __name__ == "__main__":
    main()
