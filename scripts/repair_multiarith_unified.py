import json
import re
from pathlib import Path
from datasets import load_dataset


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_number(x):
    if isinstance(x, list):
        if not x:
            return ""
        # MultiArith 常见答案字段可能是 list，比如 lSolutions
        for item in x:
            y = extract_number(item)
            if y != "":
                return y
        return ""

    if isinstance(x, dict):
        x = json.dumps(x, ensure_ascii=False)

    s = str(x or "")
    nums = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?", s)
    if not nums:
        return s.strip()

    y = nums[-1].replace(",", "")
    if "." in y:
        y = y.rstrip("0").rstrip(".")
    return y


def to_str(x):
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join(str(v) for v in x)
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def pick_question(row):
    # 常见字段优先
    q_keys = [
        "question", "Question", "sQuestion",
        "problem", "Problem", "text", "input", "query"
    ]
    for k in q_keys:
        if k in row and to_str(row[k]).strip():
            return to_str(row[k]).strip(), k

    # 兜底：找最长的字符串字段
    candidates = []
    for k, v in row.items():
        s = to_str(v).strip()
        if len(s) > 20 and not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", s):
            candidates.append((len(s), k, s))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][2], candidates[0][1]

    return "", ""


def pick_answer(row):
    # 常见答案字段优先
    a_keys = [
        "answer", "Answer", "final_ans", "FinalAnswer", "final_answer",
        "target", "Target", "label", "Label",
        "lSolutions", "solutions", "Solutions", "solution", "Solution"
    ]
    for k in a_keys:
        if k in row:
            ans = extract_number(row[k])
            if ans != "":
                return ans, k, row[k]

    # 兜底：找 key 里带 solution/answer/ans 的字段
    for k, v in row.items():
        kl = k.lower()
        if "solution" in kl or "answer" in kl or kl.endswith("ans"):
            ans = extract_number(v)
            if ans != "":
                return ans, k, v

    return "", "", None


def main():
    ds = load_dataset("ChilleD/MultiArith")
    split = "test" if "test" in ds else list(ds.keys())[0]
    rows = list(ds[split])

    raw_rows = []
    unified_rows = []
    bad = []

    for i, row in enumerate(rows):
        q, q_key = pick_question(row)
        ans, a_key, raw_ans = pick_answer(row)

        if not q or ans == "":
            bad.append({"idx": i, "keys": list(row.keys()), "row": row})
            continue

        rid = f"multiarith_test_{i}"

        raw = {
            "id": rid,
            "question": q,
            "answer": to_str(raw_ans),
            "task": "multiarith",
            "context": "",
            "meta": {
                "dataset": "multiarith",
                "split": "test",
                "hf_source": "ChilleD/MultiArith",
                "hf_split": split,
                "question_key": q_key,
                "answer_key": a_key,
            },
        }

        uni = {
            "id": rid,
            "question": q,
            "answer": ans,
            "task": "multiarith",
            "context": "",
            "meta": {
                "dataset": "multiarith",
                "split": "test",
                "hf_source": "ChilleD/MultiArith",
                "hf_split": split,
                "question_key": q_key,
                "answer_key": a_key,
            },
        }

        raw_rows.append(raw)
        unified_rows.append(uni)

    write_jsonl("data/raw/multiarith/test.jsonl", raw_rows)
    write_jsonl("data/processed/unified/multiarith/test.jsonl", unified_rows)

    print("source rows:", len(rows))
    print("raw rows:", len(raw_rows))
    print("unified rows:", len(unified_rows))
    print("bad rows:", len(bad))

    if unified_rows:
        print("example unified:")
        print(json.dumps(unified_rows[0], ensure_ascii=False, indent=2))

    if bad:
        print("\nfirst bad example:")
        print(json.dumps(bad[0], ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
