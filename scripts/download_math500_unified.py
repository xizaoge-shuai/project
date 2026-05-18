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


def extract_boxed_from_solution(solution):
    s = str(solution or "")
    m = re.findall(r"\\boxed\{([^{}]+)\}", s)
    if m:
        return m[-1].strip()
    return ""


def main():
    ds = load_dataset("HuggingFaceH4/MATH-500")

    if "test" in ds:
        split = ds["test"]
        split_name = "test"
    elif "train" in ds:
        split = ds["train"]
        split_name = "train"
    else:
        split_name = list(ds.keys())[0]
        split = ds[split_name]

    raw_rows = []
    unified_rows = []

    for i, ex in enumerate(split):
        problem = ex.get("problem") or ex.get("question") or ex.get("query") or ""
        solution = ex.get("solution", "")

        answer = (
            ex.get("answer")
            or ex.get("final_answer")
            or ex.get("target")
            or extract_boxed_from_solution(solution)
        )

        sid = f"math500_test_{i}"

        raw = dict(ex)
        raw["id"] = sid
        raw["hf_split"] = split_name
        raw_rows.append(raw)

        unified_rows.append({
            "id": sid,
            "sample_id": sid,
            "question": str(problem).strip(),
            "answer": str(answer).strip(),
            "gold_answer": str(answer).strip(),
            "solution": str(solution),
            "task": "math500",
            "context": "",
            "meta": {
                "dataset": "math500",
                "source": "HuggingFaceH4/MATH-500",
                "hf_split": split_name,
                "subject": ex.get("subject", ""),
                "level": ex.get("level", ""),
            }
        })

    write_jsonl("data/raw/math500/test.jsonl", raw_rows)
    write_jsonl("data/processed/unified/math500/test.jsonl", unified_rows)

    print("raw rows:", len(raw_rows))
    print("unified rows:", len(unified_rows))
    print("raw out:", "data/raw/math500/test.jsonl")
    print("unified out:", "data/processed/unified/math500/test.jsonl")
    print("example:")
    print(json.dumps(unified_rows[0], ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
