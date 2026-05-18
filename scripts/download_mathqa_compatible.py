import json
from pathlib import Path
from datasets import load_dataset


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_mathqa_options(opt):
    s = str(opt or "")
    choices = {}
    for lab in ["a", "b", "c", "d", "e"]:
        marker = f"{lab} )"
        if marker in s:
            tail = s.split(marker, 1)[1]
            next_pos = len(tail)
            for lab2 in ["a", "b", "c", "d", "e"]:
                if lab2 <= lab:
                    continue
                m2 = f" , {lab2} )"
                p = tail.find(m2)
                if p >= 0:
                    next_pos = min(next_pos, p)
            choices[lab] = tail[:next_pos].strip(" ,")
    return choices


def pick_field(ex, keys, default=""):
    for k in keys:
        if k in ex and ex[k] is not None:
            return ex[k]
    return default


def main():
    # 兼容 datasets>=4.0 的 MathQA 镜像
    ds = load_dataset("regisss/math_qa")

    print(ds)

    split_name = "test" if "test" in ds else ("validation" if "validation" in ds else list(ds.keys())[0])
    split = ds[split_name]

    raw_rows = []
    unified_rows = []

    print("split:", split_name)
    print("columns:", split.column_names)
    print("num_rows:", len(split))

    for i, ex in enumerate(split):
        sid = f"mathqa_{split_name}_{i}"

        q = pick_field(ex, ["Problem", "problem", "question", "Question"])
        opts = pick_field(ex, ["options", "Options", "choices"], "")
        correct = str(pick_field(ex, ["correct", "Correct", "answer", "Answer"], "")).strip().lower()

        # 只保留 a/b/c/d/e
        if correct:
            correct = correct[0].lower()

        choices = parse_mathqa_options(opts)

        raw = dict(ex)
        raw["id"] = sid
        raw["hf_source"] = "regisss/math_qa"
        raw["hf_split"] = split_name
        raw_rows.append(raw)

        unified_rows.append({
            "id": sid,
            "sample_id": sid,
            "question": str(q).strip(),
            "answer": correct,
            "gold_answer": correct,
            "task": "mathqa",
            "context": str(opts),
            "choices": choices,
            "meta": {
                "dataset": "mathqa",
                "source": "regisss/math_qa",
                "split": split_name,
            }
        })

    write_jsonl("data/raw/mathqa/test.jsonl", raw_rows)
    write_jsonl("data/processed/unified/mathqa/test.jsonl", unified_rows)

    print("raw rows:", len(raw_rows))
    print("unified rows:", len(unified_rows))
    print("example unified:")
    print(json.dumps(unified_rows[0], ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
