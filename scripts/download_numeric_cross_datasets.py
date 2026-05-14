import json
import re
from pathlib import Path
from datasets import load_dataset


SPECS = {
    "svamp": [
        "ChilleD/SVAMP",
        "tongyx361/svamp",
    ],
    "multiarith": [
        "ChilleD/MultiArith",
    ],
    "asdiv": [
        "EleutherAI/asdiv",
        "yimingzhang/asdiv",
        "nguyen-brat/asdiv",
    ],
}


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def pick_split(ds):
    for s in ["test", "validation", "val", "train"]:
        if s in ds:
            return s
    return list(ds.keys())[0]


def pick(row, names):
    for n in names:
        if n in row and row[n] is not None and str(row[n]).strip():
            return row[n]
    return None


def extract_number(x):
    """
    和 GSM8K unified 的思想一致：
    raw answer 可能是完整推理文本，但 unified answer 只保留最终数字。
    """
    s = str(x)
    if "####" in s:
        s = s.split("####")[-1]
    nums = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?", s)
    if not nums:
        return s.strip()
    y = nums[-1].replace(",", "")
    if "." in y:
        y = y.rstrip("0").rstrip(".")
    return y


def normalize_question(ds_name, row):
    body = pick(row, ["Body", "body", "context", "Context", "passage", "Passage"])
    q = pick(row, ["Question", "question", "Problem", "problem", "query", "input", "text"])

    body = str(body or "").strip()
    q = str(q or "").strip()

    if body and q:
        if body in q:
            return q
        return body + "\n" + q
    return q or body


def normalize_raw_row(ds_name, row, idx):
    """
    raw 文件也整理成项目常用格式：
    id/question/answer/task/context
    但 answer 保留原始答案文本。
    """
    q = normalize_question(ds_name, row)
    ans = pick(row, [
        "Answer", "answer", "target", "Target",
        "FinalAnswer", "final_answer",
        "correct", "Correct", "label", "Label"
    ])

    if not q or ans is None:
        return None

    return {
        "id": f"{ds_name}_test_{idx}",
        "question": q,
        "answer": str(ans),
        "task": ds_name,
        "context": "",
    }


def normalize_unified_row(raw_row, ds_name):
    """
    unified 文件严格对齐你展示的 GSM8K unified 风格：
    id/question/answer/task/context/meta
    """
    return {
        "id": raw_row["id"],
        "question": raw_row["question"],
        "answer": extract_number(raw_row["answer"]),
        "task": ds_name,
        "context": raw_row.get("context", ""),
        "meta": {
            "dataset": ds_name,
            "split": "test",
        },
    }


def load_one(ds_name):
    last_err = None
    for hf_name in SPECS[ds_name]:
        try:
            print(f"[LOAD] {ds_name}: {hf_name}")
            ds = load_dataset(hf_name)
            split = pick_split(ds)
            rows = list(ds[split])
            print(f"[OK] {ds_name}: source={hf_name}, split={split}, rows={len(rows)}")
            return hf_name, split, rows
        except Exception as e:
            last_err = e
            print(f"[FAIL] {ds_name}: {hf_name}: {repr(e)}")
    raise RuntimeError(f"Cannot load {ds_name}: {last_err}")


def main():
    for ds_name in ["svamp", "multiarith", "asdiv"]:
        source, split, rows = load_one(ds_name)

        raw_rows = []
        unified_rows = []
        bad = 0

        for i, row in enumerate(rows):
            rr = normalize_raw_row(ds_name, row, i)
            if rr is None:
                bad += 1
                continue

            # 保留来源信息，但放在不影响主字段的位置
            rr["meta"] = {
                "dataset": ds_name,
                "split": "test",
                "hf_source": source,
                "hf_split": split,
            }

            ur = normalize_unified_row(rr, ds_name)
            ur["meta"]["hf_source"] = source
            ur["meta"]["hf_split"] = split

            raw_rows.append(rr)
            unified_rows.append(ur)

        raw_out = Path(f"data/raw/{ds_name}/test.jsonl")
        unified_out = Path(f"data/processed/unified/{ds_name}/test.jsonl")

        write_jsonl(raw_out, raw_rows)
        write_jsonl(unified_out, unified_rows)

        print(f"[SAVED] {ds_name}")
        print(" raw:", raw_out, "rows=", len(raw_rows))
        print(" unified:", unified_out, "rows=", len(unified_rows))
        print(" bad:", bad)

        if raw_rows:
            print("[RAW EXAMPLE]", json.dumps(raw_rows[0], ensure_ascii=False)[:500])
        if unified_rows:
            print("[UNIFIED EXAMPLE]", json.dumps(unified_rows[0], ensure_ascii=False)[:500])


if __name__ == "__main__":
    main()
