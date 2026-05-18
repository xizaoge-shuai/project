import argparse
import json
from pathlib import Path
from datasets import load_dataset


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm_yesno(x):
    if isinstance(x, bool):
        return "yes" if x else "no"
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return "yes"
    if s in {"false", "0", "no", "n"}:
        return "no"
    return s


def flatten_hotpot_context(ctx, max_chars=8000):
    if not ctx:
        return ""
    parts = []

    if isinstance(ctx, dict):
        titles = ctx.get("title", [])
        sents = ctx.get("sentences", [])
        for title, ss in zip(titles, sents):
            if isinstance(ss, list):
                text = " ".join(str(x) for x in ss)
            else:
                text = str(ss)
            parts.append(f"[{title}] {text}")
    elif isinstance(ctx, list):
        for item in ctx:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title, ss = item[0], item[1]
                if isinstance(ss, list):
                    text = " ".join(str(x) for x in ss)
                else:
                    text = str(ss)
                parts.append(f"[{title}] {text}")
            else:
                parts.append(str(item))
    else:
        parts.append(str(ctx))

    out = "\n".join(parts)
    return out[:max_chars]


def parse_mathqa_options(opt):
    s = str(opt or "")
    # 常见格式: "a ) xxx , b ) yyy , ..."
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


def convert_strategyqa():
    # 优先用你之前用过的数据源
    ds = load_dataset("metaeval/strategy-qa")
    split_name = "test" if "test" in ds else ("validation" if "validation" in ds else list(ds.keys())[0])
    split = ds[split_name]

    raw_rows, unified = [], []
    for i, ex in enumerate(split):
        sid = f"strategyqa_{split_name}_{i}"
        q = ex.get("question", "")
        ans = norm_yesno(ex.get("answer", ex.get("label", "")))

        raw = dict(ex)
        raw["id"] = sid
        raw_rows.append(raw)

        unified.append({
            "id": sid,
            "sample_id": sid,
            "question": str(q).strip(),
            "answer": ans,
            "gold_answer": ans,
            "task": "strategyqa",
            "context": "",
            "meta": {"dataset": "strategyqa", "source": "metaeval/strategy-qa", "split": split_name}
        })

    write_jsonl("data/raw/strategyqa/test.jsonl", raw_rows)
    write_jsonl("data/processed/unified/strategyqa/test.jsonl", unified)
    print("strategyqa rows:", len(unified))


def convert_hotpotqa():
    ds = load_dataset("hotpot_qa", "distractor")
    split_name = "validation" if "validation" in ds else ("test" if "test" in ds else list(ds.keys())[0])
    split = ds[split_name]

    raw_rows, unified = [], []
    for i, ex in enumerate(split):
        sid = ex.get("id") or ex.get("_id") or f"hotpotqa_{split_name}_{i}"
        q = ex.get("question", "")
        ans = str(ex.get("answer", "")).strip()
        ctx = flatten_hotpot_context(ex.get("context", ""))

        raw = dict(ex)
        raw["id"] = sid
        raw_rows.append(raw)

        unified.append({
            "id": sid,
            "sample_id": sid,
            "question": str(q).strip(),
            "answer": ans,
            "gold_answer": ans,
            "task": "hotpotqa",
            "context": ctx,
            "meta": {"dataset": "hotpotqa", "source": "hotpot_qa/distractor", "split": split_name}
        })

    write_jsonl("data/raw/hotpotqa/test.jsonl", raw_rows)
    write_jsonl("data/processed/unified/hotpotqa/test.jsonl", unified)
    print("hotpotqa rows:", len(unified))


def convert_mathqa():
    ds = load_dataset("math_qa")
    split_name = "test" if "test" in ds else ("validation" if "validation" in ds else list(ds.keys())[0])
    split = ds[split_name]

    raw_rows, unified = [], []
    for i, ex in enumerate(split):
        sid = f"mathqa_{split_name}_{i}"
        q = ex.get("Problem", ex.get("problem", ex.get("question", "")))
        opts = ex.get("options", "")
        correct = str(ex.get("correct", ex.get("answer", ""))).strip().lower()
        choices = parse_mathqa_options(opts)

        raw = dict(ex)
        raw["id"] = sid
        raw_rows.append(raw)

        unified.append({
            "id": sid,
            "sample_id": sid,
            "question": str(q).strip(),
            "answer": correct,
            "gold_answer": correct,
            "task": "mathqa",
            "context": str(opts),
            "choices": choices,
            "meta": {"dataset": "mathqa", "source": "math_qa", "split": split_name}
        })

    write_jsonl("data/raw/mathqa/test.jsonl", raw_rows)
    write_jsonl("data/processed/unified/mathqa/test.jsonl", unified)
    print("mathqa rows:", len(unified))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["strategyqa", "hotpotqa", "mathqa", "all"])
    args = ap.parse_args()

    if args.dataset in ["strategyqa", "all"]:
        convert_strategyqa()
    if args.dataset in ["hotpotqa", "all"]:
        convert_hotpotqa()
    if args.dataset in ["mathqa", "all"]:
        convert_mathqa()


if __name__ == "__main__":
    main()
