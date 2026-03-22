from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import re

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed/unified")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_gsm8k_final_answer(answer: str) -> str:
    answer = str(answer).strip()

    # 优先匹配 GSM8K 官方格式：#### 42
    m = re.search(r"####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)", answer)
    if m:
        return m.group(1).replace(",", "")

    # 兜底：取最后一个数字
    nums = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", answer)
    if nums:
        return nums[-1].replace(",", "")

    return answer


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def preprocess_gsm8k() -> None:
    in_dir = RAW_DIR / "gsm8k"
    out_dir = OUT_DIR / "gsm8k"
    ensure_dir(out_dir)

    total = 0
    for split in ["train", "test"]:
        fp = in_dir / f"{split}.jsonl"
        if not fp.exists():
            continue

        raw_rows = read_jsonl(fp)
        out_rows: List[Dict[str, Any]] = []

        for i, item in enumerate(raw_rows):
            out_rows.append(
                {
                    "id": item.get("id", f"gsm8k_{split}_{i}"),
                    "question": item["question"],
                    "answer": extract_gsm8k_final_answer(item["answer"]),
                    "task": "gsm8k",
                    "context": "",
                    "meta": {
                        "dataset": "gsm8k",
                        "split": split,
                    },
                }
            )

        write_jsonl(out_dir / f"{split}.jsonl", out_rows)
        print(f"Preprocessed gsm8k/{split}: {len(out_rows)} rows")
        total += len(out_rows)

    print(f"Preprocessed gsm8k total: {total} rows")


def preprocess_strategyqa() -> None:
    in_dir = RAW_DIR / "strategyqa"
    out_dir = OUT_DIR / "strategyqa"
    ensure_dir(out_dir)

    total = 0
    for split in ["train", "validation", "test"]:
        fp = in_dir / f"{split}.jsonl"
        if not fp.exists():
            continue

        raw_rows = read_jsonl(fp)
        out_rows: List[Dict[str, Any]] = []

        for i, item in enumerate(raw_rows):
            answer = item.get("answer", "")
            if isinstance(answer, bool):
                answer = "yes" if answer else "no"
            answer = str(answer).strip().lower()

            out_rows.append(
                {
                    "id": item.get("id", f"strategyqa_{split}_{i}"),
                    "question": item["question"],
                    "answer": answer,
                    "task": "strategyqa",
                    "context": item.get("context", ""),
                    "meta": {
                        "dataset": "strategyqa",
                        "split": split,
                    },
                }
            )

        write_jsonl(out_dir / f"{split}.jsonl", out_rows)
        print(f"Preprocessed strategyqa/{split}: {len(out_rows)} rows")
        total += len(out_rows)

    print(f"Preprocessed strategyqa total: {total} rows")


def preprocess_hotpotqa() -> None:
    in_dir = RAW_DIR / "hotpotqa"
    out_dir = OUT_DIR / "hotpotqa"
    ensure_dir(out_dir)

    total = 0
    for split in ["train", "validation", "test"]:
        fp = in_dir / f"{split}.jsonl"
        if not fp.exists():
            continue

        raw_rows = read_jsonl(fp)
        out_rows: List[Dict[str, Any]] = []

        for i, item in enumerate(raw_rows):
            out_rows.append(
                {
                    "id": item.get("id", f"hotpotqa_{split}_{i}"),
                    "question": item["question"],
                    "answer": item["answer"],
                    "task": "hotpotqa",
                    "context": item.get("context", ""),
                    "meta": {
                        "dataset": "hotpotqa",
                        "split": split,
                    },
                }
            )

        write_jsonl(out_dir / f"{split}.jsonl", out_rows)
        print(f"Preprocessed hotpotqa/{split}: {len(out_rows)} rows")
        total += len(out_rows)

    print(f"Preprocessed hotpotqa total: {total} rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", "gsm8k", "strategyqa", "hotpotqa"],
    )
    args = parser.parse_args()

    ensure_dir(OUT_DIR)

    if args.dataset in ("all", "gsm8k") and (RAW_DIR / "gsm8k").exists():
        preprocess_gsm8k()

    if args.dataset in ("all", "strategyqa") and (RAW_DIR / "strategyqa").exists():
        preprocess_strategyqa()

    if args.dataset in ("all", "hotpotqa") and (RAW_DIR / "hotpotqa").exists():
        preprocess_hotpotqa()


if __name__ == "__main__":
    main()
