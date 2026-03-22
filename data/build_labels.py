from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_yes_no(answer: str) -> str:
    x = clean_text(answer).lower().strip(" .,:;!?\"'`()[]{}")
    if x.startswith("yes") or x in {"true", "correct"}:
        return "yes"
    if x.startswith("no") or x in {"false", "incorrect"}:
        return "no"
    return x


def normalize_math(answer: str) -> str:
    x = clean_text(answer)
    x = x.strip("`$ ")
    x = x.replace(",", "")
    x = x.strip(" .,:;!?")
    x = re.sub(r"^(the answer is)\s+", "", x, flags=re.IGNORECASE)
    x = re.sub(r"^(therefore|thus|so)\s*,?\s*", "", x, flags=re.IGNORECASE)
    return x.strip()


def normalize_answer(answer: str, task: str) -> str:
    if task == "strategyqa":
        return normalize_yes_no(answer)
    if task == "gsm8k":
        return normalize_math(answer)
    return clean_text(answer).strip("`$ ").strip(" .,:;!?")


def answers_match(pred: str, gold: str, task: str) -> bool:
    pred_n = normalize_answer(pred, task)
    gold_n = normalize_answer(gold, task)

    if task == "strategyqa":
        return pred_n == gold_n

    if task == "gsm8k":
        return pred_n == gold_n

    return pred_n.lower() == gold_n.lower()


def infer_error_type(prefix: Dict[str, Any], label_success: int) -> str:
    if label_success == 1:
        return "none"

    task = prefix.get("task", "generic")
    prefix_text = clean_text(prefix.get("prefix_text", ""))
    context = clean_text(prefix.get("context", ""))

    if task == "gsm8k":
        return "arithmetic_error"

    if task in {"strategyqa", "hotpotqa"}:
        if context and len(prefix_text.split()) < 20:
            return "missing_evidence"
        if len(prefix_text.split()) >= 20:
            return "unsupported_jump"
        return "premature_conclusion"

    return "unknown"


def infer_repairability(prefix: Dict[str, Any], label_success: int) -> int:
    """
    第一版弱规则：
    - 已经正确 => 0
    - 错误但有一定长度 => 1
    - 太短/空 => 0
    """
    if label_success == 1:
        return 0

    prefix_text = clean_text(prefix.get("prefix_text", ""))
    num_words = len(prefix_text.split())

    if 5 <= num_words <= 200:
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="prefix jsonl 文件")
    parser.add_argument("--output_dir", required=True, help="labels 输出根目录")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    prefixes = read_jsonl(input_path)

    level = (
        input_path.parent.parent.name
        if input_path.parent.parent.name in {"path_level", "step_level", "atom_level"}
        else input_path.parent.name
    )
    dataset_name = input_path.parent.name
    split = input_path.stem

    success_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []
    repair_rows: List[Dict[str, Any]] = []

    for prefix in prefixes:
        task = prefix.get("task", "generic")
        pred = prefix.get("final_answer", "")
        gold = prefix.get("gold_answer", "")

        label_success = 1 if answers_match(pred, gold, task) else 0
        error_type = infer_error_type(prefix, label_success)
        repairability = infer_repairability(prefix, label_success)

        base = {
            "prefix_id": prefix["prefix_id"],
            "sample_id": prefix["sample_id"],
            "trajectory_id": prefix["trajectory_id"],
            "dataset": prefix["dataset"],
            "split": prefix["split"],
            "task": prefix["task"],
            "level": prefix["level"],
            "question": prefix["question"],
            "context": prefix.get("context", ""),
            "gold_answer": gold,
            "prefix_text": prefix["prefix_text"],
            "prefix_num_units": prefix.get("prefix_num_units", 0),
            "final_answer": pred,
        }

        success_rows.append(
            {
                **base,
                "label_success": label_success,
                "soft_confidence": float(label_success),
            }
        )

        error_rows.append(
            {
                **base,
                "error_type": error_type,
            }
        )

        repair_rows.append(
            {
                **base,
                "repairability": repairability,
            }
        )

    write_jsonl(
        output_dir / "success" / level / dataset_name / f"{split}.jsonl", success_rows
    )
    write_jsonl(
        output_dir / "error_type" / level / dataset_name / f"{split}.jsonl", error_rows
    )
    write_jsonl(
        output_dir / "repairability" / level / dataset_name / f"{split}.jsonl",
        repair_rows,
    )

    print(f"Saved success labels: {len(success_rows)}")
    print(f"Saved error_type labels: {len(error_rows)}")
    print(f"Saved repairability labels: {len(repair_rows)}")


if __name__ == "__main__":
    main()
