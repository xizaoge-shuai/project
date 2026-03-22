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
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def atomize_text(text: str) -> List[str]:
    """
    简化版 atomizer：
    - 先按换行切
    - 再按句号/问号/感叹号切
    - 再去掉空白
    """
    text = clean_text(text)
    if not text:
        return []

    chunks: List[str] = []
    for block in text.split("\n"):
        block = clean_text(block)
        if not block:
            continue
        sents = re.split(r"(?<=[.!?。！？])\s+", block)
        for s in sents:
            s = clean_text(s)
            if s:
                chunks.append(s)
    return chunks


def build_path_level_prefixes(traj: Dict[str, Any]) -> List[Dict[str, Any]]:
    prefix_text = clean_text(traj.get("reasoning_text", ""))
    if not prefix_text:
        return []

    row = {
        "prefix_id": f'{traj["trajectory_id"]}_path_0',
        "sample_id": traj["sample_id"],
        "trajectory_id": traj["trajectory_id"],
        "dataset": traj["dataset"],
        "split": traj["split"],
        "task": traj["task"],
        "level": "path_level",
        "question": traj["question"],
        "context": traj.get("context", ""),
        "gold_answer": traj.get("gold_answer", ""),
        "prefix_index": 0,
        "prefix_text": prefix_text,
        "prefix_units": [prefix_text],
        "prefix_num_units": 1,
        "final_answer": traj.get("final_answer", ""),
    }
    return [row]


def build_step_level_prefixes(traj: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = traj.get("steps", [])
    steps = [clean_text(s) for s in steps if clean_text(s)]

    rows: List[Dict[str, Any]] = []
    acc: List[str] = []
    for i, step in enumerate(steps):
        acc.append(step)
        prefix_text = "\n".join(acc)
        rows.append(
            {
                "prefix_id": f'{traj["trajectory_id"]}_step_{i}',
                "sample_id": traj["sample_id"],
                "trajectory_id": traj["trajectory_id"],
                "dataset": traj["dataset"],
                "split": traj["split"],
                "task": traj["task"],
                "level": "step_level",
                "question": traj["question"],
                "context": traj.get("context", ""),
                "gold_answer": traj.get("gold_answer", ""),
                "prefix_index": i,
                "prefix_text": prefix_text,
                "prefix_units": acc.copy(),
                "prefix_num_units": len(acc),
                "final_answer": traj.get("final_answer", ""),
            }
        )
    return rows


def build_atom_level_prefixes(traj: Dict[str, Any]) -> List[Dict[str, Any]]:
    reasoning_text = clean_text(traj.get("reasoning_text", ""))
    atoms = atomize_text(reasoning_text)

    rows: List[Dict[str, Any]] = []
    acc: List[str] = []
    for i, atom in enumerate(atoms):
        acc.append(atom)
        prefix_text = "\n".join(acc)
        rows.append(
            {
                "prefix_id": f'{traj["trajectory_id"]}_atom_{i}',
                "sample_id": traj["sample_id"],
                "trajectory_id": traj["trajectory_id"],
                "dataset": traj["dataset"],
                "split": traj["split"],
                "task": traj["task"],
                "level": "atom_level",
                "question": traj["question"],
                "context": traj.get("context", ""),
                "gold_answer": traj.get("gold_answer", ""),
                "prefix_index": i,
                "prefix_text": prefix_text,
                "prefix_units": acc.copy(),
                "prefix_num_units": len(acc),
                "final_answer": traj.get("final_answer", ""),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="trajectory jsonl 文件")
    parser.add_argument("--output_dir", required=True, help="prefix 输出根目录")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    trajectories = read_jsonl(input_path)

    dataset_name = input_path.parent.name
    split = input_path.stem

    path_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    atom_rows: List[Dict[str, Any]] = []

    for traj in trajectories:
        path_rows.extend(build_path_level_prefixes(traj))
        step_rows.extend(build_step_level_prefixes(traj))
        atom_rows.extend(build_atom_level_prefixes(traj))

    write_jsonl(output_dir / "path_level" / dataset_name / f"{split}.jsonl", path_rows)
    write_jsonl(output_dir / "step_level" / dataset_name / f"{split}.jsonl", step_rows)
    write_jsonl(output_dir / "atom_level" / dataset_name / f"{split}.jsonl", atom_rows)

    print(f"Saved path-level prefixes: {len(path_rows)}")
    print(f"Saved step-level prefixes: {len(step_rows)}")
    print(f"Saved atom-level prefixes: {len(atom_rows)}")


if __name__ == "__main__":
    main()
