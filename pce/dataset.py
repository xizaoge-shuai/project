from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils.io import read_jsonl


LEVEL_ALIASES = {
    "path": "path_level",
    "step": "step_level",
    "atom": "atom_level",
    "path_level": "path_level",
    "step_level": "step_level",
    "atom_level": "atom_level",
}


def normalize_level(level: str) -> str:
    level = (level or "").strip().lower()
    if level not in LEVEL_ALIASES:
        raise ValueError(f"Unsupported level: {level}")
    return LEVEL_ALIASES[level]


def success_label_dir(
    dataset: str,
    level: str,
    base_dir: str = "data/processed/labels",
) -> Path:
    """
    返回 success 标签目录：
    data/processed/labels/success/{level}_level/{dataset}/
    """
    norm_level = normalize_level(level)
    return Path(base_dir) / "success" / norm_level / dataset


def discover_split_files(
    dataset: str,
    level: str,
    splits: Optional[Sequence[str]] = None,
    base_dir: str = "data/processed/labels",
) -> List[Path]:
    label_dir = success_label_dir(dataset=dataset, level=level, base_dir=base_dir)
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    if splits is None:
        return sorted(label_dir.glob("*.jsonl"))

    files: List[Path] = []
    for split in splits:
        path = label_dir / f"{split}.jsonl"
        if path.exists():
            files.append(path)
    return files


def load_pce_training_rows(
    dataset: str,
    level: str,
    splits: Optional[Sequence[str]] = None,
    base_dir: str = "data/processed/labels",
    max_rows: int = -1,
) -> List[Dict[str, Any]]:
    """
    加载 success 标签数据，用于训练 PCE。
    """
    files = discover_split_files(
        dataset=dataset, level=level, splits=splits, base_dir=base_dir
    )

    rows: List[Dict[str, Any]] = []
    for fp in files:
        part = read_jsonl(str(fp))
        for r in part:
            rr = dict(r)
            rr["__source_file"] = str(fp)
            rr["__split"] = fp.stem
            rows.append(rr)

    if max_rows > 0:
        rows = rows[:max_rows]

    return rows


def build_input_text(
    row: Dict[str, Any],
    include_task: bool = True,
    include_context: bool = True,
    include_question: bool = False,
    include_answer: bool = False,
    include_prefix_len: bool = True,
) -> str:
    pieces: List[str] = []

    if include_task:
        pieces.append(f"[TASK] {row.get('task', 'generic')}")

    if include_question:
        pieces.append(f"[QUESTION] {row.get('question', '')}")

    if include_context:
        context = row.get("context", "")
        if context:
            pieces.append(f"[CONTEXT] {context}")

    pieces.append(f"[PREFIX] {row.get('prefix_text', '')}")

    if include_answer:
        pieces.append(f"[CURRENT_ANSWER] {row.get('final_answer', '')}")

    if include_prefix_len:
        pieces.append(f"[PREFIX_LEN] {row.get('prefix_num_units', 0)}")

    return "\n".join(pieces).strip()


def build_text_label_pairs(
    rows: List[Dict[str, Any]],
    include_task: bool = True,
    include_context: bool = True,
    include_question: bool = False,
    include_answer: bool = False,
    include_prefix_len: bool = True,
) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []

    for r in rows:
        txt = build_input_text(
            r,
            include_task=include_task,
            include_context=include_context,
            include_question=include_question,
            include_answer=include_answer,
            include_prefix_len=include_prefix_len,
        )
        texts.append(txt)
        labels.append(int(r["label_success"]))

    return texts, labels


def build_text_label_meta(
    rows: List[Dict[str, Any]],
    include_task: bool = True,
    include_context: bool = True,
    include_question: bool = False,
    include_answer: bool = False,
    include_prefix_len: bool = True,
) -> Tuple[List[str], List[int], List[Dict[str, Any]]]:
    """
    返回 texts, labels, metadata，便于训练后保存预测结果。
    """
    texts, labels = build_text_label_pairs(
        rows,
        include_task=include_task,
        include_context=include_context,
        include_answer=include_answer,
        include_prefix_len=include_prefix_len,
    )

    metas: List[Dict[str, Any]] = []
    for r in rows:
        metas.append(
            {
                "prefix_id": r.get("prefix_id", ""),
                "sample_id": r.get("sample_id", ""),
                "trajectory_id": r.get("trajectory_id", ""),
                "dataset": r.get("dataset", ""),
                "split": r.get("split", r.get("__split", "")),
                "task": r.get("task", ""),
                "level": r.get("level", ""),
                "question": r.get("question", ""),
                "gold_answer": r.get("gold_answer", ""),
                "final_answer": r.get("final_answer", ""),
                "prefix_num_units": r.get("prefix_num_units", 0),
            }
        )

    return texts, labels, metas
