from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pce.dataset import build_text_label_meta, load_pce_training_rows
from pce.models.verifier_hf import VerifierHFPCE
from utils.io import read_jsonl


def parse_csv_splits(x: str) -> List[str]:
    x = (x or "").strip()
    if not x:
        return []
    return [p.strip() for p in x.split(",") if p.strip()]


def feature_set_to_kwargs(feature_set: str) -> Dict[str, bool]:
    if feature_set == "prefix_only":
        return {
            "include_task": True,
            "include_context": False,
            "include_question": False,
            "include_answer": False,
            "include_prefix_len": False,
            "include_prefix_progress": False,
        }
    if feature_set == "prefix_plus_len":
        return {
            "include_task": True,
            "include_context": False,
            "include_question": False,
            "include_answer": False,
            "include_prefix_len": True,
            "include_prefix_progress": False,
        }
    if feature_set == "prefix_plus_question":
        return {
            "include_task": True,
            "include_context": False,
            "include_question": True,
            "include_answer": False,
            "include_prefix_len": True,
            "include_prefix_progress": False,
        }
    if feature_set == "prefix_plus_question_answer":
        return {
            "include_task": True,
            "include_context": False,
            "include_question": True,
            "include_answer": True,
            "include_prefix_len": True,
            "include_prefix_progress": False,
        }
    if feature_set == "prefix_plus_len_progress":
        return {
            "include_task": True,
            "include_context": False,
            "include_question": False,
            "include_answer": False,
            "include_prefix_len": True,
            "include_prefix_progress": True,
        }
    raise ValueError(f"Unsupported feature_set: {feature_set}")


def load_light_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_prediction_rows(
    metas: List[Dict[str, Any]],
    labels: List[int],
    probs: List[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for meta, y, p in zip(metas, labels, probs):
        rows.append(
            {
                **meta,
                "label_success": int(y),
                "success_prob": float(p),
            }
        )
    return rows


def predict_success_probs_light(model: Any, texts: List[str]) -> List[float]:
    if hasattr(model, "predict_proba_texts"):
        probs = model.predict_proba_texts(texts)
        return [float(x) for x in probs]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(texts)
        probs = np.asarray(probs)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return [float(x) for x in probs[:, 1]]
        return [float(x) for x in probs.reshape(-1)]

    results: List[float] = []
    for txt in texts:
        out = model.predict(txt)
        if isinstance(out, dict) and "success_prob" in out:
            results.append(float(out["success_prob"]))
        else:
            results.append(float(out))
    return results


class HFBatchDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        metas: List[Dict[str, Any]],
        tokenizer,
        max_length: int,
    ):
        self.texts = texts
        self.labels = labels
        self.metas = metas
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label_success": int(self.labels[idx]),
            "meta": self.metas[idx],
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": [x["label_success"] for x in batch],
        "metas": [x["meta"] for x in batch],
    }


def move_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def predict_success_probs_hf(
    checkpoint_dir: str,
    texts: List[str],
    labels: List[int],
    metas: List[Dict[str, Any]],
    device: str,
    batch_size: int = 16,
) -> Tuple[List[float], List[int], List[Dict[str, Any]]]:
    model, tokenizer, _ = VerifierHFPCE.load_checkpoint(checkpoint_dir, device=device)
    model.to(device)
    model.eval()

    max_length = getattr(model.cfg, "max_length", 512)
    dataset = HFBatchDataset(texts, labels, metas, tokenizer, max_length=max_length)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    all_probs: List[float] = []
    all_labels: List[int] = []
    all_metas: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            raw_labels = batch["labels"]
            raw_metas = batch["metas"]

            batch = move_to_device(batch, device)
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            probs = out["success_prob"].detach().cpu().numpy().tolist()

            all_probs.extend([float(x) for x in probs])
            all_labels.extend([int(x) for x in raw_labels])
            all_metas.extend(raw_metas)

    return all_probs, all_labels, all_metas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, choices=["light", "hf"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset", required=True, choices=["gsm8k", "strategyqa", "hotpotqa"]
    )
    parser.add_argument(
        "--level",
        required=True,
        choices=["path", "step", "atom", "path_level", "step_level", "atom_level"],
    )
    parser.add_argument("--splits", required=True)
    parser.add_argument("--feature_set", default="prefix_plus_len_progress")
    parser.add_argument("--min_prefix_progress", type=float, default=0.0)
    parser.add_argument("--label_base_dir", default="data/processed/labels")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = load_pce_training_rows(
        dataset=args.dataset,
        level=args.level,
        splits=parse_csv_splits(args.splits),
        base_dir=args.label_base_dir,
        min_prefix_progress=args.min_prefix_progress,
    )
    if not rows:
        raise ValueError("No rows found for inference.")

    feature_kwargs = feature_set_to_kwargs(args.feature_set)
    texts, labels, metas = build_text_label_meta(rows, **feature_kwargs)

    if args.model_type == "light":
        model = load_light_model(args.checkpoint)
        probs = predict_success_probs_light(model, texts)
        out_rows = build_prediction_rows(metas, labels, probs)

    elif args.model_type == "hf":
        probs, labels_out, metas_out = predict_success_probs_hf(
            checkpoint_dir=args.checkpoint,
            texts=texts,
            labels=labels,
            metas=metas,
            device=args.device,
            batch_size=args.batch_size,
        )
        out_rows = build_prediction_rows(metas_out, labels_out, probs)

    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    save_jsonl(args.out, out_rows)
    print(f"Saved {len(out_rows)} predictions to {args.out}")


if __name__ == "__main__":
    main()
