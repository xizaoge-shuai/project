from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from pce.dataset import build_text_label_meta, load_pce_training_rows
from pce.evaluate import compute_metrics
from pce.models.verifier_hf import VerifierHFConfig, VerifierHFPCE
from utils.io import ensure_dir, read_jsonl, read_yaml


def parse_csv_splits(x: str) -> List[str]:
    x = (x or "").strip()
    if not x:
        return []
    return [p.strip() for p in x.split(",") if p.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def load_aux_label_map(
    base_dir: str,
    label_kind: str,
    dataset: str,
    level: str,
    splits: List[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    level_dir = Path(base_dir) / label_kind / level / dataset
    for split in splits:
        fp = level_dir / f"{split}.jsonl"
        if not fp.exists():
            continue
        rows = read_jsonl(str(fp))
        for r in rows:
            out[r["prefix_id"]] = r
    return out


class HFPrefixDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        success_labels: List[int],
        metas: List[Dict[str, Any]],
        tokenizer,
        max_length: int,
        error_label_map: Optional[Dict[str, Any]] = None,
        repair_label_map: Optional[Dict[str, Any]] = None,
        error_type_to_id: Optional[Dict[str, int]] = None,
        use_error_head: bool = False,
        use_repair_head: bool = False,
    ):
        self.texts = texts
        self.success_labels = success_labels
        self.metas = metas
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.error_label_map = error_label_map or {}
        self.repair_label_map = repair_label_map or {}
        self.error_type_to_id = error_type_to_id or {}
        self.use_error_head = use_error_head
        self.use_repair_head = use_repair_head

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        meta = self.metas[idx]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item: Dict[str, Any] = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "success_labels": torch.tensor(
                float(self.success_labels[idx]), dtype=torch.float
            ),
            "meta": meta,
        }

        if self.use_error_head:
            row = self.error_label_map.get(meta["prefix_id"], None)
            if row is None:
                item["error_labels"] = torch.tensor(-100, dtype=torch.long)
            else:
                error_type = row.get("error_type", "none")
                item["error_labels"] = torch.tensor(
                    self.error_type_to_id.get(error_type, -100), dtype=torch.long
                )

        if self.use_repair_head:
            row = self.repair_label_map.get(meta["prefix_id"], None)
            if row is None:
                item["repair_labels"] = torch.tensor(0.0, dtype=torch.float)
            else:
                item["repair_labels"] = torch.tensor(
                    float(row.get("repairability", 0)), dtype=torch.float
                )

        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )

    out: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "success_labels": torch.stack([x["success_labels"] for x in batch]),
        "meta": [x["meta"] for x in batch],
    }

    if "error_labels" in batch[0]:
        out["error_labels"] = torch.stack([x["error_labels"] for x in batch])

    if "repair_labels" in batch[0]:
        out["repair_labels"] = torch.stack([x["repair_labels"] for x in batch])

    return out


def move_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def predict_success_probs(
    model: VerifierHFPCE,
    dataloader: DataLoader,
    device: str,
) -> Tuple[List[float], List[int], List[Dict[str, Any]]]:
    model.eval()
    all_probs: List[float] = []
    all_labels: List[int] = []
    all_metas: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in dataloader:
            metas = batch["meta"]
            labels = batch["success_labels"].cpu().numpy().tolist()

            batch = move_to_device(batch, device)
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            probs = out["success_prob"].detach().cpu().numpy().tolist()

            all_probs.extend([float(x) for x in probs])
            all_labels.extend([int(x) for x in labels])
            all_metas.extend(metas)

    return all_probs, all_labels, all_metas


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model/pce_bert.yaml")
    parser.add_argument(
        "--dataset", required=True, choices=["gsm8k", "strategyqa", "hotpotqa"]
    )
    parser.add_argument(
        "--level",
        default="step",
        choices=["path", "step", "atom", "path_level", "step_level", "atom_level"],
    )
    parser.add_argument("--label_base_dir", default="data/processed/labels")
    parser.add_argument("--train_splits", default="train")
    parser.add_argument("--val_splits", default="validation,val")
    parser.add_argument("--test_splits", default="test")
    parser.add_argument(
        "--feature_set",
        default="prefix_plus_len_progress",
        choices=[
            "prefix_only",
            "prefix_plus_len",
            "prefix_plus_question",
            "prefix_plus_question_answer",
            "prefix_plus_len_progress",
        ],
    )
    parser.add_argument("--min_prefix_progress", type=float, default=0.67)

    parser.add_argument("--out", default="outputs/checkpoints/pce_hf")
    parser.add_argument("--metrics_out", default="outputs/metrics/pce_hf_metrics.json")
    parser.add_argument(
        "--test_pred_out", default="outputs/predictions/pce_hf_test_predictions.jsonl"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = read_yaml(args.config)

    feature_kwargs = feature_set_to_kwargs(args.feature_set)
    level_norm = {"path": "path_level", "step": "step_level", "atom": "atom_level"}.get(
        args.level, args.level
    )

    train_rows = load_pce_training_rows(
        dataset=args.dataset,
        level=args.level,
        splits=parse_csv_splits(args.train_splits),
        base_dir=args.label_base_dir,
        min_prefix_progress=args.min_prefix_progress,
    )
    test_rows = load_pce_training_rows(
        dataset=args.dataset,
        level=args.level,
        splits=parse_csv_splits(args.test_splits),
        base_dir=args.label_base_dir,
        min_prefix_progress=args.min_prefix_progress,
    )

    if not train_rows:
        raise ValueError("No training rows found.")
    if not test_rows:
        raise ValueError("No test rows found.")

    train_texts, train_labels, train_metas = build_text_label_meta(
        train_rows, **feature_kwargs
    )
    test_texts, test_labels, test_metas = build_text_label_meta(
        test_rows, **feature_kwargs
    )

    use_success_head = bool(cfg.get("use_success_head", True))
    use_error_head = bool(cfg.get("use_error_head", False))
    use_repair_head = bool(cfg.get("use_repair_head", False))

    # 读取辅助标签
    error_train_map = {}
    error_test_map = {}
    repair_train_map = {}
    repair_test_map = {}
    error_type_to_id: Dict[str, int] = {}

    if use_error_head:
        error_train_map = load_aux_label_map(
            base_dir=args.label_base_dir,
            label_kind="error_type",
            dataset=args.dataset,
            level=level_norm,
            splits=parse_csv_splits(args.train_splits),
        )
        error_test_map = load_aux_label_map(
            base_dir=args.label_base_dir,
            label_kind="error_type",
            dataset=args.dataset,
            level=level_norm,
            splits=parse_csv_splits(args.test_splits),
        )

        observed = set()
        for row in error_train_map.values():
            observed.add(row.get("error_type", "none"))
        observed = sorted(observed)
        error_type_to_id = {name: i for i, name in enumerate(observed)}

        if not error_type_to_id:
            use_error_head = False

    if use_repair_head:
        repair_train_map = load_aux_label_map(
            base_dir=args.label_base_dir,
            label_kind="repairability",
            dataset=args.dataset,
            level=level_norm,
            splits=parse_csv_splits(args.train_splits),
        )
        repair_test_map = load_aux_label_map(
            base_dir=args.label_base_dir,
            label_kind="repairability",
            dataset=args.dataset,
            level=level_norm,
            splits=parse_csv_splits(args.test_splits),
        )

        if not repair_train_map:
            use_repair_head = False

    hf_cfg = VerifierHFConfig(
        encoder_name=str(cfg.get("encoder_name", "distilbert-base-uncased")),
        hidden_size=int(cfg.get("hidden_size", 256)),
        dropout=float(cfg.get("dropout", 0.1)),
        max_length=int(cfg.get("max_length", 512)),
        use_success_head=use_success_head,
        use_error_head=use_error_head,
        use_repair_head=use_repair_head,
        num_error_labels=len(error_type_to_id),
    )

    tokenizer = VerifierHFPCE.build_tokenizer(hf_cfg.encoder_name)

    train_dataset = HFPrefixDataset(
        texts=train_texts,
        success_labels=train_labels,
        metas=train_metas,
        tokenizer=tokenizer,
        max_length=hf_cfg.max_length,
        error_label_map=error_train_map,
        repair_label_map=repair_train_map,
        error_type_to_id=error_type_to_id,
        use_error_head=use_error_head,
        use_repair_head=use_repair_head,
    )
    test_dataset = HFPrefixDataset(
        texts=test_texts,
        success_labels=test_labels,
        metas=test_metas,
        tokenizer=tokenizer,
        max_length=hf_cfg.max_length,
        error_label_map=error_test_map,
        repair_label_map=repair_test_map,
        error_type_to_id=error_type_to_id,
        use_error_head=use_error_head,
        use_repair_head=use_repair_head,
    )

    batch_size = int(cfg.get("batch_size", 16))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    model = VerifierHFPCE(hf_cfg).to(args.device)

    lr = float(cfg.get("lr", 1e-4))
    epochs = int(cfg.get("epochs", 3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_steps = epochs * max(len(train_loader), 1)
    warmup_ratio = float(cfg.get("warmup_ratio", 0.1))
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            batch = move_to_device(batch, args.device)

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                success_labels=batch["success_labels"] if use_success_head else None,
                error_labels=(
                    batch.get("error_labels", None) if use_error_head else None
                ),
                repair_labels=(
                    batch.get("repair_labels", None) if use_repair_head else None
                ),
            )

            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch + 1}/{epochs} - train_loss: {avg_loss:.6f}")

    test_probs, test_gold, test_meta = predict_success_probs(
        model, test_loader, args.device
    )
    test_metrics = compute_metrics(test_gold, test_probs)

    summary: Dict[str, Any] = {
        "dataset": args.dataset,
        "level": args.level,
        "n_train": len(train_texts),
        "n_test": len(test_texts),
        "train_positive_rate": float(np.mean(train_labels)) if train_labels else None,
        "test_positive_rate": float(np.mean(test_labels)) if test_labels else None,
        "seed": args.seed,
        "config": cfg,
        "feature_set": args.feature_set,
        "min_prefix_progress": args.min_prefix_progress,
        "model_info": model.info(),
        "test": test_metrics,
    }

    pred_rows = build_prediction_rows(test_meta, test_gold, test_probs)

    out_dir = Path(args.out)
    ensure_dir(out_dir)
    model.save_checkpoint(
        output_dir=str(out_dir),
        tokenizer=tokenizer,
        extra={
            "error_type_to_id": error_type_to_id,
            "feature_set": args.feature_set,
            "min_prefix_progress": args.min_prefix_progress,
        },
    )

    save_json(Path(args.metrics_out), summary)
    save_jsonl(Path(args.test_pred_out), pred_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved HF verifier checkpoint to {out_dir}")


if __name__ == "__main__":
    main()
