from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


@dataclass
class VerifierHFConfig:
    encoder_name: str = "distilbert-base-uncased"
    hidden_size: int = 256
    dropout: float = 0.1
    max_length: int = 512

    use_success_head: bool = True
    use_error_head: bool = False
    use_repair_head: bool = False

    num_error_labels: int = 0


class VerifierHFPCE(nn.Module):
    """
    Transformer-based Prefix Confidence Estimator.

    支持：
    - success head（二分类 / 概率）
    - error head（多类分类，可选）
    - repair head（二分类，可选）
    """

    def __init__(self, cfg: VerifierHFConfig):
        super().__init__()
        self.cfg = cfg

        self.hf_config = AutoConfig.from_pretrained(cfg.encoder_name)
        self.encoder = AutoModel.from_pretrained(
            cfg.encoder_name, config=self.hf_config
        )

        encoder_hidden = getattr(self.hf_config, "hidden_size", None)
        if encoder_hidden is None:
            encoder_hidden = getattr(self.hf_config, "dim", None)
        if encoder_hidden is None:
            raise ValueError("Cannot infer encoder hidden size from HF config.")

        self.proj = nn.Linear(encoder_hidden, cfg.hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)

        self.use_success_head = cfg.use_success_head
        self.use_error_head = cfg.use_error_head
        self.use_repair_head = cfg.use_repair_head

        if self.use_success_head:
            self.success_head = nn.Linear(cfg.hidden_size, 1)

        if self.use_error_head:
            if cfg.num_error_labels <= 0:
                raise ValueError(
                    "num_error_labels must be > 0 when use_error_head=True"
                )
            self.error_head = nn.Linear(cfg.hidden_size, cfg.num_error_labels)

        if self.use_repair_head:
            self.repair_head = nn.Linear(cfg.hidden_size, 1)

    def mean_pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-6)
        return summed / denom

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(outputs.last_hidden_state, attention_mask)
        h = self.proj(pooled)
        h = self.act(h)
        h = self.dropout(h)
        return h

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        success_labels: Optional[torch.Tensor] = None,
        error_labels: Optional[torch.Tensor] = None,
        repair_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h = self.encode(input_ids=input_ids, attention_mask=attention_mask)

        out: Dict[str, torch.Tensor] = {}
        total_loss = None

        if self.use_success_head:
            success_logits = self.success_head(h).squeeze(-1)
            out["success_logits"] = success_logits
            out["success_prob"] = torch.sigmoid(success_logits)

            if success_labels is not None:
                success_labels = success_labels.float()
                loss_fct = nn.BCEWithLogitsLoss()
                loss_success = loss_fct(success_logits, success_labels)
                out["loss_success"] = loss_success
                total_loss = (
                    loss_success if total_loss is None else total_loss + loss_success
                )

        if self.use_error_head:
            error_logits = self.error_head(h)
            out["error_logits"] = error_logits

            if error_labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss_error = loss_fct(error_logits, error_labels.long())
                out["loss_error"] = loss_error
                total_loss = (
                    loss_error if total_loss is None else total_loss + loss_error
                )

        if self.use_repair_head:
            repair_logits = self.repair_head(h).squeeze(-1)
            out["repair_logits"] = repair_logits
            out["repair_prob"] = torch.sigmoid(repair_logits)

            if repair_labels is not None:
                repair_labels = repair_labels.float()
                loss_fct = nn.BCEWithLogitsLoss()
                loss_repair = loss_fct(repair_logits, repair_labels)
                out["loss_repair"] = loss_repair
                total_loss = (
                    loss_repair if total_loss is None else total_loss + loss_repair
                )

        if total_loss is not None:
            out["loss"] = total_loss

        return out

    def info(self) -> Dict[str, Any]:
        return {
            "name": "VerifierHFPCE",
            "type": "transformer_multitask",
            "config": asdict(self.cfg),
        }

    def save_checkpoint(
        self,
        output_dir: str,
        tokenizer,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), out_dir / "pytorch_model.bin")
        tokenizer.save_pretrained(out_dir)

        payload = {
            "verifier_hf_config": asdict(self.cfg),
            "extra": extra or {},
        }
        with open(out_dir / "verifier_hf_meta.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_checkpoint(
        cls, output_dir: str, device: str = "cpu"
    ) -> tuple["VerifierHFPCE", Any, Dict[str, Any]]:
        out_dir = Path(output_dir)
        with open(out_dir / "verifier_hf_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        cfg = VerifierHFConfig(**meta["verifier_hf_config"])
        model = cls(cfg)
        state = torch.load(out_dir / "pytorch_model.bin", map_location=device)
        model.load_state_dict(state)

        tokenizer = AutoTokenizer.from_pretrained(out_dir)
        return model, tokenizer, meta.get("extra", {})

    @staticmethod
    def build_tokenizer(encoder_name: str):
        tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = (
                tokenizer.eos_token
                if tokenizer.eos_token is not None
                else tokenizer.unk_token
            )
        return tokenizer
