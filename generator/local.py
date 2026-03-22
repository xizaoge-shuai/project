from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional

from generator.base import BaseGenerator


class LocalGenerator(BaseGenerator):
    """
    正式版本地生成器：
    - backend = "vllm" 时走 vLLM 离线批量推理
    - backend = "transformers" 时走 HuggingFace fallback
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = copy.deepcopy(config)
        self.backend = self.config.get("backend", "vllm").lower()
        self.return_full_text = bool(self.config.get("return_full_text", False))
        self.batch_size = int(self.config.get("batch_size", 8))

        if self.backend == "vllm":
            self._init_vllm()
        elif self.backend == "transformers":
            self._init_transformers()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _init_vllm(self) -> None:
        from vllm import LLM

        self._SamplingParams = self._import_sampling_params()

        self.model = LLM(
            model=self.config["model_name_or_path"],
            tensor_parallel_size=int(self.config.get("tensor_parallel_size", 1)),
            max_model_len=int(self.config.get("max_model_len", 4096)),
            gpu_memory_utilization=float(
                self.config.get("gpu_memory_utilization", 0.9)
            ),
            dtype=self.config.get("dtype", "auto"),
            trust_remote_code=bool(self.config.get("trust_remote_code", True)),
            enforce_eager=bool(self.config.get("enforce_eager", False)),
        )
        self.tokenizer = None

    def _init_transformers(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = self.config["model_name_or_path"]
        use_fast = bool(self.config.get("use_fast_tokenizer", True))
        trust_remote_code = bool(self.config.get("trust_remote_code", True))
        device_map = self.config.get("device_map", "auto")
        torch_dtype_cfg = self.config.get("torch_dtype", "auto")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if torch_dtype_cfg == "auto":
            torch_dtype = "auto"
        else:
            torch_dtype = getattr(torch, torch_dtype_cfg)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        ).eval()

        self._SamplingParams = None

    def _import_sampling_params(self):
        from vllm import SamplingParams

        return SamplingParams

    def _build_sampling_params(self, override: Optional[Dict[str, Any]] = None):
        """
        vLLM SamplingParams。
        不默认暴露 best_of，避免踩当前版本兼容坑。
        """
        cfg = copy.deepcopy(self.config)
        if override:
            cfg.update(override)

        stop = cfg.get("stop", None)
        if stop is not None and not isinstance(stop, list):
            stop = [stop]

        params = self._SamplingParams(
            n=int(cfg.get("n", 1)),
            temperature=float(cfg.get("temperature", 0.7)),
            top_p=float(cfg.get("top_p", 0.95)),
            top_k=int(cfg.get("top_k", -1)),
            repetition_penalty=float(cfg.get("repetition_penalty", 1.0)),
            presence_penalty=float(cfg.get("presence_penalty", 0.0)),
            frequency_penalty=float(cfg.get("frequency_penalty", 0.0)),
            max_tokens=int(cfg.get("max_new_tokens", 256)),
            stop=stop,
            seed=cfg.get("seed", None),
        )
        return params

    def _build_hf_generate_kwargs(
        self, override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        cfg = copy.deepcopy(self.config)
        if override:
            cfg.update(override)

        temperature = float(cfg.get("temperature", 0.7))
        do_sample = temperature > 0

        kwargs = {
            "max_new_tokens": int(cfg.get("max_new_tokens", 256)),
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": float(cfg.get("top_p", 0.95)),
            "top_k": int(cfg.get("top_k", 50 if do_sample else 0)),
            "repetition_penalty": float(cfg.get("repetition_penalty", 1.0)),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        return kwargs

    def _chunk(self, items: List[str], chunk_size: int) -> List[List[str]]:
        return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    def generate_one(self, prompt: str, **kwargs) -> Dict[str, Any]:
        outputs = self.generate_many([prompt], **kwargs)
        return outputs[0]

    def generate_many(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        if self.backend == "vllm":
            return self._generate_many_vllm(prompts, **kwargs)
        return self._generate_many_transformers(prompts, **kwargs)

    def _generate_many_vllm(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        sampling_params = self._build_sampling_params(kwargs)
        started = time.time()
        raw_outputs = self.model.generate(prompts, sampling_params)
        latency = time.time() - started

        results: List[Dict[str, Any]] = []
        for output in raw_outputs:
            if not output.outputs:
                text = ""
                finish_reason = "empty"
            else:
                first = output.outputs[0]
                text = first.text
                finish_reason = getattr(first, "finish_reason", "unknown")

            if self.return_full_text:
                final_text = (output.prompt or "") + text
            else:
                final_text = text

            results.append(
                {
                    "prompt": output.prompt,
                    "text": final_text,
                    "finish_reason": finish_reason,
                    "latency": latency / max(len(raw_outputs), 1),
                    "backend": "vllm",
                    "meta": {
                        "num_candidates": len(output.outputs),
                    },
                }
            )
        return results

    def _generate_many_transformers(
        self, prompts: List[str], **kwargs
    ) -> List[Dict[str, Any]]:
        import torch

        gen_kwargs = self._build_hf_generate_kwargs(kwargs)
        results: List[Dict[str, Any]] = []

        for batch_prompts in self._chunk(prompts, self.batch_size):
            started = time.time()
            model_inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}

            with torch.no_grad():
                generated = self.model.generate(**model_inputs, **gen_kwargs)

            latency = time.time() - started

            input_len = model_inputs["input_ids"].shape[1]
            decoded_all = self.tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )

            if self.return_full_text:
                batch_texts = decoded_all
            else:
                gen_only = generated[:, input_len:]
                batch_texts = self.tokenizer.batch_decode(
                    gen_only, skip_special_tokens=True
                )

            for prompt, text in zip(batch_prompts, batch_texts):
                results.append(
                    {
                        "prompt": prompt,
                        "text": text,
                        "finish_reason": "length_or_eos",
                        "latency": latency / max(len(batch_prompts), 1),
                        "backend": "transformers",
                        "meta": {},
                    }
                )
        return results
