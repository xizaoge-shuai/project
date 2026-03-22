from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence


def batched(items: Sequence[Any], batch_size: int) -> List[List[Any]]:
    """
    将列表按 batch_size 切分。
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]


def simple_stop(steps: List[str], max_steps: int = 8) -> bool:
    """
    基于 step 数量的简单停止条件。
    """
    return len(steps) >= max_steps


def count_approx_tokens(text: str) -> int:
    """
    粗略 token 估计。
    不依赖 tokenizer，适合日志和开销估算。
    英文近似按词数；中英文混合场景下只是粗估。
    """
    text = (text or "").strip()
    if not text:
        return 0
    return max(1, math.ceil(len(text.split()) * 1.3))


def estimate_latency(tokens: int, speed_tps: float = 60.0) -> float:
    """
    用 tokens-per-second 近似推理耗时。
    """
    return round(tokens / max(speed_tps, 1e-6), 4)


def truncate_by_stop(text: str, stop: Optional[Iterable[str]] = None) -> str:
    """
    根据 stop sequences 截断文本。
    """
    if not text:
        return ""
    if stop is None:
        return text

    best_idx = None
    for s in stop:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1:
            if best_idx is None or idx < best_idx:
                best_idx = idx

    if best_idx is None:
        return text
    return text[:best_idx]


def build_generation_result(
    prompt: str,
    text: str,
    backend: str,
    finish_reason: str = "unknown",
    latency: float = 0.0,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    构造统一生成结果格式。
    """
    return {
        "prompt": prompt,
        "text": text,
        "finish_reason": finish_reason,
        "latency": float(latency),
        "backend": backend,
        "meta": meta or {},
    }


def timed_call_start() -> float:
    return time.time()


def timed_call_end(start_time: float) -> float:
    return time.time() - start_time


def safe_merge_generation_kwargs(
    base_kwargs: Dict[str, Any],
    override_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    合并生成参数，override 优先。
    """
    merged = dict(base_kwargs)
    if override_kwargs:
        merged.update(override_kwargs)
    return merged


def reached_budget(
    used_tokens: int,
    max_tokens: Optional[int] = None,
    used_steps: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> bool:
    """
    简单预算判断。
    """
    if max_tokens is not None and used_tokens >= max_tokens:
        return True
    if used_steps is not None and max_steps is not None and used_steps >= max_steps:
        return True
    return False
