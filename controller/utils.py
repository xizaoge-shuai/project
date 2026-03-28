from __future__ import annotations

from typing import Any, Dict


VALID_ACTIONS = ["continue", "prune", "backtrack", "accept"]


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0:
        return default
    return a / b


def get_state_attr(state: Any, name: str, default: Any = None) -> Any:
    return getattr(state, name, default)


def get_meta(state: Any, key: str, default: Any = None) -> Any:
    meta = get_state_attr(state, "meta", {}) or {}
    return meta.get(key, default)


def get_prefix_progress(state: Any, pce_output: Dict[str, Any]) -> float:
    # 优先从 pce_output 拿
    if "prefix_progress" in pce_output:
        return clamp(float(pce_output["prefix_progress"]), 0.0, 1.0)

    # 再从 state.meta 拿
    meta_progress = get_meta(state, "prefix_progress", None)
    if meta_progress is not None:
        return clamp(float(meta_progress), 0.0, 1.0)

    # 再尝试从 prefix 长度 / 总长度推
    prefix_items = get_state_attr(state, "prefix_items", []) or []
    prefix_len = len(prefix_items)

    total_units = (
        get_meta(state, "trajectory_total_units", None)
        or get_meta(state, "prefix_total_units", None)
        or get_meta(state, "total_steps", None)
        or prefix_len
    )
    return clamp(safe_div(prefix_len, max(int(total_units), 1)), 0.0, 1.0)


def get_budget_ratio(state: Any) -> float:
    tokens_left = float(get_state_attr(state, "tokens_left", 0.0) or 0.0)
    budget_tokens = float(get_state_attr(state, "budget_tokens", 0.0) or 0.0)

    if budget_tokens <= 0:
        # fallback
        tokens_used = float(get_state_attr(state, "tokens_used", 0.0) or 0.0)
        total = tokens_left + tokens_used
        if total <= 0:
            return 1.0
        return clamp(tokens_left / total, 0.0, 1.0)

    return clamp(tokens_left / budget_tokens, 0.0, 1.0)


def get_backtrack_ratio(state: Any) -> float:
    count = float(get_state_attr(state, "backtrack_count", 0.0) or 0.0)
    max_bt = float(get_meta(state, "max_backtracks", 2.0) or 2.0)
    return clamp(safe_div(count, max_bt, 0.0), 0.0, 1.0)


def extract_controller_features(
    state: Any, pce_output: Dict[str, Any]
) -> Dict[str, float]:
    conf = clamp(float(pce_output.get("success_prob", 0.0)), 0.0, 1.0)
    uncertainty = clamp(float(pce_output.get("uncertainty", 1.0 - conf)), 0.0, 1.0)
    repairable = 1.0 if int(pce_output.get("repairable", 0)) > 0 else 0.0
    progress = get_prefix_progress(state, pce_output)
    budget = get_budget_ratio(state)
    backtrack_ratio = get_backtrack_ratio(state)

    return {
        "conf": conf,
        "uncertainty": uncertainty,
        "repairable": repairable,
        "progress": progress,
        "budget": budget,
        "backtrack_ratio": backtrack_ratio,
        "not_repairable": 1.0 - repairable,
        "low_budget": 1.0 if budget < 0.2 else 0.0,
        "near_end": 1.0 if progress > 0.85 else 0.0,
        "early_stage": 1.0 if progress < 0.25 else 0.0,
    }
