from __future__ import annotations

import re


def _extract_last_number(text: str) -> str:
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text or "")
    return nums[-1] if nums else ""


def guess_error_type(question: str, prefix_text: str, pred: str, gold: str) -> str:
    q = (question or "").lower()
    p = (prefix_text or "").lower()

    pred_num = _extract_last_number(pred or "")
    gold_num = _extract_last_number(gold or "")

    # 数学题且当前答案和 gold 最终数值不一致
    if re.search(r"\d", q) and pred_num and gold_num and pred_num != gold_num:
        return "arithmetic_error"

    # 缺少推理支撑
    if "why" in q and "because" not in p:
        return "unsupported_jump"

    # 多跳 / context 依赖问题
    if "context" in q or "according to" in q:
        return "missing_evidence"

    return "unknown_error"


def heuristic_repairable(error_type: str, prefix_len: int) -> bool:
    if prefix_len <= 0:
        return False

    return error_type in {
        "arithmetic_error",
        "unsupported_jump",
        "missing_evidence",
    }
