from __future__ import annotations
import re

def guess_error_type(question: str, prefix_text: str, pred: str, gold: str) -> str:
    if re.search(r"\d", question) and re.search(r"\d", gold):
        return "arithmetic_error"
    if "because" not in prefix_text.lower() and "why" in question.lower():
        return "unsupported_jump"
    if "context" in question.lower():
        return "missing_evidence"
    return "unknown_error"

def heuristic_repairable(error_type: str, prefix_len: int) -> bool:
    return error_type in {"arithmetic_error", "unsupported_jump", "missing_evidence"} and prefix_len >= 1
