from __future__ import annotations
import re
from collections import Counter
from typing import Iterable

def normalize_answer(ans: str) -> str:
    ans = (ans or "").strip().lower()
    ans = ans.replace(",", "")
    return ans

def extract_last_number(text: str) -> str:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    return nums[-1] if nums else ""

def is_correct_prediction(pred: str, gold: str, answer_mode: str = "numeric") -> bool:
    pred_n = normalize_answer(pred)
    gold_n = normalize_answer(gold)
    if answer_mode == "numeric":
        return extract_last_number(pred_n) == extract_last_number(gold_n)
    return pred_n == gold_n

def majority_vote(items: Iterable[str]) -> str:
    items = [i for i in items if i is not None]
    if not items:
        return ""
    return Counter(items).most_common(1)[0][0]
