from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple


HARD_BAD_PATTERNS = [
    "Human:",
    "Assistant:",
    "Error type hint:",
    "Corrected continuation only:",
    "Given the revised reasoning trace",
    "identify and correct",
]


def truncate_generation_noise(text: str) -> str:
    """
    截断模型把 prompt 模板、Human/Assistant 对话头继续生成出来的情况。
    """
    s = str(text or "")
    cut_marks = [
        "\nHuman:",
        "\nAssistant:",
        "\nError type hint:",
        "\nCorrected continuation only:",
        "Human:",
        "Assistant:",
        "Error type hint:",
        "Corrected continuation only:",
    ]
    best = len(s)
    for m in cut_marks:
        idx = s.find(m)
        if idx >= 0:
            best = min(best, idx)
    return s[:best].strip()


def normalize_num(x: str) -> str:
    x = str(x or "").strip().replace(",", "").replace("$", "")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", x)
    if not nums:
        return ""
    v = nums[-1]
    if "." in v:
        v = v.rstrip("0").rstrip(".")
    return v


def extract_strict_final_answer(text: str) -> Optional[str]:
    """
    只接受严格格式：
      最后一条非空行必须是 Final Answer: <number>

    允许文本中只有一个 Final Answer。
    不允许 Human/Assistant/prompt 泄漏。
    """
    if text is None:
        return None

    s = truncate_generation_noise(str(text))
    if not s:
        return None

    for bad in HARD_BAD_PATTERNS:
        if bad in s:
            return None

    final_tags = re.findall(r"Final Answer\s*:", s, flags=re.IGNORECASE)
    if len(final_tags) != 1:
        return None

    lines = [x.strip() for x in s.splitlines() if x.strip()]
    if not lines:
        return None

    last = lines[-1].strip()
    m = re.fullmatch(
        r"Final Answer\s*:\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*",
        last,
        flags=re.IGNORECASE,
    )
    if not m:
        return None

    ans = m.group(1).replace(",", "")
    if "." in ans:
        ans = ans.rstrip("0").rstrip(".")
    return ans


def has_valid_local_context(valid_prefix: str, suspicious_suffix: str) -> bool:
    vp = str(valid_prefix or "").strip()
    ss = str(suspicious_suffix or "").strip()

    if not vp or vp == "[EMPTY]":
        return False
    if not ss or ss == "[EMPTY]":
        return False
    return True


def split_steps_for_repair(
    steps: List[str],
    trigger_progress: Optional[float],
    rewrite_window: int = 1,
) -> Tuple[List[str], List[str], int]:
    """
    根据 trigger_progress 从完整 steps 中切出局部修复窗口。
    """
    steps = [str(x).strip() for x in (steps or []) if str(x).strip()]
    n = len(steps)
    if n == 0:
        return [], [], -1

    try:
        p = float(trigger_progress)
    except Exception:
        p = 1.0

    p = max(0.0, min(1.0, p))

    trigger_idx = int(math.ceil(p * n)) - 1
    trigger_idx = max(0, min(n - 1, trigger_idx))

    w = max(1, int(rewrite_window))
    suffix_start = max(0, trigger_idx - w + 1)
    suffix_end = trigger_idx + 1

    valid_prefix_steps = steps[:suffix_start]
    suspicious_suffix_steps = steps[suffix_start:suffix_end]

    return valid_prefix_steps, suspicious_suffix_steps, trigger_idx


def format_steps(steps: List[str]) -> str:
    steps = [str(x).strip() for x in (steps or []) if str(x).strip()]
    if not steps:
        return "[EMPTY]"
    return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))


def build_safe_repair_prompt(
    question: str,
    valid_prefix_steps: List[str],
    suspicious_suffix_steps: List[str],
    current_answer: str,
    error_type_hint: str = "unknown_error",
) -> Optional[str]:
    valid_prefix = format_steps(valid_prefix_steps)
    suspicious_suffix = format_steps(suspicious_suffix_steps)

    if not has_valid_local_context(valid_prefix, suspicious_suffix):
        return None

    return f"""You are repairing a math reasoning trace.

This is a LOCAL REPAIR task.

Rules:
1. Keep the valid prefix unchanged.
2. Rewrite only the suspicious suffix.
3. Do not restart the whole solution.
4. Do not copy the prompt.
5. Do not output Human: or Assistant:.
6. Do not output more than one Final Answer line.
7. The last non-empty line must be exactly:
Final Answer: <number>

Question:
{question}

Valid prefix:
{valid_prefix}

Suspicious suffix:
{suspicious_suffix}

Current answer:
{current_answer}

Error type hint:
{error_type_hint}

Now produce the corrected local continuation.
The final line must be exactly:
Final Answer: <number>
"""


def safe_repair_gate(
    repair_output: str,
    original_answer: str,
) -> Dict[str, Any]:
    cleaned = truncate_generation_noise(repair_output)
    ans = extract_strict_final_answer(cleaned)

    if ans is None:
        return {
            "action": "KEEP",
            "final_answer": original_answer,
            "cleaned_output": cleaned,
            "reason": "invalid_or_malformed_repair_output",
        }

    return {
        "action": "REWRITE",
        "final_answer": ans,
        "cleaned_output": cleaned,
        "reason": "valid_strict_final_answer",
    }
