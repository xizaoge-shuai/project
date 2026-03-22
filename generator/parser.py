from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional


TaskType = Literal["generic", "gsm8k", "strategyqa", "hotpotqa"]


FINAL_ANSWER_PATTERNS = [
    r"Final Answer\s*:\s*(.+)",
    r"Answer\s*:\s*(.+)",
    r"Therefore,?\s+the answer is\s+(.+)",
    r"So,?\s+the answer is\s+(.+)",
    r"The answer is\s+(.+)",
]

STEP_PATTERNS = [
    r"(?m)^\s*Step\s*\d+[\):.\-]?\s*",
    r"(?m)^\s*\d+[\).\-]\s+",
    r"(?m)^\s*[-*]\s+",
]


def clean_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return text


def normalize_yes_no(answer: str) -> str:
    x = clean_text(answer).lower().strip(" .,:;!?\"'`()[]{}")
    if x in {"yes", "true", "correct"}:
        return "yes"
    if x in {"no", "false", "incorrect"}:
        return "no"

    if x.startswith("yes"):
        return "yes"
    if x.startswith("no"):
        return "no"
    return x


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    提取 \\boxed{...} 或 boxed{...} 中的答案。
    """
    patterns = [
        r"\\boxed\{([^{}]+)\}",
        r"boxed\{([^{}]+)\}",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return clean_text(m.group(1))
    return None


def extract_last_number(text: str) -> Optional[str]:
    """
    用于数学题兜底：提取最后一个数字/分数/百分数/小数。
    """
    candidates = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?|[-+]?\d+/\d+", text)
    if not candidates:
        return None
    return candidates[-1].replace(",", "")


def normalize_math_answer(answer: str) -> str:
    """
    对 GSM8K 一类答案做轻量规范化。
    不做过度语义推断，只做安全清洗。
    """
    answer = clean_text(answer)

    # 去掉 markdown/latex 包裹
    answer = answer.strip("`$ ")
    answer = answer.replace(",", "")

    # 去掉句尾标点
    answer = answer.strip(" .,:;!?")

    # 常见前缀
    answer = re.sub(r"^(the answer is)\s+", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"^(therefore|thus|so)\s*,?\s*", "", answer, flags=re.IGNORECASE)

    return answer.strip()


def normalize_generic_answer(answer: str) -> str:
    answer = clean_text(answer)
    answer = answer.strip("`$ ")
    answer = answer.strip(" .,:;!?")
    return answer


def normalize_answer(answer: str, task: TaskType = "generic") -> str:
    if task == "strategyqa":
        return normalize_yes_no(answer)
    if task == "gsm8k":
        return normalize_math_answer(answer)
    return normalize_generic_answer(answer)


def remove_final_answer_line(text: str) -> str:
    """
    删除最后答案行，保留 reasoning 主体。
    """
    lines = text.splitlines()
    kept = []
    for line in lines:
        if re.match(r"^\s*Final Answer\s*:", line, flags=re.IGNORECASE):
            continue
        kept.append(line)
    return clean_text("\n".join(kept))


def extract_final_answer(text: str, task: TaskType = "generic") -> str:
    """
    优先级：
    1. Final Answer:
    2. boxed{}
    3. 其他常见 answer pattern
    4. 数学题取最后一个数字
    5. 最后一行兜底
    """
    text = strip_code_fences(clean_text(text))

    # 1) 标准 Final Answer 格式
    for p in FINAL_ANSWER_PATTERNS:
        matches = re.findall(p, text, flags=re.IGNORECASE)
        if matches:
            raw = clean_text(matches[-1])
            return normalize_answer(raw, task=task)

    # 2) boxed 答案
    boxed = extract_boxed_answer(text)
    if boxed:
        return normalize_answer(boxed, task=task)

    # 3) strategyqa 强制 yes/no 兜底
    if task == "strategyqa":
        yn = normalize_yes_no(text)
        if yn in {"yes", "no"}:
            return yn

    # 4) gsm8k 数字兜底
    if task == "gsm8k":
        num = extract_last_number(text)
        if num is not None:
            return normalize_answer(num, task=task)

    # 5) 最后一行兜底
    lines = [clean_text(x) for x in text.splitlines() if clean_text(x)]
    if lines:
        return normalize_answer(lines[-1], task=task)

    return ""


def split_reasoning_steps(text: str) -> List[str]:
    """
    将模型输出切成 reasoning steps。
    这是通用 step-level 切分，不等同于 atom-level。
    """
    text = strip_code_fences(clean_text(text))
    text_wo_answer = remove_final_answer_line(text)

    if not text_wo_answer:
        return []

    # 优先按显式 Step / 编号 / 列表切分
    has_explicit_step = any(
        re.search(p, text_wo_answer, flags=re.IGNORECASE) for p in STEP_PATTERNS
    )
    if has_explicit_step:
        parts = re.split(
            r"(?m)^\s*(?:Step\s*\d+[\):.\-]?|\d+[\).\-]|[-*])\s+",
            text_wo_answer,
            flags=re.IGNORECASE,
        )
        steps = [clean_text(p) for p in parts if clean_text(p)]
        if steps:
            return steps

    # 否则按换行切分
    lines = [clean_text(x) for x in text_wo_answer.split("\n") if clean_text(x)]
    if len(lines) > 1:
        return lines

    # 再否则按句子切分
    sents = re.split(r"(?<=[.!?。！？])\s+", text_wo_answer)
    sents = [clean_text(s) for s in sents if clean_text(s)]
    if sents:
        return sents

    return [text_wo_answer]


def extract_reasoning_text(text: str) -> str:
    """
    提取除 Final Answer 外的推理正文。
    """
    text = strip_code_fences(clean_text(text))
    return remove_final_answer_line(text)


def parse_generation_output(
    text: str,
    task: TaskType = "generic",
) -> Dict[str, Any]:
    """
    解析普通 CoT / repair / backtrack 生成结果。
    返回统一结构，方便后续 build_trajectories.py / pipeline.py 使用。
    """
    text = strip_code_fences(clean_text(text))
    reasoning_text = extract_reasoning_text(text)
    steps = split_reasoning_steps(text)
    final_answer = extract_final_answer(text, task=task)

    return {
        "raw_text": text,
        "reasoning_text": reasoning_text,
        "steps": steps,
        "num_steps": len(steps),
        "final_answer": final_answer,
    }


def parse_multiple_candidates(
    outputs: List[Dict[str, Any]],
    task: TaskType = "generic",
) -> List[Dict[str, Any]]:
    """
    对 generate_many 的返回结果做统一解析。
    假设每个元素至少包含 text 字段。
    """
    parsed = []
    for item in outputs:
        text = item.get("text", "")
        obj = parse_generation_output(text, task=task)
        merged = dict(item)
        merged.update(obj)
        parsed.append(merged)
    return parsed


def parse_diagnosis_output(text: str) -> Dict[str, str]:
    """
    配套 build_error_diagnosis_prompt() 使用。
    期望格式：
      Judgment: correct / incorrect
      Error Type: arithmetic_error / ...
      Explanation: ...
    """
    text = strip_code_fences(clean_text(text))

    judgment = ""
    error_type = ""
    explanation = ""

    m = re.search(r"Judgment\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        judgment = clean_text(m.group(1)).lower()

    m = re.search(r"Error Type\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        error_type = clean_text(m.group(1)).lower()

    m = re.search(r"Explanation\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        explanation = clean_text(m.group(1))

    # 兜底规则
    if not judgment:
        lower = text.lower()
        if "incorrect" in lower:
            judgment = "incorrect"
        elif "correct" in lower:
            judgment = "correct"

    if not error_type:
        lower = text.lower()
        candidate_types = [
            "arithmetic_error",
            "contradiction",
            "unsupported_jump",
            "missing_evidence",
            "premature_conclusion",
            "none",
        ]
        for t in candidate_types:
            if t in lower:
                error_type = t
                break

    return {
        "judgment": judgment,
        "error_type": error_type,
        "explanation": explanation,
        "raw_text": text,
    }


def parse_final_answer_only(text: str, task: TaskType = "generic") -> Dict[str, str]:
    """
    给 direct-answer baseline 用。
    """
    text = strip_code_fences(clean_text(text))
    answer = extract_final_answer(text, task=task)
    return {
        "raw_text": text,
        "final_answer": answer,
    }
