from __future__ import annotations

from typing import Literal, Optional


TaskType = Literal["generic", "gsm8k", "strategyqa", "hotpotqa"]


def _clean_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return text.strip()


def _format_context(context: Optional[str]) -> str:
    context = _clean_text(context)
    if not context:
        return ""
    return f"Context:\n{context}\n\n"


def _task_instruction(task: TaskType) -> str:
    if task == "gsm8k":
        return (
            "Solve the math word problem carefully. "
            "Show concise step-by-step reasoning. "
            "Check arithmetic consistency before giving the final answer."
        )
    if task == "strategyqa":
        return (
            "Answer the question using careful step-by-step reasoning. "
            "The final answer must be either yes or no."
        )
    if task == "hotpotqa":
        return (
            "Answer the multi-hop question using the provided context. "
            "Reason step by step, use only relevant evidence, "
            "and avoid unsupported claims."
        )
    return (
        "Solve the problem carefully with step-by-step reasoning. "
        "Avoid unsupported jumps and keep the reasoning concise but complete."
    )


def _final_answer_instruction(task: TaskType) -> str:
    if task == "strategyqa":
        return (
            "At the end, output exactly one final line in the format:\n"
            "Final Answer: yes\n"
            "or\n"
            "Final Answer: no"
        )
    return (
        "At the end, output exactly one final line in the format:\n"
        "Final Answer: <answer>"
    )


def build_cot_prompt(
    question: str,
    task: TaskType = "generic",
    context: Optional[str] = None,
) -> str:
    """
    标准 CoT 轨迹生成 prompt。
    用于：
    - build_trajectories.py
    - baselines/cot.py
    - baselines/self_consistency.py
    """
    question = _clean_text(question)
    ctx = _format_context(context)

    return (
        "You are a careful reasoning assistant.\n"
        f"{_task_instruction(task)}\n"
        "Reason step by step.\n"
        f"{_final_answer_instruction(task)}\n\n"
        f"{ctx}"
        f"Question:\n{question}\n\n"
        "Reasoning:\n"
    )


def build_direct_answer_prompt(
    question: str,
    task: TaskType = "generic",
    context: Optional[str] = None,
) -> str:
    """
    不显式要求中间推理的直接回答 prompt。
    可用于：
    - 对照实验
    - no-CoT baseline
    """
    question = _clean_text(question)
    ctx = _format_context(context)

    return (
        "You are a precise question answering assistant.\n"
        f"{_task_instruction(task)}\n"
        "Do not provide unnecessary extra text.\n"
        f"{_final_answer_instruction(task)}\n\n"
        f"{ctx}"
        f"Question:\n{question}\n\n"
        "Answer:\n"
    )


def build_backtrack_prompt(
    question: str,
    valid_prefix: str,
    task: TaskType = "generic",
    context: Optional[str] = None,
    error_type: str = "unknown",
) -> str:
    """
    从“正确前缀”继续生成后续推理。
    用于：
    - controller/backtrack.py
    场景：
    - 已知前缀基本正确，只需要从该前缀继续往后写
    """
    question = _clean_text(question)
    valid_prefix = _clean_text(valid_prefix)
    ctx = _format_context(context)

    return (
        "You are repairing a reasoning process by backtracking.\n"
        "The prefix below is considered valid and should be preserved.\n"
        "Continue the reasoning from that prefix.\n"
        "Do not restart from scratch.\n"
        "Do not repeat the prefix unnecessarily.\n"
        f"Suspected previous error type: {error_type}\n"
        f"{_final_answer_instruction(task)}\n\n"
        f"{ctx}"
        f"Question:\n{question}\n\n"
        f"Valid Prefix:\n{valid_prefix}\n\n"
        "Continue the reasoning from the valid prefix:\n"
    )


def build_repair_prompt(
    question: str,
    valid_prefix: str,
    faulty_suffix: str,
    error_type: str = "unknown",
    task: TaskType = "generic",
    context: Optional[str] = None,
) -> str:
    """
    局部修复 prompt。
    用于：
    - controller/backtrack.py
    场景：
    - 前缀保留
    - 最近一段 continuation 可能错了
    - 只重写错误步及其后必要部分
    """
    question = _clean_text(question)
    valid_prefix = _clean_text(valid_prefix)
    faulty_suffix = _clean_text(faulty_suffix)
    ctx = _format_context(context)

    return (
        "You are revising a reasoning trace.\n"
        "The prefix is considered correct and must remain unchanged.\n"
        "Only rewrite the faulty step and the necessary continuation.\n"
        "Do not restart the entire solution.\n"
        "Do not modify the valid prefix.\n"
        "Keep the corrected continuation concise and logically consistent.\n"
        f"Detected error type: {error_type}\n"
        f"{_final_answer_instruction(task)}\n\n"
        f"{ctx}"
        f"Question:\n{question}\n\n"
        f"Correct Prefix:\n{valid_prefix}\n\n"
        f"Faulty Continuation:\n{faulty_suffix}\n\n"
        "Rewritten Continuation:\n"
    )


def build_error_diagnosis_prompt(
    question: str,
    prefix: str,
    task: TaskType = "generic",
    context: Optional[str] = None,
) -> str:
    """
    错误诊断 prompt。
    可选用于：
    - 分析最近前缀哪里可能错
    - 给 repair/backtrack 提供更细的错误类型
    """
    question = _clean_text(question)
    prefix = _clean_text(prefix)
    ctx = _format_context(context)

    return (
        "You are analyzing a partial reasoning trace.\n"
        "Decide whether the reasoning is likely correct so far.\n"
        "If there is a likely issue, identify the most plausible error type.\n"
        "Possible error types include:\n"
        "- arithmetic_error\n"
        "- contradiction\n"
        "- unsupported_jump\n"
        "- missing_evidence\n"
        "- premature_conclusion\n"
        "- none\n\n"
        "Output in the following format:\n"
        "Judgment: <correct_or_incorrect>\n"
        "Error Type: <type>\n"
        "Explanation: <short explanation>\n\n"
        f"{ctx}"
        f"Question:\n{question}\n\n"
        f"Partial Reasoning:\n{prefix}\n\n"
        "Diagnosis:\n"
    )


def build_continue_from_prefix_prompt(question: str, prefix_text: str) -> str:
    return (
        "You are continuing an existing reasoning trace.\n"
        "Keep the existing prefix unchanged and continue the reasoning from there.\n"
        "At the end, output:\n"
        "Final Answer: <answer>\n\n"
        f"Question: {question}\n\n"
        "Existing Reasoning Prefix:\n"
        f"{prefix_text}\n\n"
        "Continue the reasoning from the current point:\n"
    )


def build_self_consistency_prompt(
    question: str,
    task: TaskType = "generic",
    context: Optional[str] = None,
) -> str:
    """
    Self-Consistency 通常仍可复用 CoT prompt。
    单独保留函数是为了后续做 prompt ablation 更方便。
    """
    return build_cot_prompt(question=question, task=task, context=context)
