from __future__ import annotations

from typing import Dict, Any

from reasoning.state import ReasoningState
from reasoning.trace import TraceRecord
from utils.tokenizer_utils import count_tokens
from utils.eval_utils import is_correct_prediction
from pce.dataset import build_input_text


class ReasoningPipeline:
    def __init__(self, pce, controller, answer_mode: str = "numeric"):
        self.pce = pce
        self.controller = controller
        self.answer_mode = answer_mode

    def build_pce_text(self, state: ReasoningState, total_steps: int) -> str:
        """
        和离线推理保持一致：
        默认按 prefix_plus_len_progress 构造输入。
        这和 pce/inference.py 里的默认 feature_set 对齐。
        """
        progress = len(state.prefix_items) / max(1, total_steps)

        row = {
            "task": "gsm8k" if self.answer_mode == "numeric"
                    else ("strategyqa" if self.answer_mode == "yesno" else "hotpotqa"),
            "context": state.context,
            "question": state.question,
            "final_answer": state.current_answer,
            "prefix_text": state.prefix_text(),
            "prefix_num_units": len(state.prefix_items),
            "prefix_progress": progress,
        }

        text = build_input_text(
            row,
            include_task=True,
            include_context=False,
            include_question=False,
            include_answer=False,
            include_prefix_len=True,
            include_prefix_progress=True,
        )
        return text

    def run(
        self,
        sample_id: str,
        question: str,
        candidate_steps: list[str],
        gold_answer: str,
        context: str = "",
        budget_tokens: int = 256,
    ) -> Dict[str, Any]:
        state = ReasoningState(
            question=question,
            gold_answer=gold_answer,
            context=context,
            budget_tokens=budget_tokens,
        )
        trace = TraceRecord(sample_id=sample_id)

        total_steps = len(candidate_steps)

        for step in candidate_steps:
            state.prefix_items.append(step)
            state.steps_used += 1
            state.tokens_used += count_tokens(step)

            if "Answer:" in step:
                state.current_answer = step.split("Answer:", 1)[-1].strip()

            pce_text = self.build_pce_text(state, total_steps=total_steps)
            pred = self.pce.predict(text=pce_text)

            action = self.controller.act(state=state, pce_output=pred)

            state.action_history.append(action)
            state.confidence_history.append(float(pred["success_prob"]))

            trace.actions.append(action)
            trace.confidences.append(float(pred["success_prob"]))
            trace.prefixes.append(state.prefix_text())

            if action == "backtrack":
                if state.prefix_items:
                    state.prefix_items = state.prefix_items[:-1]
                    state.backtrack_count += 1
                continue

            if action in ("accept", "prune"):
                break

        if not state.current_answer:
            # 如果还没有显式 Answer:，至少把整段 prefix 交给评估函数
            state.current_answer = state.prefix_text()

        trace.final_answer = state.current_answer
        trace.backtrack_count = state.backtrack_count
        trace.tokens = state.tokens_used
        trace.latency = state.latency_used
        trace.is_correct = int(
            is_correct_prediction(
                state.current_answer,
                gold_answer,
                answer_mode=self.answer_mode,
            )
        )
        return trace.to_dict()