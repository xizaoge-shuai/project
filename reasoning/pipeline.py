from __future__ import annotations
from typing import Dict, Any
from reasoning.state import ReasoningState
from reasoning.trace import TraceRecord
from utils.tokenizer_utils import count_tokens
from utils.eval_utils import is_correct_prediction

class ReasoningPipeline:
    def __init__(self, pce, controller, answer_mode: str = "numeric"):
        self.pce = pce
        self.controller = controller
        self.answer_mode = answer_mode

    def run(self, sample_id: str, question: str, candidate_steps: list[str], gold_answer: str, context: str = "", budget_tokens: int = 256) -> Dict[str, Any]:
        state = ReasoningState(question=question, gold_answer=gold_answer, context=context, budget_tokens=budget_tokens)
        trace = TraceRecord(sample_id=sample_id)
        for step in candidate_steps:
            state.prefix_items.append(step)
            state.steps_used += 1
            state.tokens_used += count_tokens(step)
            state.current_answer = step.split("Answer:", 1)[-1].strip() if "Answer:" in step else state.current_answer

            pred = self.pce.predict(
                question=state.question,
                prefix_text=state.prefix_text(),
                current_answer=state.current_answer,
                context=state.context,
                budget_state={"tokens_left": state.tokens_left, "steps_used": state.steps_used, "backtrack_count": state.backtrack_count},
            )
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
            state.current_answer = state.prefix_items[-1] if state.prefix_items else ""
        trace.final_answer = state.current_answer
        trace.backtrack_count = state.backtrack_count
        trace.tokens = state.tokens_used
        trace.latency = state.latency_used
        trace.is_correct = int(is_correct_prediction(state.current_answer, gold_answer, answer_mode=self.answer_mode))
        return trace.to_dict()
