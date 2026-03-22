from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ReasoningState:
    question: str
    gold_answer: str
    context: str = ""
    prefix_items: List[str] = field(default_factory=list)
    current_answer: str = ""
    steps_used: int = 0
    tokens_used: int = 0
    latency_used: float = 0.0
    budget_tokens: int = 256
    budget_latency: float = 10.0
    backtrack_count: int = 0
    action_history: List[str] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_left(self) -> int:
        return max(0, self.budget_tokens - self.tokens_used)

    @property
    def latency_left(self) -> float:
        return max(0.0, self.budget_latency - self.latency_used)

    def prefix_text(self) -> str:
        return "\n".join(self.prefix_items)
