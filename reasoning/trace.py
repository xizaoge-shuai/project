from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any

@dataclass
class TraceRecord:
    sample_id: str
    actions: List[str] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    prefixes: List[str] = field(default_factory=list)
    final_answer: str = ""
    is_correct: int = 0
    tokens: int = 0
    latency: float = 0.0
    backtrack_count: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
