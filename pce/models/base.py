from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any

class BasePCE(ABC):
    @abstractmethod
    def fit(self, texts, y, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, question: str, prefix_text: str, current_answer: str = "", context: str = "", budget_state: dict | None = None) -> Dict[str, Any]:
        raise NotImplementedError
