from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseController(ABC):
    @abstractmethod
    def act(self, state, pce_output: Dict[str, Any]) -> str:
        raise NotImplementedError
