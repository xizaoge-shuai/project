from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseController(ABC):
    """
    统一接口：
    输入当前 state + pce_output
    输出动作字符串：continue / prune / backtrack / accept
    """

    @abstractmethod
    def act(self, state: Any, pce_output: Dict[str, Any]) -> str:
        raise NotImplementedError

    def __call__(self, state: Any, pce_output: Dict[str, Any]) -> str:
        return self.act(state, pce_output)
