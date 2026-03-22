from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseGenerator(ABC):
    """
    统一的生成器抽象接口。
    所有生成后端（vLLM、本地 transformers、远程 API）都应实现这一接口。
    """

    def __init__(self, name: str, backend: str):
        self.name = name
        self.backend = backend

    @abstractmethod
    def generate_one(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """
        输入单个 prompt，输出统一格式字典。
        返回结果建议至少包含：
        {
            "prompt": str,
            "text": str,
            "finish_reason": str,
            "latency": float,
            "backend": str,
            "meta": dict,
        }
        """
        raise NotImplementedError

    @abstractmethod
    def generate_many(self, prompts: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """
        输入多个 prompts，返回对应结果列表。
        """
        raise NotImplementedError

    def generate_stepwise(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """
        默认 stepwise 接口。
        第一版直接复用 generate_one。
        后面如果做 token/step 级流式控制，可以重写。
        """
        return self.generate_one(prompt, **kwargs)

    def healthcheck(self) -> bool:
        """
        可选健康检查接口。默认返回 True。
        """
        return True

    def info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
        }
