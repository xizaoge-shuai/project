from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional

import requests

from generator.base import BaseGenerator
from generator.utils import (
    batched,
    build_generation_result,
    safe_merge_generation_kwargs,
    truncate_by_stop,
)


class APIGenerator(BaseGenerator):
    """
    正式版 API 生成器。

    当前主实现：
    - provider = "openai_compatible"
      使用通用 HTTP 接口 /chat/completions

    可扩展：
    - 后续可再加 anthropic / google / 其他 provider 分支
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = copy.deepcopy(config)
        provider = self.config.get("provider", "openai_compatible")
        model_name = self.config.get(
            "model_name", self.config.get("model", "unknown-api-model")
        )

        super().__init__(name=model_name, backend=f"api:{provider}")

        self.provider = provider
        self.model_name = model_name
        self.base_url = self.config.get("base_url", "").rstrip("/")
        self.api_key = self.config.get("api_key", "")
        self.timeout = float(self.config.get("timeout", 120))
        self.max_retries = int(self.config.get("max_retries", 2))
        self.batch_size = int(self.config.get("batch_size", 8))

        if self.provider == "openai_compatible":
            if not self.base_url:
                raise ValueError(
                    "For provider='openai_compatible', 'base_url' is required."
                )
            if not self.api_key:
                raise ValueError(
                    "For provider='openai_compatible', 'api_key' is required."
                )

    def healthcheck(self) -> bool:
        if self.provider == "openai_compatible":
            return bool(self.base_url and self.api_key)
        return False

    def _default_gen_kwargs(self) -> Dict[str, Any]:
        return {
            "temperature": float(self.config.get("temperature", 0.7)),
            "top_p": float(self.config.get("top_p", 0.95)),
            "max_new_tokens": int(self.config.get("max_new_tokens", 256)),
            "stop": self.config.get("stop", None),
        }

    def generate_one(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        results = self.generate_many([prompt], **kwargs)
        return results[0]

    def generate_many(self, prompts: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        if self.provider == "openai_compatible":
            return self._generate_many_openai_compatible(prompts, **kwargs)
        raise NotImplementedError(f"Unsupported provider: {self.provider}")

    def _generate_many_openai_compatible(
        self, prompts: List[str], **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        使用 OpenAI-compatible chat/completions 风格接口。
        注意：很多闭源 API、代理服务、自建推理服务都兼容这个格式。
        """
        merged_kwargs = safe_merge_generation_kwargs(self._default_gen_kwargs(), kwargs)
        results: List[Dict[str, Any]] = []

        # 很多 API 不支持真正 batch，这里按 prompt 逐个请求；
        # 外层仍保留 batched，便于后续限流与并发扩展。
        for batch_prompts in batched(prompts, self.batch_size):
            for prompt in batch_prompts:
                results.append(
                    self._request_one_openai_compatible(prompt, merged_kwargs)
                )

        return results

    def _request_one_openai_compatible(
        self,
        prompt: str,
        gen_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": gen_kwargs.get("temperature", 0.7),
            "top_p": gen_kwargs.get("top_p", 0.95),
            "max_tokens": gen_kwargs.get("max_new_tokens", 256),
        }

        if gen_kwargs.get("stop") is not None:
            payload["stop"] = gen_kwargs["stop"]

        last_err: Optional[Exception] = None
        for _ in range(self.max_retries + 1):
            started = time.time()
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                latency = time.time() - started

                text = self._extract_text_from_openai_response(data)
                text = truncate_by_stop(text, gen_kwargs.get("stop"))
                finish_reason = self._extract_finish_reason_from_openai_response(data)

                meta = {
                    "provider": self.provider,
                    "model_name": self.model_name,
                    "usage": data.get("usage", {}),
                    "raw_response": data,
                }

                return build_generation_result(
                    prompt=prompt,
                    text=text,
                    backend=self.backend,
                    finish_reason=finish_reason,
                    latency=latency,
                    meta=meta,
                )

            except Exception as e:
                last_err = e

        return build_generation_result(
            prompt=prompt,
            text="",
            backend=self.backend,
            finish_reason="error",
            latency=0.0,
            meta={
                "provider": self.provider,
                "model_name": self.model_name,
                "error": repr(last_err),
            },
        )

    @staticmethod
    def _extract_text_from_openai_response(data: Dict[str, Any]) -> str:
        """
        兼容常见 OpenAI-compatible 返回格式。
        """
        choices = data.get("choices", [])
        if not choices:
            return ""

        choice0 = choices[0]

        # chat/completions
        message = choice0.get("message", {})
        if "content" in message:
            content = message["content"]
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # 某些服务会返回 block list
                texts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(block.get("text", ""))
                return "\n".join(t for t in texts if t)

        # 老式 completions
        if "text" in choice0:
            return choice0["text"]

        return ""

    @staticmethod
    def _extract_finish_reason_from_openai_response(data: Dict[str, Any]) -> str:
        choices = data.get("choices", [])
        if not choices:
            return "unknown"
        return choices[0].get("finish_reason", "unknown")
