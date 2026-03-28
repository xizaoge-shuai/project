from __future__ import annotations

from typing import Dict

import numpy as np

from controller.base import BaseController
from controller.utils import extract_controller_features


class LearnedController(BaseController):
    """
    这不是 RL controller 的最终版，但比之前那个占位实现合理得多。
    它是一个可配置的线性策略：
    - 输入：controller features
    - 输出：四个动作分数
    - 然后取 argmax

    建议：
    - 主实验先不要用它
    - 主实验先用 Threshold / BudgetAware
    - 这版 LearnedController 作为后续 offline bandit / policy initialization 的入口
    """

    def __init__(
        self,
        continue_weights: Dict[str, float] | None = None,
        prune_weights: Dict[str, float] | None = None,
        backtrack_weights: Dict[str, float] | None = None,
        accept_weights: Dict[str, float] | None = None,
        bias: Dict[str, float] | None = None,
    ):
        self.continue_weights = continue_weights or {
            "conf": 0.4,
            "uncertainty": -0.3,
            "budget": 0.2,
            "early_stage": 0.2,
        }
        self.prune_weights = prune_weights or {
            "conf": -0.8,
            "not_repairable": 0.7,
            "progress": 0.3,
            "low_budget": 0.3,
        }
        self.backtrack_weights = backtrack_weights or {
            "conf": -0.6,
            "repairable": 0.8,
            "progress": 0.2,
            "backtrack_ratio": -0.5,
        }
        self.accept_weights = accept_weights or {
            "conf": 1.0,
            "uncertainty": -0.5,
            "near_end": 0.3,
            "budget": 0.1,
        }
        self.bias = bias or {
            "continue": 0.0,
            "prune": 0.0,
            "backtrack": 0.0,
            "accept": 0.0,
        }

    def _score(self, weights: Dict[str, float], feats: Dict[str, float]) -> float:
        s = 0.0
        for k, w in weights.items():
            s += w * feats.get(k, 0.0)
        return s

    def act(self, state, pce_output: dict) -> str:
        feats = extract_controller_features(state, pce_output)

        scores = {
            "continue": self.bias["continue"]
            + self._score(self.continue_weights, feats),
            "prune": self.bias["prune"] + self._score(self.prune_weights, feats),
            "backtrack": self.bias["backtrack"]
            + self._score(self.backtrack_weights, feats),
            "accept": self.bias["accept"] + self._score(self.accept_weights, feats),
        }

        # 不允许无限 backtrack
        max_backtracks = (
            state.meta.get("max_backtracks", 2) if getattr(state, "meta", None) else 2
        )
        if getattr(state, "backtrack_count", 0) >= max_backtracks:
            scores["backtrack"] = -1e9

        return max(scores.items(), key=lambda x: x[1])[0]
