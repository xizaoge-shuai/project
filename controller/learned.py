from __future__ import annotations
import numpy as np
from controller.base import BaseController

class LearnedController(BaseController):
    """
    可运行的轻量占位实现：
    用手工特征 + 线性打分代替真正 RL，后续可直接替换。
    """
    def __init__(self):
        self.W = np.array([
            [ 1.0, -0.5, -0.1, -0.2],   # continue
            [-1.5,  0.8,  0.2,  0.4],   # prune
            [-0.5,  0.6, -0.1,  0.7],   # backtrack
            [ 1.8, -0.7,  0.1, -0.2],   # accept
        ])

    def _features(self, state, pce_output):
        return np.array([
            float(pce_output.get("success_prob", 0.0)),
            float(state.steps_used),
            float(state.tokens_left),
            float(pce_output.get("repairable", 0)),
        ])

    def act(self, state, pce_output: dict) -> str:
        feats = self._features(state, pce_output)
        scores = self.W @ feats
        return ["continue", "prune", "backtrack", "accept"][int(np.argmax(scores))]
