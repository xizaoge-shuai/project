from __future__ import annotations
from controller.base import BaseController

class ThresholdController(BaseController):
    def __init__(self, tau_prune: float = 0.25, tau_backtrack: float = 0.45, tau_accept: float = 0.88, max_backtracks: int = 2):
        self.tau_prune = tau_prune
        self.tau_backtrack = tau_backtrack
        self.tau_accept = tau_accept
        self.max_backtracks = max_backtracks

    def act(self, state, pce_output: dict) -> str:
        conf = float(pce_output.get("success_prob", 0.0))
        repairable = int(pce_output.get("repairable", 0))
        if conf >= self.tau_accept:
            return "accept"
        if conf < self.tau_prune and not repairable:
            return "prune"
        if conf < self.tau_backtrack and repairable and state.backtrack_count < self.max_backtracks:
            return "backtrack"
        return "continue"
