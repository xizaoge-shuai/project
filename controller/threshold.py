from __future__ import annotations

from controller.base import BaseController
from controller.utils import extract_controller_features


class ThresholdController(BaseController):
    """
    主实验推荐使用的 controller。
    核心特点：
    1. 不再只看 success_prob，也会看 progress / budget / uncertainty / repairable
    2. 避免在特别早期就粗暴 prune
    3. near_end 时允许更积极 accept
    """

    def __init__(
        self,
        tau_prune: float = 0.25,
        tau_backtrack: float = 0.45,
        tau_accept: float = 0.88,
        max_backtracks: int = 2,
        min_progress_for_prune: float = 0.15,
        min_progress_for_backtrack: float = 0.10,
        max_uncertainty_for_accept: float = 0.20,
        near_end_accept_boost: float = 0.05,
    ):
        self.tau_prune = tau_prune
        self.tau_backtrack = tau_backtrack
        self.tau_accept = tau_accept
        self.max_backtracks = max_backtracks

        self.min_progress_for_prune = min_progress_for_prune
        self.min_progress_for_backtrack = min_progress_for_backtrack
        self.max_uncertainty_for_accept = max_uncertainty_for_accept
        self.near_end_accept_boost = near_end_accept_boost

    def act(self, state, pce_output: dict) -> str:
        f = extract_controller_features(state, pce_output)

        conf = f["conf"]
        uncertainty = f["uncertainty"]
        repairable = f["repairable"]
        progress = f["progress"]
        budget = f["budget"]

        backtrack_count = getattr(state, "backtrack_count", 0)

        # 1) near-end / high-confidence 时更愿意 accept
        effective_accept = self.tau_accept
        if progress > 0.85:
            effective_accept = max(0.0, self.tau_accept - self.near_end_accept_boost)

        if conf >= effective_accept and uncertainty <= self.max_uncertainty_for_accept:
            return "accept"

        # 2) 特别早期不要轻易 prune
        #    否则 atom-level 容易把局部波动误伤成全局坏路径
        if progress < self.min_progress_for_prune:
            if (
                repairable
                and conf < self.tau_backtrack
                and backtrack_count < self.max_backtracks
            ):
                return "backtrack"
            return "continue"

        # 3) 很低置信度时：
        #    - 可修复且还允许 backtrack -> backtrack
        #    - 否则 prune
        if conf < self.tau_prune:
            if (
                repairable
                and progress >= self.min_progress_for_backtrack
                and backtrack_count < self.max_backtracks
            ):
                return "backtrack"
            return "prune"

        # 4) 中低置信度但仍可能可修复 -> backtrack
        if (
            conf < self.tau_backtrack
            and repairable
            and progress >= self.min_progress_for_backtrack
            and backtrack_count < self.max_backtracks
        ):
            return "backtrack"

        # 5) 预算极低时，若当前置信度已经不差，则直接 accept，避免继续烧 token
        if budget < 0.10 and conf >= self.tau_backtrack:
            return "accept"

        return "continue"
