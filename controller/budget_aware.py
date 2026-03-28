from __future__ import annotations

from controller.threshold import ThresholdController
from controller.utils import extract_controller_features


class BudgetAwareController(ThresholdController):
    """
    比 ThresholdController 多一步：
    根据 budget ratio 动态调整阈值。
    不直接永久修改成员变量，而是在 act 内计算“有效阈值”。
    """

    def __init__(
        self,
        tau_prune: float = 0.25,
        tau_backtrack: float = 0.45,
        tau_accept: float = 0.88,
        max_backtracks: int = 2,
        tight_budget_bonus: float = 0.08,
        ample_budget_relax: float = 0.05,
        min_progress_for_prune: float = 0.15,
        min_progress_for_backtrack: float = 0.10,
        max_uncertainty_for_accept: float = 0.20,
        near_end_accept_boost: float = 0.05,
    ):
        super().__init__(
            tau_prune=tau_prune,
            tau_backtrack=tau_backtrack,
            tau_accept=tau_accept,
            max_backtracks=max_backtracks,
            min_progress_for_prune=min_progress_for_prune,
            min_progress_for_backtrack=min_progress_for_backtrack,
            max_uncertainty_for_accept=max_uncertainty_for_accept,
            near_end_accept_boost=near_end_accept_boost,
        )
        self.tight_budget_bonus = tight_budget_bonus
        self.ample_budget_relax = ample_budget_relax

    def act(self, state, pce_output: dict) -> str:
        f = extract_controller_features(state, pce_output)

        conf = f["conf"]
        uncertainty = f["uncertainty"]
        repairable = f["repairable"]
        progress = f["progress"]
        budget = f["budget"]
        backtrack_count = getattr(state, "backtrack_count", 0)

        # budget 紧张：更倾向 accept / prune，少 backtrack
        # budget 宽裕：更允许 continue / backtrack
        tau_accept = self.tau_accept
        tau_prune = self.tau_prune
        tau_backtrack = self.tau_backtrack
        max_backtracks = self.max_backtracks

        if budget < 0.20:
            tau_accept = max(0.0, tau_accept - self.tight_budget_bonus)
            tau_prune = min(1.0, tau_prune + self.tight_budget_bonus * 0.5)
            tau_backtrack = max(0.0, tau_backtrack - self.tight_budget_bonus * 0.5)
            max_backtracks = max(0, self.max_backtracks - 1)

        elif budget > 0.60:
            tau_accept = min(1.0, tau_accept + self.ample_budget_relax)
            tau_prune = max(0.0, tau_prune - self.ample_budget_relax)
            tau_backtrack = min(1.0, tau_backtrack + self.ample_budget_relax * 0.5)

        # near-end 时更积极 accept
        if progress > 0.85:
            tau_accept = max(0.0, tau_accept - self.near_end_accept_boost)

        if conf >= tau_accept and uncertainty <= self.max_uncertainty_for_accept:
            return "accept"

        if progress < self.min_progress_for_prune:
            if repairable and conf < tau_backtrack and backtrack_count < max_backtracks:
                return "backtrack"
            return "continue"

        if conf < tau_prune:
            if (
                repairable
                and progress >= self.min_progress_for_backtrack
                and backtrack_count < max_backtracks
            ):
                return "backtrack"
            return "prune"

        if (
            conf < tau_backtrack
            and repairable
            and progress >= self.min_progress_for_backtrack
            and backtrack_count < max_backtracks
        ):
            return "backtrack"

        if budget < 0.10 and conf >= tau_backtrack:
            return "accept"

        return "continue"
