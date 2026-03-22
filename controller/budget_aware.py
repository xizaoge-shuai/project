from __future__ import annotations
from controller.threshold import ThresholdController

class BudgetAwareController(ThresholdController):
    def __init__(self, tau_prune: float = 0.25, tau_backtrack: float = 0.45, tau_accept: float = 0.88, max_backtracks: int = 2, tight_budget_bonus: float = 0.08, ample_budget_bonus: float = -0.05):
        super().__init__(tau_prune, tau_backtrack, tau_accept, max_backtracks)
        self.tight_budget_bonus = tight_budget_bonus
        self.ample_budget_bonus = ample_budget_bonus

    def act(self, state, pce_output: dict) -> str:
        bonus = 0.0
        if state.tokens_left < 32:
            bonus += self.tight_budget_bonus
        elif state.tokens_left > state.budget_tokens * 0.5:
            bonus += self.ample_budget_bonus
        old_accept, old_prune = self.tau_accept, self.tau_prune
        self.tau_accept = min(0.99, old_accept + bonus)
        self.tau_prune = max(0.01, old_prune + bonus / 2)
        action = super().act(state, pce_output)
        self.tau_accept, self.tau_prune = old_accept, old_prune
        return action
