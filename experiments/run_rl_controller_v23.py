from __future__ import annotations

import sys
import json
import re
import pickle
import random
import argparse
from pathlib import Path
from statistics import mean
from typing import Dict, Any, List, Tuple
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io import read_jsonl
from utils.tokenizer_utils import count_tokens
from utils.eval_utils import is_correct_prediction
from pce.dataset import build_input_text


# =========================
# Basic utils
# =========================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pce(path: str):
    if not path.endswith(".pkl"):
        raise ValueError(
            f"run_rl_controller_v22.py 目前只支持 light verifier 的 .pkl checkpoint，"
            f"你给的是: {path}"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def infer_split_path(dataset: str, split: str) -> str:
    candidates = [
        Path(f"data/processed/trajectories/{dataset}/{split}.jsonl"),
        Path(f"data/processed/trajectories/{dataset}/trajectories.jsonl"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"Cannot find trajectory file for dataset={dataset}, split={split}")


def get_answer_mode(dataset: str) -> str:
    if dataset == "gsm8k":
        return "numeric"
    if dataset == "strategyqa":
        return "yesno"
    return "span"


def compute_final_answer(prefix_items: List[str]) -> str:
    for s in reversed(prefix_items):
        if "Answer:" in s:
            return s.split("Answer:", 1)[-1].strip()
    return "\n".join(prefix_items).strip()


def extract_candidate_answer(prefix_items: List[str], dataset: str) -> str:
    """
    用于主动 accept：
    即使当前 prefix 里没有显式 'Answer:'，也尝试从已有推理中抽取候选答案。

    GSM8K:
    - 取当前 prefix 中最后一个数字作为 candidate answer。
    - 这不是完美答案抽取器，但足够让 RL 有机会学习 active accept。
    """
    text = "\n".join(prefix_items).strip()
    if not text:
        return ""

    # 先优先使用显式 Answer:
    for s in reversed(prefix_items):
        if "Answer:" in s:
            return s.split("Answer:", 1)[-1].strip()

    if dataset == "gsm8k":
        clean = text.replace(",", "")
        nums = re.findall(r"-?\d+(?:\.\d+)?", clean)
        return nums[-1] if nums else ""

    # 其他任务先退化为最后一个非空行
    lines = [x.strip() for x in text.split("\n") if x.strip()]
    return lines[-1] if lines else ""


def build_pce_text(
    question: str,
    context: str,
    prefix_items: List[str],
    current_answer: str,
    steps_used: int,
    total_steps: int,
    task: str,
) -> str:
    """
    对齐当前主线 checkpoint:
    outputs/checkpoints/pce_atom_rollout_p08_light.pkl

    对应离线 feature_set:
    prefix_plus_len_progress

    即：
    [TASK]
    [PREFIX]
    [PREFIX_LEN]
    [PREFIX_PROGRESS]
    """
    progress = steps_used / max(1, total_steps)
    row = {
        "task": task,
        "context": context,
        "question": question,
        "final_answer": current_answer,
        "prefix_text": "\n".join(prefix_items).strip(),
        "prefix_num_units": steps_used,
        "prefix_progress": progress,
    }
    return build_input_text(
        row,
        include_task=True,
        include_context=False,
        include_question=False,
        include_answer=False,
        include_prefix_len=True,
        include_prefix_progress=True,
    )


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_checkpoint(path: Path, policy: nn.Module, extra: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), **extra}, path)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =========================
# Environment
# =========================

class RLControllerEnv:
    ACTION_CONTINUE = 0
    ACTION_ACCEPT = 1
    ACTION_PRUNE = 2

    ACTION_NAMES = {
        0: "continue",
        1: "accept",
        2: "prune",
    }

    def __init__(
        self,
        pce_model,
        dataset: str,
        budget_tokens: int,
        lambda_token: float = 0.05,
        prune_good_penalty: float = 2.0,
        prune_bad_bonus: float = 0.2,
        wrong_accept_penalty: float = 1.0,
        budget_exceed_penalty: float = 0.3,
        min_accept_progress: float = 0.30,
        min_prune_progress: float = 0.30,
        min_steps_before_accept: int = 2,
        min_steps_before_prune: int = 2,
        strict_budget: bool = False,
        budget_safety_margin: int = 0,
        early_accept_bonus: float = 0.2,
    ):
        self.pce = pce_model
        self.dataset = dataset
        self.answer_mode = get_answer_mode(dataset)
        self.budget_tokens = budget_tokens

        self.lambda_token = lambda_token
        self.prune_good_penalty = prune_good_penalty
        self.prune_bad_bonus = prune_bad_bonus
        self.wrong_accept_penalty = wrong_accept_penalty
        self.budget_exceed_penalty = budget_exceed_penalty
        self.strict_budget = strict_budget
        self.budget_safety_margin = budget_safety_margin
        self.early_accept_bonus = early_accept_bonus
        self.min_accept_progress = min_accept_progress
        self.min_prune_progress = min_prune_progress
        self.min_steps_before_accept = min_steps_before_accept
        self.min_steps_before_prune = min_steps_before_prune

        self.row: Dict[str, Any] = {}
        self.prefix_items: List[str] = []
        self.idx = 0
        self.tokens_used = 0
        self.current_answer = ""
        self.candidate_answer = ""
        self.done = False
        self.full_traj_correct = False

    def reset(self, row: Dict[str, Any]) -> np.ndarray:
        self.row = row
        self.prefix_items = []
        self.idx = 0
        self.tokens_used = 0
        self.current_answer = ""
        self.done = False
        self.full_traj_correct = self._trajectory_final_is_correct(row)
        return self._advance_until_decision_point()

    def _trajectory_final_is_correct(self, row: Dict[str, Any]) -> bool:
        steps = row.get("steps", [])
        gold = row.get("gold_answer", "")
        if not steps:
            return False
        final_text = compute_final_answer(steps)
        return bool(is_correct_prediction(final_text, gold, answer_mode=self.answer_mode))

    def _terminal_state(self) -> np.ndarray:
        return np.zeros(6, dtype=np.float32)

    def _get_progress(self) -> float:
        total_steps = len(self.row.get("steps", []))
        steps_used = len(self.prefix_items)
        return steps_used / max(1, total_steps)

    def _advance_until_decision_point(self) -> np.ndarray:
        steps = self.row.get("steps", [])
        if self.idx >= len(steps):
            self.done = True
            return self._terminal_state()

        step = steps[self.idx]
        self.prefix_items.append(step)
        self.tokens_used += count_tokens(step)

        if "Answer:" in step:
            self.current_answer = step.split("Answer:", 1)[-1].strip()

        # v23: 即使没有显式 Answer，也从当前 prefix 抽取候选答案，
        # 让 controller 有机会学习主动 accept。
        self.candidate_answer = self.current_answer or extract_candidate_answer(
            self.prefix_items,
            self.dataset,
        )

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        total_steps = len(self.row.get("steps", []))
        steps_used = len(self.prefix_items)
        progress = self._get_progress()

        pce_text = build_pce_text(
            question=self.row.get("question", ""),
            context=self.row.get("context", ""),
            prefix_items=self.prefix_items,
            current_answer=self.current_answer,
            steps_used=steps_used,
            total_steps=total_steps,
            task=self.dataset,
        )

        pred = self.pce.predict(text=pce_text)
        success_prob = float(pred["success_prob"])
        uncertainty = 1.0 - success_prob
        tokens_left_ratio = max(0.0, self.budget_tokens - self.tokens_used) / max(1, self.budget_tokens)
        current_answer_flag = 1.0 if (self.current_answer or self.candidate_answer) else 0.0

        return np.array(
            [
                success_prob,
                uncertainty,
                progress,
                steps_used / max(1, total_steps),
                tokens_left_ratio,
                current_answer_flag,
            ],
            dtype=np.float32,
        )

    def get_action_mask(self) -> np.ndarray:
        """
        动作顺序:
        [continue, accept, prune]

        v2.2 strict-budget 版本：
        - 没有答案 / 太早时，禁止 accept
        - 太早时，禁止 prune
        - 如果 strict_budget=True，且继续下一步会超过预算，则禁止 continue
        """
        if self.done:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        steps = self.row.get("steps", [])
        steps_used = len(self.prefix_items)
        progress = self._get_progress()
        has_answer = bool(self.current_answer or self.candidate_answer)

        # 默认允许 continue
        allow_continue = 1.0

        # strict budget: 如果下一步会超过预算，就禁止 continue
        if self.strict_budget:
            next_idx = self.idx + 1

            # 如果还有下一步，估算继续之后会不会超过 budget
            if next_idx < len(steps):
                next_step_tokens = count_tokens(steps[next_idx])
                predicted_tokens = (
                    self.tokens_used
                    + next_step_tokens
                    + int(self.budget_safety_margin)
                )

                if predicted_tokens > self.budget_tokens:
                    allow_continue = 0.0

            # 如果已经是最后一步，continue 只是自然结束，不额外消耗下一步
            else:
                allow_continue = 1.0

        allow_accept = 1.0 if (
            has_answer
            and progress >= self.min_accept_progress
            and steps_used >= self.min_steps_before_accept
        ) else 0.0

        allow_prune = 1.0 if (
            progress >= self.min_prune_progress
            and steps_used >= self.min_steps_before_prune
        ) else 0.0

        # 兜底：如果 strict budget 禁止 continue，同时 accept/prune 都被 mask，
        # 那么必须打开一个终止动作，避免没有合法动作。
        if allow_continue < 0.5 and allow_accept < 0.5 and allow_prune < 0.5:
            if has_answer:
                allow_accept = 1.0
            else:
                allow_prune = 1.0

        return np.array(
            [allow_continue, allow_accept, allow_prune],
            dtype=np.float32,
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.done:
            return self._terminal_state(), 0.0, True, {}

        mask = self.get_action_mask()
        if mask[action] < 0.5:
            action = self.ACTION_CONTINUE

        current_step = self.row["steps"][self.idx]
        step_tokens = count_tokens(current_step)

        reward = - self.lambda_token * (step_tokens / max(1, self.budget_tokens))

        if action == self.ACTION_ACCEPT:
            final_answer = (
                self.current_answer
                or self.candidate_answer
                or compute_final_answer(self.prefix_items)
            )
            is_correct = bool(
                is_correct_prediction(
                    final_answer,
                    self.row["gold_answer"],
                    answer_mode=self.answer_mode,
                )
            )
            if is_correct:
                reward += 1.0 + self.early_accept_bonus * (1.0 - self._get_progress())
            else:
                reward += -self.wrong_accept_penalty
            self.done = True
            return self._terminal_state(), reward, True, {
                "action": "accept",
                "is_correct": int(is_correct),
                "tokens_used": self.tokens_used,
                "progress": self._get_progress(),
            }

        if action == self.ACTION_PRUNE:
            reward += self.prune_bad_bonus if (not self.full_traj_correct) else -self.prune_good_penalty
            self.done = True
            return self._terminal_state(), reward, True, {
                "action": "prune",
                "is_correct": 0,
                "tokens_used": self.tokens_used,
                "progress": self._get_progress(),
            }

        # continue
        self.idx += 1

        if self.tokens_used > self.budget_tokens:
            reward -= self.budget_exceed_penalty
            self.done = True
            final_answer = compute_final_answer(self.prefix_items)
            is_correct = bool(
                is_correct_prediction(
                    final_answer,
                    self.row["gold_answer"],
                    answer_mode=self.answer_mode,
                )
            )
            return self._terminal_state(), reward, True, {
                "action": "budget_exceeded",
                "is_correct": int(is_correct),
                "tokens_used": self.tokens_used,
                "progress": self._get_progress(),
            }

        if self.idx >= len(self.row["steps"]):
            self.done = True
            final_answer = compute_final_answer(self.prefix_items)
            is_correct = bool(
                is_correct_prediction(
                    final_answer,
                    self.row["gold_answer"],
                    answer_mode=self.answer_mode,
                )
            )
            reward += 1.0 if is_correct else 0.0
            return self._terminal_state(), reward, True, {
                "action": "end",
                "is_correct": int(is_correct),
                "tokens_used": self.tokens_used,
                "progress": self._get_progress(),
            }

        next_state = self._advance_until_decision_point()
        return next_state, reward, False, {
            "action": "continue",
            "is_correct": 0,
            "tokens_used": self.tokens_used,
            "progress": self._get_progress(),
        }


# =========================
# Policy
# =========================

class PolicyNet(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, num_actions: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def masked_logits(policy: PolicyNet, state: np.ndarray, action_mask: np.ndarray) -> torch.Tensor:
    x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    logits = policy(x).squeeze(0)
    mask = torch.tensor(action_mask, dtype=torch.float32)
    return logits.masked_fill(mask < 0.5, -1e9)


def masked_dist(policy: PolicyNet, state: np.ndarray, action_mask: np.ndarray):
    logits = masked_logits(policy, state, action_mask)
    return torch.distributions.Categorical(logits=logits), logits


# =========================
# Teacher policy for warm start
# =========================

def teacher_action_from_state(
    state: np.ndarray,
    action_mask: np.ndarray,
    teacher: str,
    tau_prune: float = 0.25,
    tau_accept: float = 0.70,
    tight_budget_bonus: float = 0.08,
    ample_budget_relax: float = 0.05,
    near_end_accept_boost: float = 0.05,
) -> int:
    """
    用一个简单 teacher 产生 imitation label。
    teacher 可选：
    - threshold
    - budget_aware

    这里不用直接 import 原 controller，是为了避免 ReasoningState 接口不一致。
    但逻辑上对应 threshold/budget-aware 的核心行为。
    """
    conf = float(state[0])
    progress = float(state[2])
    budget_ratio = float(state[4])

    local_tau_accept = tau_accept
    local_tau_prune = tau_prune

    if teacher == "budget_aware":
        if budget_ratio < 0.20:
            local_tau_accept = max(0.0, local_tau_accept - tight_budget_bonus)
            local_tau_prune = min(1.0, local_tau_prune + tight_budget_bonus * 0.5)
        elif budget_ratio > 0.60:
            local_tau_accept = min(1.0, local_tau_accept + ample_budget_relax)
            local_tau_prune = max(0.0, local_tau_prune - ample_budget_relax)

        if progress > 0.85:
            local_tau_accept = max(0.0, local_tau_accept - near_end_accept_boost)

    # 先 accept，再 prune，再 continue
    if action_mask[RLControllerEnv.ACTION_ACCEPT] > 0.5 and conf >= local_tau_accept:
        return RLControllerEnv.ACTION_ACCEPT

    if action_mask[RLControllerEnv.ACTION_PRUNE] > 0.5 and conf < local_tau_prune:
        return RLControllerEnv.ACTION_PRUNE

    return RLControllerEnv.ACTION_CONTINUE


def collect_teacher_transitions(
    env: RLControllerEnv,
    rows: List[Dict[str, Any]],
    teacher: str,
    max_transitions: int,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    transitions: List[Tuple[np.ndarray, np.ndarray, int]] = []

    for row in rows:
        state = env.reset(row)
        done = False
        while not done:
            mask = env.get_action_mask()
            action = teacher_action_from_state(state, mask, teacher=teacher)
            transitions.append((state.copy(), mask.copy(), int(action)))
            next_state, _, done, _ = env.step(action)
            state = next_state

            if max_transitions > 0 and len(transitions) >= max_transitions:
                return transitions

    return transitions


def imitation_pretrain(
    policy: PolicyNet,
    optimizer,
    transitions: List[Tuple[np.ndarray, np.ndarray, int]],
    epochs: int,
    batch_size: int,
) -> None:
    if epochs <= 0 or not transitions:
        return

    for ep in range(1, epochs + 1):
        random.shuffle(transitions)
        losses = []

        for i in range(0, len(transitions), batch_size):
            batch = transitions[i:i + batch_size]

            loss = 0.0
            for state, mask, action in batch:
                logits = masked_logits(policy, state, mask)
                target = torch.tensor([action], dtype=torch.long)
                loss = loss + nn.functional.cross_entropy(logits.unsqueeze(0), target)

            loss = loss / max(1, len(batch))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))

        print(f"[imitation {ep:03d}] ce_loss={mean(losses):.4f} transitions={len(transitions)}")


# =========================
# RL training
# =========================

def discounted_returns(rewards: List[float], gamma: float) -> List[float]:
    ret = 0.0
    out = []
    for r in reversed(rewards):
        ret = float(r) + gamma * ret
        out.append(ret)
    out.reverse()
    return out


def run_episode(
    env: RLControllerEnv,
    policy: PolicyNet,
    row: Dict[str, Any],
    train: bool,
):
    state = env.reset(row)

    log_probs = []
    rewards = []
    entropies = []
    infos = []
    action_trace = []

    done = False
    while not done:
        mask = env.get_action_mask()
        dist, logits = masked_dist(policy, state, mask)

        if train:
            action = dist.sample()
        else:
            action = torch.argmax(logits, dim=-1)

        action_int = int(action.item())

        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        action_trace.append(RLControllerEnv.ACTION_NAMES[action_int])

        next_state, reward, done, info = env.step(action_int)

        rewards.append(float(reward))
        infos.append(info)
        state = next_state

    final_info = infos[-1] if infos else {}
    final_info["action_trace"] = action_trace
    return log_probs, rewards, entropies, final_info


def train_one_batch(
    env: RLControllerEnv,
    policy: PolicyNet,
    optimizer,
    rows: List[Dict[str, Any]],
    batch_episodes: int,
    gamma: float,
    entropy_coef: float,
):
    all_log_probs = []
    all_returns = []
    all_entropies = []
    episode_rewards = []

    for _ in range(batch_episodes):
        row = random.choice(rows)
        log_probs, rewards, entropies, _ = run_episode(env, policy, row, train=True)

        returns = discounted_returns(rewards, gamma=gamma)
        all_log_probs.extend(log_probs)
        all_returns.extend(returns)
        all_entropies.extend(entropies)
        episode_rewards.append(float(sum(rewards)))

    returns_t = torch.tensor(all_returns, dtype=torch.float32)
    if len(returns_t) > 1:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

    policy_loss = 0.0
    for lp, G in zip(all_log_probs, returns_t):
        policy_loss = policy_loss + (-lp * G)

    policy_loss = policy_loss / max(1, len(all_log_probs))

    entropy_bonus = torch.stack(all_entropies).mean() if all_entropies else torch.tensor(0.0)
    loss = policy_loss - entropy_coef * entropy_bonus

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "entropy": float(entropy_bonus.detach().item()),
        "train_avg_episode_reward": mean(episode_rewards) if episode_rewards else 0.0,
    }


def evaluate_policy(
    policy: PolicyNet,
    pce_model,
    rows: List[Dict[str, Any]],
    dataset: str,
    budget_tokens: int,
    env_kwargs: Dict[str, Any],
):
    env = RLControllerEnv(
        pce_model=pce_model,
        dataset=dataset,
        budget_tokens=budget_tokens,
        **env_kwargs,
    )

    results = []
    terminal_counter = Counter()

    for row in rows:
        _, rewards, _, info = run_episode(env, policy, row, train=False)
        terminal_action = info.get("action", "unknown")
        terminal_counter[terminal_action] += 1

        results.append({
            "sample_id": row.get("sample_id", ""),
            "reward_sum": float(sum(rewards)),
            "is_correct": int(info.get("is_correct", 0)),
            "tokens_used": float(info.get("tokens_used", 0.0)),
            "terminal_action": terminal_action,
            "progress": float(info.get("progress", 0.0)),
            "action_trace": info.get("action_trace", []),
        })

    summary = {
        "n_samples": len(results),
        "final_accuracy": mean(r["is_correct"] for r in results) if results else None,
        "avg_tokens": mean(r["tokens_used"] for r in results) if results else None,
        "avg_reward": mean(r["reward_sum"] for r in results) if results else None,
        "terminal_action_counter": dict(terminal_counter),
    }
    return {"summary": summary, "details": results}


def tradeoff_score(summary: Dict[str, Any], budget_tokens: int, coef: float) -> float:
    acc = float(summary.get("final_accuracy", 0.0))
    avg_tokens = float(summary.get("avg_tokens", 0.0))
    return acc - coef * (avg_tokens / max(1, budget_tokens))


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "strategyqa", "hotpotqa"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--budget_tokens", type=int, default=128)

    parser.add_argument("--train_path", default=None)
    parser.add_argument("--eval_path", default=None)
    parser.add_argument("--train_limit", type=int, default=300)
    parser.add_argument("--eval_limit", type=int, default=200)

    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--batch_episodes", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # reward
    parser.add_argument("--lambda_token", type=float, default=0.05)
    parser.add_argument("--prune_good_penalty", type=float, default=2.0)
    parser.add_argument("--prune_bad_bonus", type=float, default=0.2)
    parser.add_argument("--wrong_accept_penalty", type=float, default=1.0)
    parser.add_argument("--budget_exceed_penalty", type=float, default=0.3)

    # masks
    parser.add_argument("--min_accept_progress", type=float, default=0.30)
    parser.add_argument("--min_prune_progress", type=float, default=0.30)
    parser.add_argument("--min_steps_before_accept", type=int, default=2)
    parser.add_argument("--min_steps_before_prune", type=int, default=2)
    parser.add_argument("--strict_budget", type=int, default=0)
    parser.add_argument("--budget_safety_margin", type=int, default=0)
    parser.add_argument("--early_accept_bonus", type=float, default=0.2)
    # entropy
    parser.add_argument("--entropy_coef", type=float, default=0.01)

    # teacher warm start
    parser.add_argument("--teacher", default="budget_aware", choices=["threshold", "budget_aware", "none"])
    parser.add_argument("--imitation_epochs", type=int, default=5)
    parser.add_argument("--imitation_batch_size", type=int, default=64)
    parser.add_argument("--imitation_max_transitions", type=int, default=5000)

    # save
    parser.add_argument("--tradeoff_coef", type=float, default=0.1)
    parser.add_argument("--out_dir", default="outputs/rl_v22")

    args = parser.parse_args()
    set_seed(args.seed)

    train_path = args.train_path or infer_split_path(args.dataset, "train")
    eval_path = args.eval_path or infer_split_path(args.dataset, "test")

    train_rows = read_jsonl(train_path)
    eval_rows = read_jsonl(eval_path)

    if args.train_limit > 0:
        train_rows = train_rows[:args.train_limit]
    if args.eval_limit > 0:
        eval_rows = eval_rows[:args.eval_limit]

    pce_model = load_pce(args.checkpoint)

    env_kwargs = {
        "lambda_token": args.lambda_token,
        "prune_good_penalty": args.prune_good_penalty,
        "prune_bad_bonus": args.prune_bad_bonus,
        "wrong_accept_penalty": args.wrong_accept_penalty,
        "budget_exceed_penalty": args.budget_exceed_penalty,
        "min_accept_progress": args.min_accept_progress,
        "min_prune_progress": args.min_prune_progress,
        "min_steps_before_accept": args.min_steps_before_accept,
        "min_steps_before_prune": args.min_steps_before_prune,
        "strict_budget": bool(args.strict_budget),
        "budget_safety_margin": args.budget_safety_margin,
        "early_accept_bonus": args.early_accept_bonus,
    }

    policy = PolicyNet(input_dim=6, hidden_dim=args.hidden_dim, num_actions=3)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    train_env = RLControllerEnv(
        pce_model=pce_model,
        dataset=args.dataset,
        budget_tokens=args.budget_tokens,
        **env_kwargs,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # save config
    save_json(out_dir / "config.json", vars(args))

    # imitation warm start
    if args.teacher != "none" and args.imitation_epochs > 0:
        print(f"[info] collecting teacher transitions: teacher={args.teacher}")
        transitions = collect_teacher_transitions(
            env=train_env,
            rows=train_rows,
            teacher=args.teacher,
            max_transitions=args.imitation_max_transitions,
        )
        print(f"[info] collected transitions = {len(transitions)}")
        imitation_pretrain(
            policy=policy,
            optimizer=optimizer,
            transitions=transitions,
            epochs=args.imitation_epochs,
            batch_size=args.imitation_batch_size,
        )

    best_acc = -1e9
    best_reward = -1e9
    best_tradeoff = -1e9
    history = []

    for ep in range(1, args.episodes + 1):
        train_stats = train_one_batch(
            env=train_env,
            policy=policy,
            optimizer=optimizer,
            rows=train_rows,
            batch_episodes=args.batch_episodes,
            gamma=args.gamma,
            entropy_coef=args.entropy_coef,
        )

        eval_result = evaluate_policy(
            policy=policy,
            pce_model=pce_model,
            rows=eval_rows,
            dataset=args.dataset,
            budget_tokens=args.budget_tokens,
            env_kwargs=env_kwargs,
        )

        summary = eval_result["summary"]
        eval_acc = float(summary["final_accuracy"])
        eval_reward = float(summary["avg_reward"])
        eval_tradeoff = tradeoff_score(summary, args.budget_tokens, args.tradeoff_coef)

        record = {
            "episode": ep,
            **train_stats,
            "eval_accuracy": eval_acc,
            "eval_avg_tokens": float(summary["avg_tokens"]),
            "eval_avg_reward": eval_reward,
            "eval_tradeoff": eval_tradeoff,
            "terminal_action_counter": summary["terminal_action_counter"],
        }
        history.append(record)
        save_json(out_dir / "training_history.json", {"history": history})

        print(
            f"[ep {ep:03d}] "
            f"loss={train_stats['loss']:.4f} "
            f"ploss={train_stats['policy_loss']:.4f} "
            f"ent={train_stats['entropy']:.4f} "
            f"train_r={train_stats['train_avg_episode_reward']:.4f} "
            f"eval_acc={eval_acc:.4f} "
            f"eval_tokens={float(summary['avg_tokens']):.2f} "
            f"eval_reward={eval_reward:.4f} "
            f"eval_tradeoff={eval_tradeoff:.4f} "
            f"actions={summary['terminal_action_counter']}"
        )

        extra = {
            "dataset": args.dataset,
            "budget_tokens": args.budget_tokens,
            "checkpoint": args.checkpoint,
            "episode": ep,
            "summary": summary,
            "args": vars(args),
        }

        if eval_acc > best_acc:
            best_acc = eval_acc
            save_checkpoint(out_dir / "best_acc.pt", policy, extra)
            save_json(out_dir / "best_acc_summary.json", summary)
            save_jsonl(out_dir / "best_acc_details.jsonl", eval_result["details"])

        if eval_reward > best_reward:
            best_reward = eval_reward
            save_checkpoint(out_dir / "best_reward.pt", policy, extra)
            save_json(out_dir / "best_reward_summary.json", summary)
            save_jsonl(out_dir / "best_reward_details.jsonl", eval_result["details"])

        if eval_tradeoff > best_tradeoff:
            best_tradeoff = eval_tradeoff
            save_checkpoint(out_dir / "best_tradeoff.pt", policy, extra)
            save_json(out_dir / "best_tradeoff_summary.json", summary)
            save_jsonl(out_dir / "best_tradeoff_details.jsonl", eval_result["details"])

        save_checkpoint(out_dir / "latest.pt", policy, extra)
        save_json(out_dir / "latest_summary.json", summary)
        save_jsonl(out_dir / "latest_details.jsonl", eval_result["details"])

    print(f"Saved RL-v2.2 artifacts to {out_dir}")


if __name__ == "__main__":
    main()
