from __future__ import annotations

import sys
import json
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
# Utils
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
            f"当前 RL v2.1 只支持 pickle light verifier checkpoint，"
            f"你给的是: {path}"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def infer_split_path(dataset: str, split: str) -> str:
    cands = [
        Path(f"data/processed/trajectories/{dataset}/{split}.jsonl"),
        Path(f"data/processed/trajectories/{dataset}/trajectories.jsonl"),
    ]
    for c in cands:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        f"Cannot find trajectory file for dataset={dataset}, split={split}"
    )


def get_answer_mode(dataset: str) -> str:
    if dataset == "gsm8k":
        return "numeric"
    if dataset == "strategyqa":
        return "yesno"
    return "span"


def compute_final_answer(prefix_items: List[str]) -> str:
    """
    优先抽取最后一个显式 Answer:；否则把整个 prefix 当作最终文本。
    """
    for s in reversed(prefix_items):
        if "Answer:" in s:
            return s.split("Answer:", 1)[-1].strip()
    return "\n".join(prefix_items).strip()


def build_pce_text(
    question: str,
    context: str,
    prefix_items: List[str],
    current_answer: str,
    steps_used: int,
    total_steps: int,
    task: str = "gsm8k",
) -> str:
    """
    和你当前 light 主线对齐：
    prefix_plus_len_progress
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


# =========================
# Env
# =========================

class RLControllerEnv:
    """
    RL v2.1:
    - 动作空间：continue / accept / prune
    - 加 action mask：
      1) 没有显式答案时不能 accept
      2) progress 太低时不能 accept
    """

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
        dataset: str = "gsm8k",
        budget_tokens: int = 128,
        lambda_token: float = 0.05,
        prune_good_penalty: float = 1.0,
        prune_bad_bonus: float = 0.2,
        wrong_accept_penalty: float = 0.5,
        budget_exceed_penalty: float = 0.3,
        min_accept_progress: float = 0.15,
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
        self.min_accept_progress = min_accept_progress

        self.row: Dict[str, Any] = {}
        self.prefix_items: List[str] = []
        self.idx = 0
        self.tokens_used = 0
        self.current_answer = ""
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
        return bool(
            is_correct_prediction(final_text, gold, answer_mode=self.answer_mode)
        )

    def _advance_until_decision_point(self) -> np.ndarray:
        if self.idx >= len(self.row["steps"]):
            self.done = True
            return self._terminal_state()

        step = self.row["steps"][self.idx]
        self.prefix_items.append(step)
        self.tokens_used += count_tokens(step)

        if "Answer:" in step:
            self.current_answer = step.split("Answer:", 1)[-1].strip()

        return self._get_state()

    def _terminal_state(self) -> np.ndarray:
        return np.zeros(6, dtype=np.float32)

    def _get_progress(self) -> float:
        total_steps = len(self.row["steps"])
        steps_used = len(self.prefix_items)
        return steps_used / max(1, total_steps)

    def _get_state(self) -> np.ndarray:
        total_steps = len(self.row["steps"])
        steps_used = len(self.prefix_items)
        progress = self._get_progress()

        pce_text = build_pce_text(
            question=self.row["question"],
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
        tokens_left_ratio = max(
            0.0, self.budget_tokens - self.tokens_used
        ) / max(1, self.budget_tokens)
        current_answer_flag = 1.0 if self.current_answer else 0.0

        state = np.array(
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
        return state

    def get_action_mask(self) -> np.ndarray:
        """
        1 表示可选，0 表示 mask 掉
        动作顺序：[continue, accept, prune]
        """
        if self.done:
            return np.array([0, 0, 0], dtype=np.float32)

        progress = self._get_progress()
        has_answer = bool(self.current_answer)

        allow_continue = 1.0
        allow_prune = 1.0

        # accept 需要同时满足：
        # - 已出现显式答案
        # - progress 达到最低门槛
        allow_accept = 1.0 if (has_answer and progress >= self.min_accept_progress) else 0.0

        return np.array([allow_continue, allow_accept, allow_prune], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.done:
            return self._terminal_state(), 0.0, True, {}

        mask = self.get_action_mask()
        if mask[action] < 0.5:
            # safety: 非法动作一律回退为 continue
            action = self.ACTION_CONTINUE

        current_step = self.row["steps"][self.idx]
        step_tokens = count_tokens(current_step)

        # 每步 token cost
        reward = - self.lambda_token * (step_tokens / max(1, self.budget_tokens))

        # ========== accept ==========
        if action == self.ACTION_ACCEPT:
            final_answer = compute_final_answer(self.prefix_items)
            is_correct = bool(
                is_correct_prediction(
                    final_answer,
                    self.row["gold_answer"],
                    answer_mode=self.answer_mode,
                )
            )
            reward += 1.0 if is_correct else -self.wrong_accept_penalty
            self.done = True
            info = {
                "action": "accept",
                "is_correct": int(is_correct),
                "tokens_used": self.tokens_used,
                "progress": self._get_progress(),
            }
            return self._terminal_state(), reward, True, info

        # ========== prune ==========
        if action == self.ACTION_PRUNE:
            reward += self.prune_bad_bonus if (not self.full_traj_correct) else -self.prune_good_penalty
            self.done = True
            info = {
                "action": "prune",
                "is_correct": 0,
                "tokens_used": self.tokens_used,
                "progress": self._get_progress(),
            }
            return self._terminal_state(), reward, True, info

        # ========== continue ==========
        self.idx += 1

        # budget exceed
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
            info = {
                "action": "budget_exceeded",
                "is_correct": int(is_correct),
                "tokens_used": self.tokens_used,
                "progress": self._get_progress(),
            }
            return self._terminal_state(), reward, True, info

        # 自然到结尾
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
            info = {
                "action": "end",
                "is_correct": int(is_correct),
                "tokens_used": self.tokens_used,
                "progress": self._get_progress(),
            }
            return self._terminal_state(), reward, True, info

        next_state = self._advance_until_decision_point()
        info = {
            "action": "continue",
            "is_correct": 0,
            "tokens_used": self.tokens_used,
            "progress": self._get_progress(),
        }
        return next_state, reward, False, info


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


def masked_categorical(policy: PolicyNet, state: np.ndarray, action_mask: np.ndarray):
    x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    logits = policy(x).squeeze(0)

    mask = torch.tensor(action_mask, dtype=torch.float32)
    masked_logits = logits.masked_fill(mask < 0.5, -1e9)

    dist = torch.distributions.Categorical(logits=masked_logits)
    return dist, masked_logits


# =========================
# Train / Eval
# =========================

def run_episode(env: RLControllerEnv, policy: PolicyNet, row: Dict[str, Any], train: bool = True):
    state = env.reset(row)

    log_probs = []
    rewards = []
    traj_infos = []
    action_trace = []

    done = False
    while not done:
        action_mask = env.get_action_mask()
        dist, masked_logits = masked_categorical(policy, state, action_mask)

        if train:
            action = dist.sample()
        else:
            action = torch.argmax(masked_logits, dim=-1)

        action_int = int(action.item())
        log_probs.append(dist.log_prob(action))
        action_trace.append(RLControllerEnv.ACTION_NAMES[action_int])

        next_state, reward, done, info = env.step(action_int)
        rewards.append(float(reward))
        traj_infos.append(info)
        state = next_state

    final_info = traj_infos[-1] if traj_infos else {}
    final_info["action_trace"] = action_trace
    return log_probs, rewards, final_info


def discounted_returns(rewards: List[float], gamma: float = 1.0) -> torch.Tensor:
    out = []
    ret = 0.0
    for r in reversed(rewards):
        ret = r + gamma * ret
        out.append(ret)
    out.reverse()
    x = torch.tensor(out, dtype=torch.float32)
    if len(x) > 1:
        x = (x - x.mean()) / (x.std() + 1e-8)
    return x


def evaluate_policy(
    policy: PolicyNet,
    pce_model,
    rows: List[Dict[str, Any]],
    dataset: str,
    budget_tokens: int,
    lambda_token: float,
    prune_good_penalty: float,
    prune_bad_bonus: float,
    wrong_accept_penalty: float,
    budget_exceed_penalty: float,
    min_accept_progress: float,
) -> Dict[str, Any]:
    env = RLControllerEnv(
        pce_model=pce_model,
        dataset=dataset,
        budget_tokens=budget_tokens,
        lambda_token=lambda_token,
        prune_good_penalty=prune_good_penalty,
        prune_bad_bonus=prune_bad_bonus,
        wrong_accept_penalty=wrong_accept_penalty,
        budget_exceed_penalty=budget_exceed_penalty,
        min_accept_progress=min_accept_progress,
    )

    results = []
    terminal_counter = Counter()

    for row in rows:
        _, rewards, info = run_episode(env, policy, row, train=False)
        terminal_action = info.get("action", "unknown")
        terminal_counter[terminal_action] += 1

        results.append(
            {
                "sample_id": row.get("sample_id", ""),
                "reward_sum": float(sum(rewards)),
                "is_correct": int(info.get("is_correct", 0)),
                "tokens_used": float(info.get("tokens_used", 0.0)),
                "terminal_action": terminal_action,
                "progress": float(info.get("progress", 0.0)),
                "action_trace": info.get("action_trace", []),
            }
        )

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


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_checkpoint(path: Path, policy: PolicyNet, extra: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            **extra,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "strategyqa", "hotpotqa"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--budget_tokens", type=int, default=128)

    parser.add_argument("--train_path", default=None)
    parser.add_argument("--eval_path", default=None)

    parser.add_argument("--train_limit", type=int, default=300)
    parser.add_argument("--eval_limit", type=int, default=200)

    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # reward params
    parser.add_argument("--lambda_token", type=float, default=0.05)
    parser.add_argument("--prune_good_penalty", type=float, default=1.0)
    parser.add_argument("--prune_bad_bonus", type=float, default=0.2)
    parser.add_argument("--wrong_accept_penalty", type=float, default=0.5)
    parser.add_argument("--budget_exceed_penalty", type=float, default=0.3)

    # action mask
    parser.add_argument("--min_accept_progress", type=float, default=0.15)

    # model selection
    parser.add_argument("--tradeoff_coef", type=float, default=0.1)

    parser.add_argument("--out_dir", default="outputs/rl")
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

    policy = PolicyNet(input_dim=6, hidden_dim=args.hidden_dim, num_actions=3)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    env = RLControllerEnv(
        pce_model=pce_model,
        dataset=args.dataset,
        budget_tokens=args.budget_tokens,
        lambda_token=args.lambda_token,
        prune_good_penalty=args.prune_good_penalty,
        prune_bad_bonus=args.prune_bad_bonus,
        wrong_accept_penalty=args.wrong_accept_penalty,
        budget_exceed_penalty=args.budget_exceed_penalty,
        min_accept_progress=args.min_accept_progress,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_acc = -1e9
    best_reward = -1e9
    best_tradeoff = -1e9

    history = []

    for ep in range(1, args.episodes + 1):
        row = random.choice(train_rows)

        log_probs, rewards, _ = run_episode(env, policy, row, train=True)
        returns = discounted_returns(rewards, gamma=args.gamma)

        loss = 0.0
        for lp, G in zip(log_probs, returns):
            loss = loss + (-lp * G)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        eval_result = evaluate_policy(
            policy=policy,
            pce_model=pce_model,
            rows=eval_rows,
            dataset=args.dataset,
            budget_tokens=args.budget_tokens,
            lambda_token=args.lambda_token,
            prune_good_penalty=args.prune_good_penalty,
            prune_bad_bonus=args.prune_bad_bonus,
            wrong_accept_penalty=args.wrong_accept_penalty,
            budget_exceed_penalty=args.budget_exceed_penalty,
            min_accept_progress=args.min_accept_progress,
        )

        summary = eval_result["summary"]
        eval_acc = float(summary["final_accuracy"])
        eval_avg_reward = float(summary["avg_reward"])
        eval_tradeoff = tradeoff_score(summary, args.budget_tokens, args.tradeoff_coef)

        epoch_record = {
            "episode": ep,
            "train_loss": float(loss.detach().item()),
            "eval_accuracy": eval_acc,
            "eval_avg_tokens": float(summary["avg_tokens"]),
            "eval_avg_reward": eval_avg_reward,
            "eval_tradeoff": eval_tradeoff,
            "terminal_action_counter": summary["terminal_action_counter"],
        }
        history.append(epoch_record)

        # 每轮都写 history，方便你随时看
        save_json(out_dir / "training_history.json", {"history": history})

        print(
            f"[ep {ep:03d}] "
            f"loss={float(loss.detach().item()):.4f} "
            f"eval_acc={eval_acc:.4f} "
            f"eval_tokens={float(summary['avg_tokens']):.2f} "
            f"eval_reward={eval_avg_reward:.4f} "
            f"eval_tradeoff={eval_tradeoff:.4f}"
        )

        extra = {
            "dataset": args.dataset,
            "budget_tokens": args.budget_tokens,
            "checkpoint": args.checkpoint,
            "episode": ep,
            "summary": summary,
            "lambda_token": args.lambda_token,
            "prune_good_penalty": args.prune_good_penalty,
            "prune_bad_bonus": args.prune_bad_bonus,
            "wrong_accept_penalty": args.wrong_accept_penalty,
            "budget_exceed_penalty": args.budget_exceed_penalty,
            "min_accept_progress": args.min_accept_progress,
            "tradeoff_coef": args.tradeoff_coef,
        }

        if eval_acc > best_acc:
            best_acc = eval_acc
            save_checkpoint(out_dir / "best_acc.pt", policy, extra)
            save_json(out_dir / "best_acc_summary.json", summary)

        if eval_avg_reward > best_reward:
            best_reward = eval_avg_reward
            save_checkpoint(out_dir / "best_reward.pt", policy, extra)
            save_json(out_dir / "best_reward_summary.json", summary)

        if eval_tradeoff > best_tradeoff:
            best_tradeoff = eval_tradeoff
            save_checkpoint(out_dir / "best_tradeoff.pt", policy, extra)
            save_json(out_dir / "best_tradeoff_summary.json", summary)

        # latest
        save_checkpoint(out_dir / "latest.pt", policy, extra)
        save_json(out_dir / "latest_summary.json", summary)

    print(f"Saved RL artifacts to {out_dir}")


if __name__ == "__main__":
    main()