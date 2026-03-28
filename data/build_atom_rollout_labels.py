from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generator.api import APIGenerator
from generator.local import LocalGenerator
from utils.io import ensure_dir, read_jsonl, read_yaml, write_jsonl


# =========================
# 基础工具
# =========================


def set_seed(seed: int) -> None:
    random.seed(seed)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def attach_prefix_progress(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    对每条 prefix 补充：
    - trajectory_total_units
    - prefix_progress = prefix_num_units / trajectory_total_units
    """
    total_by_traj: Dict[str, int] = {}

    for r in rows:
        tid = r.get("trajectory_id", "")
        curr = int(r.get("prefix_num_units", 0))
        total_by_traj[tid] = max(total_by_traj.get(tid, 0), curr)

    out: List[Dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        tid = rr.get("trajectory_id", "")
        curr = int(rr.get("prefix_num_units", 0))
        total = max(total_by_traj.get(tid, curr), 1)
        rr["trajectory_total_units"] = total
        rr["prefix_progress"] = float(curr) / float(total)
        out.append(rr)

    return out


def infer_split_from_path(path: str) -> str:
    return Path(path).stem


def infer_level_from_path(path: str) -> str:
    """
    例如：
    data/processed/prefixes/atom_level/gsm8k/train.jsonl -> atom_level
    """
    p = Path(path)
    parts = list(p.parts)
    for i, token in enumerate(parts):
        if token in {"path_level", "step_level", "atom_level"}:
            return token
    return "atom_level"


def infer_dataset_from_path(path: str, fallback: str = "gsm8k") -> str:
    """
    例如：
    data/processed/prefixes/atom_level/gsm8k/train.jsonl -> gsm8k
    """
    p = Path(path)
    parts = list(p.parts)
    for token in parts:
        if token in {"gsm8k", "strategyqa", "hotpotqa"}:
            return token
    return fallback


# =========================
# 生成器构建
# =========================


def build_generator(gen_cfg: Dict[str, Any]):
    backend = str(gen_cfg.get("backend", "vllm")).lower()

    # 优先尝试你项目里已有的构造方式
    if backend in {"api", "openai", "anthropic"}:
        try:
            return APIGenerator(**gen_cfg)
        except TypeError:
            provider = gen_cfg.get("provider", backend)
            model_name = gen_cfg.get(
                "model_name", gen_cfg.get("model_name_or_path", "api-model")
            )
            return APIGenerator(provider=provider, model_name=model_name, **gen_cfg)

    # 本地模型
    try:
        return LocalGenerator(gen_cfg)
    except TypeError:
        return LocalGenerator(**gen_cfg)


def build_continue_from_prefix_prompt(question: str, prefix_text: str) -> str:
    return (
        "You are continuing an existing reasoning trace.\n"
        "Keep the existing prefix unchanged.\n"
        "Continue the reasoning from the current point.\n"
        "At the end, output the final answer in the format:\n"
        "Final Answer: <answer>\n\n"
        f"Question: {question}\n\n"
        "Existing Reasoning Prefix:\n"
        f"{prefix_text}\n\n"
        "Continue the reasoning:\n"
    )


def extract_generated_text(obj: Any) -> str:
    """
    尽量兼容不同 generator 输出格式。
    """
    if obj is None:
        return ""

    if isinstance(obj, str):
        return obj

    if isinstance(obj, dict):
        for key in [
            "text",
            "output",
            "completion",
            "generated_text",
            "raw_text",
            "response",
            "reasoning_text",
        ]:
            if key in obj and obj[key] is not None:
                return str(obj[key])

        # vLLM / 其他 wrapper 可能把 outputs 包在 list 里
        if "outputs" in obj and isinstance(obj["outputs"], list) and obj["outputs"]:
            first = obj["outputs"][0]
            if isinstance(first, dict):
                if "text" in first:
                    return str(first["text"])
            return str(first)

    return str(obj)


def batch_generate(generator, prompts: List[str], **kwargs) -> List[Any]:
    """
    优先调用 generate_many；若没有则退化为逐条 generate_one。
    """
    if hasattr(generator, "generate_many"):
        try:
            return generator.generate_many(prompts, **kwargs)
        except TypeError:
            return generator.generate_many(prompts)

    outs = []
    for p in prompts:
        if hasattr(generator, "generate_one"):
            try:
                outs.append(generator.generate_one(p, **kwargs))
            except TypeError:
                outs.append(generator.generate_one(p))
        else:
            raise AttributeError("Generator has neither generate_many nor generate_one")
    return outs


# =========================
# 答案解析与匹配
# =========================

FINAL_ANSWER_PATTERNS = [
    re.compile(r"Final Answer\s*:\s*(.+)", flags=re.IGNORECASE | re.DOTALL),
    re.compile(r"Answer\s*:\s*(.+)", flags=re.IGNORECASE | re.DOTALL),
]


def parse_final_answer(text: str) -> str:
    text = (text or "").strip()

    for pat in FINAL_ANSWER_PATTERNS:
        m = pat.search(text)
        if m:
            ans = m.group(1).strip()
            ans = ans.splitlines()[0].strip()
            return ans

    # fallback：取最后一个非空行
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return lines[-1]

    return ""


def clean_text(x: str) -> str:
    x = str(x).strip()
    x = x.replace("\u00a0", " ")
    x = re.sub(r"\s+", " ", x)
    return x


def extract_gsm8k_final_answer(answer: str) -> str:
    answer = clean_text(answer)

    # 官方 GSM8K 常见格式：#### 42
    m = re.search(r"####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)", answer)
    if m:
        return m.group(1).replace(",", "")

    # fallback：取最后一个数字
    nums = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", answer)
    if nums:
        return nums[-1].replace(",", "")

    return answer


def normalize_math(answer: str) -> str:
    x = clean_text(answer)

    m = re.search(r"####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)", x)
    if m:
        return m.group(1).replace(",", "")

    x = x.strip("`$ ")
    x = x.replace(",", "")
    x = x.strip(" .,:;!?")
    x = re.sub(r"^(the answer is)\s+", "", x, flags=re.IGNORECASE)
    x = re.sub(r"^(therefore|thus|so)\s*,?\s*", "", x, flags=re.IGNORECASE)

    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", x)
    if nums:
        return nums[-1]

    return x.strip()


def normalize_yesno(answer: str) -> str:
    x = clean_text(answer).lower()
    if "yes" in x:
        return "yes"
    if "no" in x:
        return "no"
    return x


def normalize_text_answer(answer: str) -> str:
    x = clean_text(answer).lower()
    x = re.sub(r"[^\w\s]", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def answers_match(dataset: str, pred_answer: str, gold_answer: str) -> bool:
    dataset = dataset.lower()

    if dataset == "gsm8k":
        return normalize_math(pred_answer) == normalize_math(gold_answer)

    if dataset == "strategyqa":
        return normalize_yesno(pred_answer) == normalize_yesno(gold_answer)

    if dataset == "hotpotqa":
        return normalize_text_answer(pred_answer) == normalize_text_answer(gold_answer)

    # fallback
    return normalize_text_answer(pred_answer) == normalize_text_answer(gold_answer)


# =========================
# 主逻辑
# =========================


def make_rollout_label(
    row: Dict[str, Any],
    rollout_answers: List[str],
    rollout_correct: List[int],
    success_hi: float,
    success_lo: float,
    dataset: str,
) -> Optional[Dict[str, Any]]:
    """
    构造 rollout-based success label。
    中间模糊区间直接返回 None（丢弃）。
    """
    n = max(len(rollout_correct), 1)
    local_success_rate = float(sum(rollout_correct)) / float(n)

    if local_success_rate >= success_hi:
        label_success = 1
    elif local_success_rate <= success_lo:
        label_success = 0
    else:
        return None

    out = dict(row)
    out["local_success_rate"] = local_success_rate
    out["rollout_answers"] = rollout_answers
    out["rollout_correct"] = rollout_correct
    out["label_success"] = int(label_success)
    out["dataset"] = dataset
    return out


def make_repairability_label(
    row: Dict[str, Any],
    local_success_rate: float,
    success_hi: float,
    success_lo: float,
    dataset: str,
) -> Optional[Dict[str, Any]]:
    """
    repairability 的定义：
    - 如果原始 trajectory 最终答错，但 rollout 表现好 => repairability = 1
    - 如果原始 trajectory 最终答错，且 rollout 表现差 => repairability = 0
    - 原始 trajectory 本来就答对：不纳入 repairability 数据
    """
    original_final_answer = str(row.get("final_answer", ""))
    gold_answer = str(row.get("gold_answer", ""))

    original_success = answers_match(dataset, original_final_answer, gold_answer)
    if original_success:
        return None

    if local_success_rate >= success_hi:
        label_repairability = 1
    elif local_success_rate <= success_lo:
        label_repairability = 0
    else:
        return None

    out = dict(row)
    out["local_success_rate"] = local_success_rate
    out["repairability"] = int(label_repairability)
    out["dataset"] = dataset
    return out


def rollout_one_prefix(
    generator,
    dataset: str,
    row: Dict[str, Any],
    num_rollouts: int,
    max_new_tokens: Optional[int] = None,
) -> Tuple[List[str], List[int]]:
    question = str(row.get("question", ""))
    prefix_text = str(row.get("prefix_text", ""))
    gold_answer = str(row.get("gold_answer", ""))

    prompt = build_continue_from_prefix_prompt(question, prefix_text)
    prompts = [prompt] * num_rollouts

    gen_kwargs = {}
    if max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = max_new_tokens

    outputs = batch_generate(generator, prompts, **gen_kwargs)

    rollout_answers: List[str] = []
    rollout_correct: List[int] = []

    for out in outputs:
        text = extract_generated_text(out)
        ans = parse_final_answer(text)
        rollout_answers.append(ans)
        rollout_correct.append(int(answers_match(dataset, ans, gold_answer)))

    return rollout_answers, rollout_correct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="atom_level prefix jsonl")
    parser.add_argument("--output_dir", default="data/processed/labels_rollout")
    parser.add_argument("--generator_config", required=True)
    parser.add_argument(
        "--dataset", default="", choices=["", "gsm8k", "strategyqa", "hotpotqa"]
    )
    parser.add_argument("--num_rollouts", type=int, default=3)
    parser.add_argument("--success_hi", type=float, default=0.67)
    parser.add_argument("--success_lo", type=float, default=0.33)
    parser.add_argument("--min_prefix_progress", type=float, default=0.9)
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    input_path = Path(args.input)
    split = infer_split_from_path(str(input_path))
    level = infer_level_from_path(str(input_path))
    dataset = args.dataset or infer_dataset_from_path(str(input_path))

    if level != "atom_level":
        raise ValueError(f"This script is designed for atom_level, got: {level}")

    rows = read_jsonl(str(input_path))
    rows = attach_prefix_progress(rows)
    # 过滤 prefix_progress
    filtered: List[Dict[str, Any]] = []
    for r in rows:
        prog = safe_float(r.get("prefix_progress", 0.0), 0.0)
        if prog >= args.min_prefix_progress:
            filtered.append(r)

    if args.max_examples > 0:
        filtered = filtered[: args.max_examples]

    print(f"[INFO] input rows = {len(rows)}")
    print(
        f"[INFO] filtered rows (progress >= {args.min_prefix_progress}) = {len(filtered)}"
    )
    print(f"[INFO] dataset={dataset}, split={split}, level={level}")

    gen_cfg = read_yaml(args.generator_config)
    generator = build_generator(gen_cfg)
    if len(filtered) == 0:
        success_out = Path(args.output_dir) / "success" / level / dataset / f"{split}.jsonl"
        repair_out = Path(args.output_dir) / "repairability" / level / dataset / f"{split}.jsonl"
        meta_out = Path(args.output_dir) / "meta" / level / dataset / f"{split}.json"

        ensure_dir(success_out.parent)
        ensure_dir(repair_out.parent)
        ensure_dir(meta_out.parent)

        write_jsonl(str(success_out), [])
        write_jsonl(str(repair_out), [])

        meta = {
            "input": str(input_path),
            "output_dir": args.output_dir,
            "dataset": dataset,
            "split": split,
            "level": level,
            "num_input_rows": len(rows),
            "num_filtered_rows": 0,
            "num_success_rows": 0,
            "num_repair_rows": 0,
            "num_rollouts": args.num_rollouts,
            "success_hi": args.success_hi,
            "success_lo": args.success_lo,
            "min_prefix_progress": args.min_prefix_progress,
                "max_examples": args.max_examples,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "generator_config": args.generator_config,
            "note": "No rows passed min_prefix_progress filter.",
            }
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print("[WARN] No rows passed filter. Saved empty outputs and exit.")
        return
    
    success_rows: List[Dict[str, Any]] = []
    repair_rows: List[Dict[str, Any]] = []

    kept_success = 0
    kept_repair = 0
    dropped_ambiguous = 0

    for idx, row in enumerate(filtered):
        rollout_answers, rollout_correct = rollout_one_prefix(
            generator=generator,
            dataset=dataset,
            row=row,
            num_rollouts=args.num_rollouts,
            max_new_tokens=args.max_new_tokens,
        )

        local_success_rate = float(sum(rollout_correct)) / float(
            max(len(rollout_correct), 1)
        )

        success_label_row = make_rollout_label(
            row=row,
            rollout_answers=rollout_answers,
            rollout_correct=rollout_correct,
            success_hi=args.success_hi,
            success_lo=args.success_lo,
            dataset=dataset,
        )

        if success_label_row is not None:
            success_rows.append(success_label_row)
            kept_success += 1
        else:
            dropped_ambiguous += 1

        repair_row = make_repairability_label(
            row=row,
            local_success_rate=local_success_rate,
            success_hi=args.success_hi,
            success_lo=args.success_lo,
            dataset=dataset,
        )
        if repair_row is not None:
            repair_rows.append(repair_row)
            kept_repair += 1

        if (idx + 1) % 20 == 0:
            print(
                f"[INFO] processed {idx + 1}/{len(filtered)} | "
                f"success_kept={kept_success} | repair_kept={kept_repair} | "
                f"dropped_ambiguous={dropped_ambiguous}"
            )

    # 输出目录
    success_out = Path(args.output_dir) / "success" / level / dataset / f"{split}.jsonl"
    repair_out = (
        Path(args.output_dir) / "repairability" / level / dataset / f"{split}.jsonl"
    )
    meta_out = Path(args.output_dir) / "meta" / level / dataset / f"{split}.json"

    ensure_dir(success_out.parent)
    ensure_dir(repair_out.parent)
    ensure_dir(meta_out.parent)

    write_jsonl(str(success_out), success_rows)
    write_jsonl(str(repair_out), repair_rows)

    meta = {
        "input": str(input_path),
        "output_dir": args.output_dir,
        "dataset": dataset,
        "split": split,
        "level": level,
        "num_input_rows": len(rows),
        "num_filtered_rows": len(filtered),
        "num_success_rows": len(success_rows),
        "num_repair_rows": len(repair_rows),
        "num_rollouts": args.num_rollouts,
        "success_hi": args.success_hi,
        "success_lo": args.success_lo,
        "min_prefix_progress": args.min_prefix_progress,
        "max_examples": args.max_examples,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "generator_config": args.generator_config,
    }
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] saved success labels -> {success_out}")
    print(f"[DONE] saved repairability labels -> {repair_out}")
    print(f"[DONE] saved meta -> {meta_out}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
