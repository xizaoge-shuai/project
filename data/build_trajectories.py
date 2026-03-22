from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from generator.api import APIGenerator
from generator.local import LocalGenerator
from generator.parser import parse_generation_output
from generator.prompts import build_cot_prompt
from generator.utils import count_approx_tokens


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_generator(cfg: Dict[str, Any]):
    backend = cfg.get("backend", None)
    provider = cfg.get("provider", None)

    if provider is not None:
        return APIGenerator(cfg)
    if backend is not None:
        return LocalGenerator(cfg)

    raise ValueError("Generator config must contain either 'backend' or 'provider'.")


def infer_split_from_filename(path: Path) -> str:
    name = path.stem.lower()
    if "train" in name:
        return "train"
    if "val" in name:
        return "val"
    if "valid" in name:
        return "validation"
    if "test" in name:
        return "test"
    return "unknown"


def build_single_prompt(sample: Dict[str, Any]) -> str:
    return build_cot_prompt(
        question=sample["question"],
        task=sample.get("task", "generic"),
        context=sample.get("context", ""),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="统一格式 jsonl 文件路径")
    parser.add_argument("--output_dir", required=True, help="轨迹输出目录")
    parser.add_argument("--generator_config", required=True, help="生成器配置 yaml")
    parser.add_argument(
        "--num_samples", type=int, default=3, help="每个问题采样多少条轨迹"
    )
    parser.add_argument(
        "--max_examples", type=int, default=-1, help="最多处理多少条样本，-1 表示全部"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    gen_cfg = load_yaml(args.generator_config)
    generator = build_generator(gen_cfg)

    samples = read_jsonl(input_path)
    if args.max_examples > 0:
        samples = samples[: args.max_examples]

    split = infer_split_from_filename(input_path)
    dataset_name = input_path.parent.name

    all_rows: List[Dict[str, Any]] = []

    for sample in samples:
        prompt = build_single_prompt(sample)
        prompts = [prompt for _ in range(args.num_samples)]

        outputs = generator.generate_many(prompts)

        for k, out in enumerate(outputs):
            parsed = parse_generation_output(
                text=out.get("text", ""),
                task=sample.get("task", "generic"),
            )

            row = {
                "sample_id": sample["id"],
                "trajectory_id": f'{sample["id"]}_traj_{k}',
                "dataset": dataset_name,
                "split": split,
                "task": sample.get("task", "generic"),
                "question": sample["question"],
                "context": sample.get("context", ""),
                "gold_answer": sample.get("answer", ""),
                "prompt": prompt,
                "raw_text": parsed["raw_text"],
                "reasoning_text": parsed["reasoning_text"],
                "steps": parsed["steps"],
                "num_steps": parsed["num_steps"],
                "final_answer": parsed["final_answer"],
                "generator_backend": out.get("backend", ""),
                "finish_reason": out.get("finish_reason", ""),
                "latency": out.get("latency", 0.0),
                "approx_prompt_tokens": count_approx_tokens(prompt),
                "approx_output_tokens": count_approx_tokens(parsed["raw_text"]),
                "meta": out.get("meta", {}),
            }
            all_rows.append(row)

    output_path = output_dir / dataset_name / f"{split}.jsonl"
    write_jsonl(output_path, all_rows)
    print(f"Saved {len(all_rows)} trajectories to {output_path}")


if __name__ == "__main__":
    main()
