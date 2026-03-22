from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
from utils.io import read_jsonl, write_jsonl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_traces", required=True, help="训练 PCE 用的轨迹文件标识，例如 dummy-a")
    parser.add_argument("--test_traces", required=True, help="测试轨迹文件标识，例如 dummy-b")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--level", default="step")
    parser.add_argument("--report", default="outputs/metrics/cross_model_report.jsonl")
    args = parser.parse_args()

    # 第一版：记录实验计划与输入来源。
    # 实际运行时，可以在不同 generator 标识下各自生成 trajectories / labels / PCE checkpoint，再统一评估。
    report = [{
        "dataset": args.dataset,
        "level": args.level,
        "train_generator": args.train_traces,
        "test_generator": args.test_traces,
        "note": "请分别用不同 generator 构建 trajectories 和 labels，然后训练/测试对应 checkpoint。",
    }]
    write_jsonl(args.report, report)
    print(f"Saved cross-model plan to {args.report}")

if __name__ == "__main__":
    main()
