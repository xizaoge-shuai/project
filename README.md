# Prefix Confidence Estimator (PCE) Reasoning Control Project

一个面向黑盒/开源大模型推理控制的实验脚手架。

## 支持的主流程

1. 下载/整理数据
2. 生成 reasoning trajectories
3. 切分为 path / step / atom 级前缀
4. 自动构造 success / error type / repairability 标签
5. 训练 PCE
6. 用 PCE + Controller 做在线推理控制
7. 运行 baseline / cross-model / overhead / ablation 实验

## 快速开始

```bash
bash scripts/prepare_data.sh
python data/build_trajectories.py --dataset gsm8k --generator dummy
python data/build_prefixes.py --dataset gsm8k --level step
python data/build_labels.py --dataset gsm8k --level step
python pce/train.py --config configs/model/pce_mlp.yaml --dataset gsm8k --level step
python experiments/run_pce.py --dataset gsm8k --controller threshold
```

## 说明

- 这是一版“能跑通的第一版”工程，不追求把所有复杂方法都一次写满。
- `local.py`、`api.py`、`learned.py`、`gg_like.py` 都提供了可运行的基础实现或占位实现，方便后续替换。
- 默认包含一个 `DummyGenerator`，即便没有接入真实 LLM 也能走通整条实验流水线。
