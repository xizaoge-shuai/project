# AutoDL GSM8K Full Experiment Log - 2026-05-10

## Dataset

- Dataset: official GSM8K full test
- Samples: 1319
- Reasoning trajectories: 3957
- Prefix predictions: 71521
- Main PCE: Atom-level rollout p08 light raw
- Main generator: Qwen2.5-7B-Instruct
- Main repair setting: safev3, tau=0.46, rewrite_window=1

## Main Results

| Method | Accuracy | Note |
|---|---:|---|
| First trajectory | 0.8021 | single first trajectory |
| Majority voting | 0.8886 | sample-level majority |
| PCE weighted_tail5 | 0.8931 | PCE as soft aggregation weight |
| Safe repair + weighted_tail5 | 0.8984 | tau=0.46 safev3 |
| Selective judge all_disagree | 0.9105 | current best |
| Oracle any-after | 0.9401 | upper bound |

## Key Findings

1. PCE is not reliable as a hard top-1 trajectory selector, but it is useful as a soft aggregation weight.
2. Safe local rewrite gives a small but positive gain when combined with weighted voting.
3. Error decomposition shows remaining errors include both selection errors and generation errors.
4. Selective answer judge on all-disagree samples gives the best current result: 0.9105.
5. Expanding judge trigger to margin<=0.30 increases broken cases and performs worse than all_disagree.
6. Three-head repairability gate is too conservative under current sparse repairability labels. It triggers very few cases and does not improve the main result.

## Final Decision

Use the following as the current main pipeline:

PCE weighted aggregation + safe local rewrite + selective answer judge.

Three-head repairability is kept as an ablation / negative result, not as the main method.
