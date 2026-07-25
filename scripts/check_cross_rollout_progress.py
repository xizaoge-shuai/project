from __future__ import annotations

import json
from pathlib import Path

datasets = ["svamp", "asdiv", "math500", "mathqa"]
splits = {
    "train": 500,
    "val": 200,
    "test": 500,
}
settings = {
    "p09": Path("data/processed/labels_cross_table_rollout_p09"),
    "p08": Path("data/processed/labels_cross_table_rollout_p08"),
}

print(
    "| Threshold | Dataset | Split | Actual | Expected max | Status |"
)
print("|---|---|---|---:|---:|---|")

finished = 0
total = len(settings) * len(datasets) * len(splits)

for tag, root in settings.items():
    for dataset in datasets:
        for split, expected in splits.items():
            success = (
                root
                / "success"
                / "atom_level"
                / dataset
                / f"{split}.jsonl"
            )
            meta = (
                root
                / "meta"
                / "atom_level"
                / dataset
                / f"{split}.json"
            )

            actual = 0
            if success.exists():
                with success.open(encoding="utf-8") as f:
                    actual = sum(1 for line in f if line.strip())

            meta_done = False
            filtered = None

            if meta.exists():
                try:
                    data = json.loads(meta.read_text(encoding="utf-8"))
                    filtered = data.get("num_filtered_rows")
                    meta_done = True
                except Exception:
                    pass

            expected_actual = (
                min(expected, filtered)
                if filtered is not None
                else expected
            )

            if meta_done and success.exists():
                status = "DONE"
                finished += 1
            elif success.exists() and actual > 0:
                status = "PARTIAL"
            else:
                status = "PENDING"

            print(
                f"| {tag} | {dataset} | {split} | "
                f"{actual} | {expected_actual} | {status} |"
            )

print()
print(f"Completed stages: {finished}/{total}")
print(f"Stage progress: {finished / total * 100:.1f}%")
