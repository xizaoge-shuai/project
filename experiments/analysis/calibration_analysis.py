from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils.io import read_jsonl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", default="outputs/figures/reliability.png")
    args = parser.parse_args()
    rows = read_jsonl(args.predictions)
    probs = np.array([float(r["success_prob"]) for r in rows])
    labels = np.array([int(r["label_success"]) for r in rows])
    bins = np.linspace(0, 1, 11)
    xs, ys = [], []
    for i in range(10):
        mask = (probs >= bins[i]) & (probs < bins[i+1] if i < 9 else probs <= bins[i+1])
        if mask.any():
            xs.append(probs[mask].mean())
            ys.append(labels[mask].mean())
    plt.figure()
    plt.plot([0,1], [0,1], linestyle="--")
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved reliability plot to {args.out}")

if __name__ == "__main__":
    main()
