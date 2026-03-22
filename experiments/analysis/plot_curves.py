from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--x", default="avg_tokens")
    parser.add_argument("--y", default="accuracy")
    parser.add_argument("--out", default="outputs/figures/frontier.png")
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    plt.figure()
    plt.plot(df[args.x], df[args.y], marker="o")
    plt.xlabel(args.x)
    plt.ylabel(args.y)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved plot to {args.out}")

if __name__ == "__main__":
    main()
