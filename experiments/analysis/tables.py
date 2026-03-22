from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    print(df.to_latex(index=False, float_format=lambda x: f"{x:.4f}"))

if __name__ == "__main__":
    main()
