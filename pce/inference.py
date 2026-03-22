from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse, pickle
from utils.io import read_yaml
from pce.models.verifier import VerifierPCE

def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--answer", default="")
    parser.add_argument("--context", default="")
    args = parser.parse_args()
    pce = load_model(args.checkpoint)
    out = pce.predict(args.question, args.prefix, args.answer, args.context, budget_state={"tokens_left": 128})
    print(out)

if __name__ == "__main__":
    main()
