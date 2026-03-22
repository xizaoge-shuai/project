from __future__ import annotations
import re
from typing import List

def step_level_atoms(steps: List[str]) -> List[str]:
    return [s.strip() for s in steps if s and s.strip()]

def atom_level_atoms(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?。])\s+|(?<=, )|(?<=，)", text or "")
    return [p.strip() for p in parts if p.strip()]
