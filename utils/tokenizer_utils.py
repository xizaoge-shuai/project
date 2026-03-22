from __future__ import annotations
import re
from typing import List

def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())

def count_tokens(text: str) -> int:
    return len(simple_tokenize(text))
