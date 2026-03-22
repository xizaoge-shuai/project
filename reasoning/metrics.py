from __future__ import annotations
from typing import List, Dict

def summarize_trace(trace: dict) -> dict:
    return {
        "final_answer": trace.get("final_answer", ""),
        "is_correct": trace.get("is_correct", 0),
        "tokens": trace.get("tokens", 0),
        "latency": trace.get("latency", 0.0),
        "backtrack_count": trace.get("backtrack_count", 0),
        "avg_confidence": sum(trace.get("confidences", []) or [0.0]) / max(len(trace.get("confidences", [])), 1),
    }
