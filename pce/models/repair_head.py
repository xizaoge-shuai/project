from __future__ import annotations

def repair_from_error(error_type: str) -> int:
    return int(error_type in {"arithmetic_error", "unsupported_jump", "missing_evidence"})
