from __future__ import annotations
import numpy as np

def temperature_scale(probs, temperature: float = 1.0):
    probs = np.asarray(probs, dtype=float)
    logits = np.log(np.clip(probs, 1e-6, 1 - 1e-6) / np.clip(1 - probs, 1e-6, 1 - 1e-6))
    scaled = 1 / (1 + np.exp(-logits / max(temperature, 1e-6)))
    return scaled
