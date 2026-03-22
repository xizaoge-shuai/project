from __future__ import annotations
import numpy as np

class SuccessHead:
    def __call__(self, logits):
        return 1 / (1 + np.exp(-logits))
