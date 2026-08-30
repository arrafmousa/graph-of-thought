"""Class-free vector helpers for latent-merge heuristics (no numpy).

Holds no project-owned class (AGENTS.md section 4). All math is pure Python so
the library imports and runs on a CPU-only machine without numpy.
"""
from __future__ import annotations

import base64
import math
from array import array
from typing import Optional


def decode(data: str) -> list[float]:
    """Decode a base64 float32 string (as written by the generation stage)."""
    buffer = array("f")
    buffer.frombytes(base64.b64decode(data.encode("ascii")))
    return list(buffer)


def cosine(a: Optional[list[float]], b: Optional[list[float]]) -> float:
    """Cosine similarity; returns 0.0 if either vector is missing or zero."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def mean_pool(vectors: list[list[float]]) -> Optional[list[float]]:
    """Element-wise mean of equally sized vectors, or ``None`` if empty."""
    usable = [v for v in vectors if v]
    if not usable:
        return None
    dim = len(usable[0])
    totals = [0.0] * dim
    count = 0
    for vector in usable:
        if len(vector) != dim:
            continue
        for i, value in enumerate(vector):
            totals[i] += value
        count += 1
    if count == 0:
        return None
    return [total / count for total in totals]
