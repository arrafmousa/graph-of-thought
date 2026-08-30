"""Class-free codec for compact, dependency-free hidden-state storage.

Hidden vectors are stored as base64-encoded little-endian float32 arrays using
only the standard library (no numpy required on the local machine). Holds no
project-owned class (AGENTS.md section 4).
"""
from __future__ import annotations

import base64
from array import array


def encode(vector: list[float]) -> str:
    """Encode a float vector to a base64 string of float32 bytes."""
    buffer = array("f", vector)
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def decode(data: str) -> list[float]:
    """Decode a base64 float32 string back into a list of floats."""
    buffer = array("f")
    buffer.frombytes(base64.b64decode(data.encode("ascii")))
    return list(buffer)
