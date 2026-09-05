"""Class-free numeric parsing helpers shared within the dataset library.

Holds no project-owned class (AGENTS.md section 4 permits class-free modules).
"""
from __future__ import annotations

import re
from typing import Optional

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def last_number(text: str) -> Optional[str]:
    """Return the last number in ``text`` normalized to a canonical string."""
    match = None
    for match in _NUMBER.finditer(text):
        pass
    if match is None:
        return None
    value = match.group(0).replace(",", "").rstrip(".")
    try:
        number = float(value)
    except ValueError:
        return None
    if number.is_integer():
        return str(int(number))
    return repr(number)


def normalize_math_text(text: str) -> str:
    """Normalize superficial LaTeX formatting for exact answer comparison."""
    normalized = text.strip().splitlines()[0] if text.strip() else ""
    normalized = normalized.strip("$ ")
    if normalized.startswith("\\boxed{") and normalized.endswith("}"):
        normalized = normalized[7:-1]
    for token in ("\\left", "\\right", " ", "\t", "\r", "\n"):
        normalized = normalized.replace(token, "")
    return normalized.casefold()


def extract_delimited(text: str, delimiter: str) -> Optional[str]:
    """Return the answer wrapped by ``delimiter``.

    Prefers the region between the last delimiter pair (``#### answer ####``);
    falls back to the suffix after a single delimiter (``#### answer``).
    """
    if not delimiter or delimiter not in text:
        return None
    parts = text.split(delimiter)
    region = parts[-2] if len(parts) >= 3 else parts[-1]
    region = region.strip()
    return region or None


def has_delimited_pair(text: str, delimiter: str) -> bool:
    """Whether ``text`` contains a closing pair of ``delimiter``."""
    return bool(delimiter) and text.count(delimiter) >= 2
