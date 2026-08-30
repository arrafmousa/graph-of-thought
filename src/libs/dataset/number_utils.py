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
