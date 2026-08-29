"""Deterministic, dependency-free plain-text helpers for terminal rendering.

This module contains no project-owned classes; it holds shared text helpers
used by tiles for terminal output (AGENTS.md section 4 permits class-free
modules).
"""
from __future__ import annotations

from typing import Any, Sequence

_BLOCKS = "▁▂▃▄▅▆▇█"


def ascii_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    columns = len(headers)
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i in range(columns):
            widths[i] = max(widths[i], len(str(row[i])))

    def fmt(cells: Sequence[Any]) -> str:
        return " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))

    out = [fmt(headers), "-+-".join("-" * w for w in widths)]
    if rows:
        out.extend(fmt(row) for row in rows)
    else:
        out.append("(no rows)")
    return "\n".join(out)


def sparkline(values: Sequence[float]) -> str:
    nums = [float(v) for v in values]
    if not nums:
        return "(no data)"
    low, high = min(nums), max(nums)
    if high == low:
        return _BLOCKS[len(_BLOCKS) // 2] * len(nums)
    span = high - low
    return "".join(_BLOCKS[int((v - low) / span * (len(_BLOCKS) - 1))] for v in nums)
