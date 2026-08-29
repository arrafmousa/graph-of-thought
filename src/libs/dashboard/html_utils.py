"""Deterministic, dependency-free HTML/SVG helpers for dashboard templates.

This module contains no project-owned classes; it holds shared rendering
helpers used by the dashboard template family (AGENTS.md section 4 permits
class-free modules).
"""
from __future__ import annotations

import html
from typing import Any, Iterable, Sequence

_STYLE = """
body { font-family: system-ui, sans-serif; margin: 0; background: #0f1117; color: #e6e6e6; }
header { padding: 20px 28px; background: #161a23; border-bottom: 1px solid #2a2f3a; }
h1 { margin: 0; font-size: 20px; }
h2 { font-size: 16px; margin: 24px 0 8px; color: #9fb3c8; }
main { padding: 20px 28px; }
.tiles { display: flex; flex-wrap: wrap; gap: 12px; }
.tile { background: #161a23; border: 1px solid #2a2f3a; border-radius: 8px; padding: 14px 18px; min-width: 140px; }
.tile .label { font-size: 12px; color: #8a94a6; text-transform: uppercase; letter-spacing: .5px; }
.tile .value { font-size: 22px; font-weight: 600; margin-top: 4px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #2a2f3a; }
th { color: #9fb3c8; }
.error { color: #ff6b6b; }
.chart { background: #161a23; border: 1px solid #2a2f3a; border-radius: 8px; padding: 12px; }
.panel { margin: 18px 0; }
.panel > h2 { margin-bottom: 8px; }
code { background: #0b0d12; padding: 1px 5px; border-radius: 4px; }
"""


def escape(value: Any) -> str:
    return html.escape(str(value))


def document(title: str, body: str, refresh_seconds: int) -> str:
    refresh = (
        f"<meta http-equiv=\"refresh\" content=\"{refresh_seconds}\">"
        if refresh_seconds > 0
        else ""
    )
    return (
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"{refresh}<title>{escape(title)}</title><style>{_STYLE}</style></head>"
        f"<body><header><h1>{escape(title)}</h1></header><main>{body}</main></body></html>\n"
    )


def tiles(pairs: Sequence[tuple[str, Any]]) -> str:
    cells = "".join(
        f"<div class=\"tile\"><div class=\"label\">{escape(label)}</div>"
        f"<div class=\"value\">{escape(value)}</div></div>"
        for label, value in pairs
    )
    return f"<div class=\"tiles\">{cells}</div>"


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{escape(c)}</td>" for c in row)
        body_rows.append(f"<tr>{cells}</tr>")
    if not body_rows:
        body_rows.append(
            f"<tr><td colspan=\"{len(headers)}\">No data</td></tr>"
        )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def line_chart(points: Sequence[tuple[float, float]], label: str) -> str:
    if len(points) < 2:
        return f"<div class=\"chart\">{escape(label)}: insufficient data</div>"
    width, height, pad = 640, 200, 30
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = (max_x - min_x) or 1.0
    span_y = (max_y - min_y) or 1.0

    def sx(x: float) -> float:
        return pad + (x - min_x) / span_x * (width - 2 * pad)

    def sy(y: float) -> float:
        return height - pad - (y - min_y) / span_y * (height - 2 * pad)

    poly = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
    return (
        f"<div class=\"chart\"><div>{escape(label)}</div>"
        f"<svg width=\"{width}\" height=\"{height}\">"
        f"<polyline fill=\"none\" stroke=\"#4dabf7\" stroke-width=\"2\" points=\"{poly}\"/>"
        f"</svg><div style=\"font-size:12px;color:#8a94a6\">"
        f"min={min_y:.4g} max={max_y:.4g}</div></div>"
    )
