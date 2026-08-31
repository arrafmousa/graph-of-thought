"""Class-free HTML/SVG helpers for the graph report (no project-owned class)."""
from __future__ import annotations

import html
from typing import Any

_STYLE = """
body { font-family: system-ui, sans-serif; margin: 0; background: #0f1117; color: #e6e6e6; }
header { padding: 18px 26px; background: #161a23; border-bottom: 1px solid #2a2f3a; }
h1 { margin: 0; font-size: 19px; }
h2 { font-size: 15px; margin: 22px 0 8px; color: #9fb3c8; }
main { padding: 18px 26px; }
.panel { background: #161a23; border: 1px solid #2a2f3a; border-radius: 8px; padding: 12px 16px; margin: 14px 0; overflow-x: auto; }
.tiles { display: flex; flex-wrap: wrap; gap: 10px; }
.tile { background: #0b0d12; border: 1px solid #2a2f3a; border-radius: 8px; padding: 10px 14px; min-width: 130px; }
.tile .label { font-size: 11px; color: #8a94a6; text-transform: uppercase; letter-spacing: .5px; }
.tile .value { font-size: 20px; font-weight: 600; margin-top: 4px; }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
th, td { text-align: left; padding: 5px 9px; border-bottom: 1px solid #2a2f3a; }
th { color: #9fb3c8; }
.q { color: #c9d4e2; font-size: 13px; white-space: pre-wrap; }
.legend { font-size: 12px; color: #8a94a6; margin: 6px 0; }
.bar { height: 12px; background: #4dabf7; border-radius: 3px; display: inline-block; vertical-align: middle; }
"""


def escape(value: Any) -> str:
    return html.escape(str(value))


def document(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{escape(title)}</title><style>{_STYLE}</style></head>"
        f"<body><header><h1>{escape(title)}</h1></header><main>{body}</main></body></html>\n"
    )


def color_for(cluster_id: int) -> str:
    if cluster_id < 0:
        return "#5a6473"
    hue = (cluster_id * 47) % 360
    return f"hsl({hue}, 65%, 55%)"


def tiles(pairs: list[tuple[str, Any]]) -> str:
    cells = "".join(
        f"<div class=\"tile\"><div class=\"label\">{escape(label)}</div>"
        f"<div class=\"value\">{escape(value)}</div></div>"
        for label, value in pairs
    )
    return f"<div class=\"tiles\">{cells}</div>"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body: list[str] = []
    for row in rows:
        cells = "".join(f"<td>{escape(c)}</td>" for c in row)
        body.append(f"<tr>{cells}</tr>")
    if not body:
        body.append(f"<tr><td colspan=\"{len(headers)}\">No data</td></tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def histogram(title: str, counts: dict[str, int]) -> str:
    if not counts:
        return f"<div>{escape(title)}: no data</div>"
    peak = max(counts.values()) or 1
    rows = []
    for key, value in counts.items():
        width = int(200 * value / peak)
        rows.append(
            f"<tr><td>{escape(key)}</td>"
            f"<td><span class=\"bar\" style=\"width:{width}px\"></span> {value}</td></tr>"
        )
    return (
        f"<div class=\"legend\">{escape(title)}</div>"
        f"<table>{''.join(rows)}</table>"
    )


def lanes_svg(lanes: list[dict[str, Any]]) -> str:
    if not lanes:
        return "<div>No chains</div>"
    pad = 60
    step = 26
    row_h = 46
    radius = 7
    max_tokens = max((len(lane["tokens"]) for lane in lanes), default=1)
    width = pad + max_tokens * step + pad
    height = pad + len(lanes) * row_h
    parts = [f"<svg width=\"{width}\" height=\"{height}\">"]
    for r, lane in enumerate(lanes):
        y = pad + r * row_h
        label = f"CoT {lane['chain_id']}"
        parts.append(
            f"<text x=\"6\" y=\"{y + 4}\" fill=\"#9fb3c8\" font-size=\"12\">{escape(label)}</text>"
        )
        tokens = lane["tokens"]
        for i in range(1, len(tokens)):
            x0 = pad + (i - 1) * step
            x1 = pad + i * step
            parts.append(
                f"<line x1=\"{x0}\" y1=\"{y}\" x2=\"{x1}\" y2=\"{y}\" stroke=\"#2a2f3a\" stroke-width=\"2\"/>"
            )
        for i, token in enumerate(tokens):
            x = pad + i * step
            fill = color_for(int(token["cluster_id"]))
            stroke = "#e6e6e6" if token.get("join") else "#0f1117"
            stroke_w = 3 if token.get("join") else 1
            title = (
                f"chain {lane['chain_id']} depth {token['index']} "
                f"cluster {token['cluster_id']} token {token['text']!r}"
            )
            if token.get("terminal"):
                ring = "#3ddc84" if lane.get("correct") else "#ff6b6b"
                parts.append(
                    f"<circle cx=\"{x}\" cy=\"{y}\" r=\"{radius + 3}\" fill=\"none\" "
                    f"stroke=\"{ring}\" stroke-width=\"2\"/>"
                )
            parts.append(
                f"<circle cx=\"{x}\" cy=\"{y}\" r=\"{radius}\" fill=\"{fill}\" "
                f"stroke=\"{stroke}\" stroke-width=\"{stroke_w}\">"
                f"<title>{escape(title)}</title></circle>"
            )
    parts.append("</svg>")
    return "".join(parts)


def consolidated_svg(graph: dict[str, Any]) -> str:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    root_id = graph.get("root_id", -1)
    node_by_id: dict[int, dict[str, Any]] = {node["cluster_id"]: node for node in nodes}
    by_depth: dict[int, list[dict[str, Any]]] = {}
    depth_of = {root_id: -1}
    for node in nodes:
        by_depth.setdefault(node["depth"], []).append(node)
        depth_of[node["cluster_id"]] = node["depth"]
    by_depth.setdefault(-1, []).insert(0, {"cluster_id": root_id, "texts": ["<root>"], "depth": -1})

    pad = 40
    step_x = 96
    row_h = 44
    radius = 11
    positions: dict[int, tuple[int, int]] = {}
    max_slots = 1
    for depth in sorted(by_depth):
        column = sorted(by_depth[depth], key=lambda n: n["cluster_id"])
        max_slots = max(max_slots, len(column))
        for slot, node in enumerate(column):
            x = pad + (depth + 1) * step_x
            y = pad + slot * row_h
            positions[node["cluster_id"]] = (x, y)
    width = pad + (max(by_depth) + 2) * step_x
    height = pad + max_slots * row_h + pad
    parts = [f"<svg width=\"{width}\" height=\"{height}\">"]
    for src, dst in edges:
        if src in positions and dst in positions:
            x0, y0 = positions[src]
            x1, y1 = positions[dst]
            parts.append(
                f"<line x1=\"{x0}\" y1=\"{y0}\" x2=\"{x1}\" y2=\"{y1}\" "
                f"stroke=\"#2a2f3a\" stroke-width=\"1.5\"/>"
            )
    for cluster_id, (x, y) in positions.items():
        fill = color_for(int(cluster_id))
        label = "root" if cluster_id == root_id else str(cluster_id)
        parts.append(
            f"<circle cx=\"{x}\" cy=\"{y}\" r=\"{radius}\" fill=\"{fill}\" "
            f"stroke=\"#0f1117\" stroke-width=\"1\"><title>{escape(_node_title(cluster_id, node_by_id.get(cluster_id)))}</title></circle>"
            f"<text x=\"{x}\" y=\"{y - radius - 3}\" fill=\"#9fb3c8\" font-size=\"10\" "
            f"text-anchor=\"middle\">{escape(label)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _node_title(cluster_id: int, node: Any) -> str:
    """Hover text for a consolidated node: cluster, member chains, and tokens."""
    if not node:
        return f"cluster {cluster_id} (root)"
    members = node.get("members", [])
    chains = sorted({member[0] for member in members})
    tokens = node.get("texts", [])
    pairs = ", ".join(f"chain {member[0]}@{member[1]}" for member in members)
    token_text = " | ".join(str(t) for t in tokens)
    return (
        f"cluster {cluster_id} | chains {chains} | tokens: {token_text} | members: {pairs}"
    )
