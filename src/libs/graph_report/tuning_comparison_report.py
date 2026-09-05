"""Cross-configuration comparison dashboard for the tuning phase.

Renders the comparison table over every swept (heuristic, threshold) configuration
with links to each per-configuration dashboard, plus the discoverable vocabulary of
available heuristics/policies so the user can pick what to carry into the full run.
"""
from __future__ import annotations

from typing import Any

from . import report_html

_HEADERS = [
    "heuristic",
    "threshold",
    "total merges",
    "mean node reduction",
    "join nodes",
    "branch nodes",
    "mean d_in",
    "mean d_out",
    "max d_in",
    "all DAG valid",
    "dashboard",
]


class TuningComparisonReport:
    """Render the comparison table across all swept configurations."""

    def render(
        self,
        *,
        rows: list[dict[str, Any]],
        available: dict[str, list[str]],
    ) -> str:
        title = "Tuning — configuration comparison"
        body = "".join([self._intro(available), self._table(rows)])
        return report_html.document(title, body)

    def _intro(self, available: dict[str, list[str]]) -> str:
        pairs = [(name, ", ".join(values)) for name, values in available.items()]
        rows = [f"<tr><td>{report_html.escape(n)}</td><td>{report_html.escape(v)}</td></tr>" for n, v in pairs]
        return (
            "<section class=\"panel\"><h2>selectable vocabulary (enums)</h2>"
            "<div class=\"legend\">Pick a heuristic + threshold below and set them in the "
            "full-run config (<code>configs/graph_pipeline/&lt;experiment&gt;/*.json</code>).</div>"
            f"<table>{''.join(rows)}</table></section>"
        )

    def _table(self, rows: list[dict[str, Any]]) -> str:
        table_rows = []
        for row in rows:
            label = row.get("config_label", "")
            link = row.get("dashboard_path", "")
            table_rows.append(
                [
                    row.get("heuristic"),
                    row.get("threshold"),
                    row.get("total_merges"),
                    f"{row.get('mean_node_reduction', 0.0):.2%}",
                    row.get("join_nodes"),
                    row.get("branch_nodes"),
                    f"{row.get('mean_in_degree', 0.0):.3f}",
                    f"{row.get('mean_out_degree', 0.0):.3f}",
                    row.get("max_in_degree"),
                    row.get("dag_valid"),
                    self._link(link, label),
                ]
            )
        return (
            "<section class=\"panel\"><h2>comparison table</h2>"
            f"{self._raw_table(_HEADERS, table_rows)}</section>"
        )

    @staticmethod
    def _link(path: str, label: str) -> str:
        if not path:
            return ""
        return f"<a href=\"{report_html.escape(path)}\" style=\"color:#4dabf7\">{report_html.escape(label)}</a>"

    @staticmethod
    def _raw_table(headers: list[str], rows: list[list[Any]]) -> str:
        head = "".join(f"<th>{report_html.escape(h)}</th>" for h in headers)
        body = []
        for row in rows:
            cells = []
            for index, cell in enumerate(row):
                # Last column is a pre-rendered anchor; do not escape it.
                cells.append(f"<td>{cell if index == len(row) - 1 else report_html.escape(cell)}</td>")
            body.append(f"<tr>{''.join(cells)}</tr>")
        if not body:
            body.append(f"<tr><td colspan=\"{len(headers)}\">No data</td></tr>")
        return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
