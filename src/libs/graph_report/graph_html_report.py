"""Assemble the standalone HTML graph report for one question (research plan section 17)."""
from __future__ import annotations

from typing import Any

from . import report_html


class GraphHtmlReport:
    """Render lanes, consolidated DAG, histograms, and a merge table to HTML."""

    def render(
        self,
        *,
        question: dict[str, Any],
        lanes: list[dict[str, Any]],
        graph: dict[str, Any],
        stats: dict[str, Any],
    ) -> str:
        title = (
            f"Reasoning graph — {question.get('question_id', '?')} "
            f"[{graph.get('heuristic', '?')} @ {graph.get('threshold', '?')}]"
        )
        body = "".join(
            [
                self._question_panel(question, graph),
                self._summary_panel(stats),
                self._lanes_panel(lanes),
                self._graph_panel(graph),
                self._histograms_panel(stats),
                self._merge_panel(graph.get("merges", [])),
            ]
        )
        return report_html.document(title, body)

    def _question_panel(self, question: dict[str, Any], graph: dict[str, Any]) -> str:
        rows = [
            ("dataset", question.get("dataset", "?")),
            ("gold answer", question.get("gold_answer", "?")),
            ("heuristic", graph.get("heuristic", "?")),
            ("threshold", graph.get("threshold", "?")),
            ("DAG valid", graph.get("dag_valid", "?")),
        ]
        return (
            "<section class=\"panel\"><h2>question</h2>"
            f"<div class=\"q\">{report_html.escape(question.get('question', ''))}</div>"
            f"{report_html.tiles(rows)}</section>"
        )

    def _summary_panel(self, stats: dict[str, Any]) -> str:
        pairs = [
            ("raw nodes", stats.get("raw_token_nodes", 0)),
            ("graph nodes", stats.get("consolidated_nodes", 0)),
            ("node reduction", f"{stats.get('node_reduction', 0.0):.2%}"),
            ("merges", stats.get("merge_events", 0)),
            ("join nodes", stats.get("join_nodes", 0)),
            ("branch nodes", stats.get("branch_nodes", 0)),
            ("graph depth", stats.get("graph_depth", 0)),
            ("largest cluster", stats.get("largest_cluster", 0)),
        ]
        return f"<section class=\"panel\"><h2>summary</h2>{report_html.tiles(pairs)}</section>"

    def _lanes_panel(self, lanes: list[dict[str, Any]]) -> str:
        legend = (
            "<div class=\"legend\">Each row is one sampled chain; token color = cluster "
            "(shared color across rows = a join). Bold ring = merged/join node; "
            "green/red ring = correct/incorrect terminal.</div>"
        )
        return (
            "<section class=\"panel\"><h2>chain-of-thought lanes (view A)</h2>"
            f"{legend}{report_html.lanes_svg(lanes)}</section>"
        )

    def _graph_panel(self, graph: dict[str, Any]) -> str:
        return (
            "<section class=\"panel\"><h2>consolidated graph (view B)</h2>"
            f"{report_html.consolidated_svg(graph)}</section>"
        )

    def _histograms_panel(self, stats: dict[str, Any]) -> str:
        histograms = stats.get("histograms", {})
        blocks = [
            report_html.histogram("in-degree", histograms.get("in_degree", {})),
            report_html.histogram("out-degree", histograms.get("out_degree", {})),
            report_html.histogram("cluster size", histograms.get("cluster_size", {})),
        ]
        return (
            "<section class=\"panel\"><h2>histograms (view C)</h2>"
            + "".join(blocks)
            + "</section>"
        )

    def _merge_panel(self, merges: list[dict[str, Any]]) -> str:
        headers = [
            "cluster",
            "node a",
            "node b",
            "similarity",
            "winner",
            "loser",
            "win logprob",
            "lose logprob",
            "token a",
            "token b",
        ]
        rows = []
        for merge in merges:
            rows.append(
                [
                    merge.get("cluster_id"),
                    merge.get("node_a"),
                    merge.get("node_b"),
                    f"{merge.get('similarity', 0.0):.4f}",
                    merge.get("winner_chain"),
                    merge.get("loser_chain"),
                    f"{merge.get('winner_recent_mean_logprob', 0.0):.3f}",
                    f"{merge.get('loser_recent_mean_logprob', 0.0):.3f}",
                    merge.get("token_a"),
                    merge.get("token_b"),
                ]
            )
        return (
            "<section class=\"panel\"><h2>merge diagnostics</h2>"
            f"{report_html.table(headers, rows)}</section>"
        )
