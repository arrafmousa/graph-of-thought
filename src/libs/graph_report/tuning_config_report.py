"""Per-configuration tuning dashboard (research plan sections 15, 16, 3.3).

Renders one HTML dashboard for a single (heuristic, threshold) configuration during
the hyperparameter phase: aggregated graph-feature tiles, degree/cluster histograms,
and two merge-sample tables — highest-similarity merges and borderline (near-threshold)
merges — with each chain's recent token context so a human can judge which merges are
semantically sound.
"""
from __future__ import annotations

from typing import Any

from . import report_html

_SAMPLE_HEADERS = [
    "question",
    "a -> b",
    "similarity",
    "token a",
    "token b",
    "context a",
    "context b",
    "winner",
    "win logprob",
    "lose logprob",
]


class TuningConfigReport:
    """Render a dashboard for one swept configuration."""

    def render(
        self,
        *,
        config_label: str,
        stats: dict[str, Any],
        high_samples: list[dict[str, Any]],
        edge_samples: list[dict[str, Any]],
    ) -> str:
        title = f"Tuning — {config_label}"
        body = "".join(
            [
                self._summary(stats),
                self._histograms(stats),
                self._samples("highest-similarity merges", high_samples),
                self._samples("borderline (near-threshold) merges", edge_samples),
            ]
        )
        return report_html.document(title, body)

    def _summary(self, stats: dict[str, Any]) -> str:
        pairs = [
            ("questions", stats.get("questions", 0)),
            ("total merges", stats.get("total_merges", 0)),
            ("mean node reduction", f"{stats.get('mean_node_reduction', 0.0):.2%}"),
            ("join nodes", stats.get("join_nodes", 0)),
            ("branch nodes", stats.get("branch_nodes", 0)),
            ("mean d_in", f"{stats.get('mean_in_degree', 0.0):.3f}"),
            ("mean d_out", f"{stats.get('mean_out_degree', 0.0):.3f}"),
            ("max d_in", stats.get("max_in_degree", 0)),
            ("largest cluster", stats.get("largest_cluster", 0)),
            ("all DAG valid", stats.get("dag_valid", "?")),
        ]
        return f"<section class=\"panel\"><h2>aggregated features</h2>{report_html.tiles(pairs)}</section>"

    def _histograms(self, stats: dict[str, Any]) -> str:
        histograms = stats.get("histograms", {})
        blocks = [
            report_html.histogram("in-degree (d_in)", histograms.get("in_degree", {})),
            report_html.histogram("out-degree (d_out)", histograms.get("out_degree", {})),
            report_html.histogram("cluster size", histograms.get("cluster_size", {})),
        ]
        return "<section class=\"panel\"><h2>histograms</h2>" + "".join(blocks) + "</section>"

    def _samples(self, heading: str, samples: list[dict[str, Any]]) -> str:
        rows = []
        for sample in samples:
            rows.append(
                [
                    sample.get("question_id"),
                    f"{sample.get('node_a')} -> {sample.get('node_b')}",
                    f"{sample.get('similarity', 0.0):.4f}",
                    sample.get("token_a"),
                    sample.get("token_b"),
                    sample.get("context_a"),
                    sample.get("context_b"),
                    sample.get("winner_chain"),
                    f"{sample.get('winner_recent_mean_logprob', 0.0):.3f}",
                    f"{sample.get('loser_recent_mean_logprob', 0.0):.3f}",
                ]
            )
        return (
            f"<section class=\"panel\"><h2>{report_html.escape(heading)}</h2>"
            f"{report_html.table(_SAMPLE_HEADERS, rows)}</section>"
        )
