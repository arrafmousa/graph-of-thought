"""Static report for semantic merge judgments and whole-graph quality."""
from __future__ import annotations

from typing import Any

from . import report_html


class SemanticEvaluationReport:
    """Render an offline comparison report for a complete merge evaluation run."""

    def render(
        self,
        *,
        summary: dict[str, Any],
        review_pairs: list[dict[str, Any]],
    ) -> str:
        body = "".join(
            [
                self._methodology(summary),
                self._configurations(summary.get("configurations", [])),
                self._datasets(summary.get("dataset_configurations", [])),
                self._graphs(summary.get("graphs", [])),
                self._pairs(review_pairs),
            ]
        )
        return report_html.document("Semantic merge evaluation", body)

    @staticmethod
    def _methodology(summary: dict[str, Any]) -> str:
        experiment = summary.get("experiment", {})
        pairs = [
            ("datasets", experiment.get("datasets", 0)),
            ("questions", experiment.get("questions", 0)),
            ("unique judged pairs", experiment.get("unique_pairs", 0)),
            ("merge occurrences", experiment.get("merge_occurrences", 0)),
            ("graphs", experiment.get("graphs", 0)),
        ]
        return (
            "<section class=\"panel\"><h2>experiment</h2>"
            f"{report_html.tiles(pairs)}"
            "<p class=\"legend\">Pair judgments compare only the two reasoning prefixes "
            "available at the merge point; final answers are withheld from the judge. "
            "Continuation agreement is then measured independently from completed chains. "
            "The whole-graph score is an F-score over configured merge precision and node "
            "reduction, which is a coverage proxy rather than ground-truth recall.</p></section>"
        )

    @staticmethod
    def _configurations(rows: list[dict[str, Any]]) -> str:
        ordered = sorted(
            rows,
            key=lambda row: row.get("mean_graph_quality_score", 0.0),
            reverse=True,
        )
        values = [
            [
                row.get("heuristic"),
                row.get("threshold"),
                row.get("graphs"),
                row.get("accepted_merges"),
                f"{row.get('mean_semantic_score', 0.0):.3f}",
                SemanticEvaluationReport._percent(
                    row.get("judge_redundancy_safe_probability")
                ),
                SemanticEvaluationReport._percent(
                    row.get("both_answers_parse_valid_probability")
                ),
                SemanticEvaluationReport._percent(row.get("same_final_answer_probability")),
                SemanticEvaluationReport._percent(row.get("different_final_answer_probability")),
                f"{row.get('mean_coverage_proxy', 0.0):.3f}",
                f"{row.get('mean_graph_quality_score', 0.0):.3f}",
            ]
            for row in ordered
        ]
        headers = [
            "heuristic",
            "threshold",
            "graphs",
            "merges",
            "semantic",
            "judge safe",
            "answers parsed",
            "same answer",
            "different answer",
            "coverage",
            "graph quality",
        ]
        return (
            "<section class=\"panel\"><h2>configuration ranking</h2>"
            f"{report_html.table(headers, values)}</section>"
        )

    @staticmethod
    def _datasets(rows: list[dict[str, Any]]) -> str:
        values = [
            [
                row.get("dataset"),
                row.get("heuristic"),
                row.get("threshold"),
                row.get("accepted_merges"),
                f"{row.get('mean_semantic_score', 0.0):.3f}",
                SemanticEvaluationReport._percent(
                    row.get("both_answers_parse_valid_probability")
                ),
                SemanticEvaluationReport._percent(row.get("same_final_answer_probability")),
                f"{row.get('mean_graph_quality_score', 0.0):.3f}",
            ]
            for row in rows
        ]
        return (
            "<section class=\"panel\"><h2>per-dataset comparison</h2>"
            f"{report_html.table(['dataset', 'heuristic', 'threshold', 'merges', 'semantic', 'answers parsed', 'same answer', 'quality'], values)}</section>"
        )

    @staticmethod
    def _graphs(rows: list[dict[str, Any]]) -> str:
        ordered = sorted(
            rows,
            key=lambda row: row.get("graph_quality_score", 0.0),
        )
        values = [
            [
                row.get("dataset"),
                row.get("question_id"),
                row.get("heuristic"),
                row.get("threshold"),
                row.get("accepted_merges"),
                f"{row.get('mean_semantic_score', 0.0):.3f}",
                SemanticEvaluationReport._percent(row.get("same_final_answer_probability")),
                f"{row.get('coverage_proxy', 0.0):.3f}",
                f"{row.get('graph_quality_score', 0.0):.3f}",
                row.get("graph_path"),
            ]
            for row in ordered
        ]
        links = "".join(
            f"<li><a href=\"{report_html.escape(row['report_path'].split('artifacts/reports/', 1)[-1])}\">"
            f"{report_html.escape(row['dataset'])} · {report_html.escape(row['heuristic'])} "
            f"@ {report_html.escape(row['threshold'])} · {report_html.escape(row['question_id'])}</a></li>"
            for row in ordered
            if row.get("report_path")
        )
        return (
            "<section class=\"panel\"><h2>graphs, lowest quality first</h2>"
            f"{report_html.table(['dataset', 'question', 'heuristic', 'threshold', 'merges', 'semantic', 'same answer', 'coverage', 'quality', 'graph JSON'], values)}"
            f"<h2>selected visual reports</h2><ul>{links}</ul></section>"
        )

    @staticmethod
    def _pairs(rows: list[dict[str, Any]]) -> str:
        values = [
            [
                row.get("dataset"),
                row.get("question_id"),
                row.get("heuristic"),
                row.get("threshold"),
                f"{row.get('similarity', 0.0):.4f}",
                row.get("equivalence_label"),
                row.get("equivalence_score"),
                f"{row.get('answer_a_parse_valid')}/{row.get('answer_b_parse_valid')}",
                row.get("same_final_answer"),
                row.get("judge_redundancy_safe"),
                row.get("path_a"),
                row.get("path_b"),
                row.get("critical_difference"),
            ]
            for row in rows
        ]
        headers = [
            "dataset",
            "question",
            "heuristic",
            "threshold",
            "similarity",
            "label",
            "score",
            "answers parsed A/B",
            "same answer",
            "safe",
            "path A",
            "path B",
            "critical difference",
        ]
        return (
            "<section class=\"panel\"><h2>manual merge review</h2>"
            "<p class=\"legend\">This table prioritizes the lowest judge scores. "
            "The complete set is preserved in JSONL.</p>"
            f"{report_html.table(headers, values)}</section>"
        )

    @staticmethod
    def _percent(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.2%}"