"""Aggregate pair judgments and continuation outcomes into graph-quality metrics."""
from __future__ import annotations

from collections import defaultdict
from typing import Any


class GraphMergeQualityAnalyzer:
    """Score every graph from all of its accepted merge events."""

    def analyze(
        self,
        *,
        occurrences: list[dict[str, Any]],
        judgments: dict[str, dict[str, Any]],
        graphs: list[dict[str, Any]],
        max_judge_score: int,
        semantic_weight: float,
        continuation_weight: float,
        quality_beta: float,
        missing_answer_agreement_score: float,
    ) -> dict[str, Any]:
        enriched = [
            self._enrich(
                occurrence,
                judgments[occurrence["pair_id"]],
                max_judge_score,
            )
            for occurrence in occurrences
        ]
        by_graph: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for occurrence in enriched:
            by_graph[occurrence["graph_id"]].append(occurrence)

        graph_rows = [
            self._graph_row(
                graph,
                by_graph.get(graph["graph_id"], []),
                semantic_weight,
                continuation_weight,
                quality_beta,
                missing_answer_agreement_score,
            )
            for graph in graphs
        ]
        config_rows = self._group_rows(graph_rows, ("heuristic", "threshold"))
        dataset_config_rows = self._group_rows(
            graph_rows, ("dataset", "heuristic", "threshold")
        )
        return {
            "pair_occurrences": enriched,
            "graphs": graph_rows,
            "configurations": config_rows,
            "dataset_configurations": dataset_config_rows,
        }

    @staticmethod
    def _enrich(
        occurrence: dict[str, Any],
        judgment_record: dict[str, Any],
        max_judge_score: int,
    ) -> dict[str, Any]:
        judgment = judgment_record["judgment"]
        same_final_answer = occurrence.get("same_final_answer")
        correct_a = occurrence.get("correct_a")
        correct_b = occurrence.get("correct_b")
        if correct_a is True and correct_b is True:
            correctness_outcome = "both_correct"
        elif correct_a is False and correct_b is False:
            correctness_outcome = "both_incorrect"
        else:
            correctness_outcome = "mixed_or_unknown"
        return {
            **occurrence,
            "equivalence_label": judgment["equivalence_label"],
            "equivalence_score": judgment["equivalence_score"],
            "normalized_equivalence_score": judgment["equivalence_score"] / max_judge_score,
            "judge_redundancy_safe": judgment["redundancy_safe"],
            "judge_confidence": judgment["confidence"],
            "shared_information": judgment["shared_information"],
            "critical_difference": judgment["critical_difference"],
            "same_final_answer": same_final_answer,
            "correctness_outcome": correctness_outcome,
        }

    def _graph_row(
        self,
        graph: dict[str, Any],
        occurrences: list[dict[str, Any]],
        semantic_weight: float,
        continuation_weight: float,
        quality_beta: float,
        missing_answer_agreement_score: float,
    ) -> dict[str, Any]:
        semantic = self._mean(
            [row["normalized_equivalence_score"] for row in occurrences]
        )
        known = [row for row in occurrences if row["same_final_answer"] is not None]
        same = sum(row["same_final_answer"] is True for row in known)
        different = sum(row["same_final_answer"] is False for row in known)
        unavailable = len(occurrences) - len(known)
        continuation = (
            same / len(known) if known else missing_answer_agreement_score
        )
        weight_total = semantic_weight + continuation_weight
        merge_precision = (
            semantic_weight * semantic + continuation_weight * continuation
        ) / weight_total
        coverage = float(graph["stats"]["node_reduction"])
        quality = self._f_score(merge_precision, coverage, quality_beta)
        return {
            **graph,
            "accepted_merges": len(occurrences),
            "mean_semantic_score": semantic,
            "equivalent_merge_probability": self._ratio(
                sum(row["equivalence_label"] == "equivalent" for row in occurrences),
                len(occurrences),
            ),
            "judge_redundancy_safe_probability": self._ratio(
                sum(row["judge_redundancy_safe"] for row in occurrences),
                len(occurrences),
            ),
            "same_final_answer_count": same,
            "different_final_answer_count": different,
            "answer_unavailable_count": unavailable,
            "both_answers_parse_valid_count": sum(
                row.get("answer_a_parse_valid") is True
                and row.get("answer_b_parse_valid") is True
                for row in occurrences
            ),
            "both_answers_parse_valid_probability": self._ratio(
                sum(
                    row.get("answer_a_parse_valid") is True
                    and row.get("answer_b_parse_valid") is True
                    for row in occurrences
                ),
                len(occurrences),
            ),
            "invalid_answer_pair_count": sum(
                row.get("answer_a_parse_valid") is not True
                or row.get("answer_b_parse_valid") is not True
                for row in occurrences
            ),
            "same_final_answer_probability": self._ratio(same, len(known)),
            "different_final_answer_probability": self._ratio(different, len(known)),
            "both_correct_count": sum(
                row["correctness_outcome"] == "both_correct" for row in occurrences
            ),
            "both_incorrect_count": sum(
                row["correctness_outcome"] == "both_incorrect" for row in occurrences
            ),
            "mixed_or_unknown_count": sum(
                row["correctness_outcome"] == "mixed_or_unknown"
                for row in occurrences
            ),
            "merge_precision": merge_precision,
            "coverage_proxy": coverage,
            "graph_quality_score": quality,
        }

    def _group_rows(
        self, rows: list[dict[str, Any]], keys: tuple[str, ...]
    ) -> list[dict[str, Any]]:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[tuple(row[key] for key in keys)].append(row)
        result: list[dict[str, Any]] = []
        for values, group in sorted(groups.items(), key=lambda item: item[0]):
            totals = {
                "accepted_merges": sum(row["accepted_merges"] for row in group),
                "same_final_answer_count": sum(
                    row["same_final_answer_count"] for row in group
                ),
                "different_final_answer_count": sum(
                    row["different_final_answer_count"] for row in group
                ),
                "answer_unavailable_count": sum(
                    row["answer_unavailable_count"] for row in group
                ),
                "both_answers_parse_valid_count": sum(
                    row["both_answers_parse_valid_count"] for row in group
                ),
                "invalid_answer_pair_count": sum(
                    row["invalid_answer_pair_count"] for row in group
                ),
                "both_correct_count": sum(row["both_correct_count"] for row in group),
                "both_incorrect_count": sum(row["both_incorrect_count"] for row in group),
                "mixed_or_unknown_count": sum(
                    row["mixed_or_unknown_count"] for row in group
                ),
            }
            known = (
                totals["same_final_answer_count"]
                + totals["different_final_answer_count"]
            )
            result.append(
                {
                    **dict(zip(keys, values)),
                    "graphs": len(group),
                    **totals,
                    "mean_semantic_score": self._weighted_mean(
                        group, "mean_semantic_score", "accepted_merges"
                    ),
                    "equivalent_merge_probability": self._weighted_mean(
                        group, "equivalent_merge_probability", "accepted_merges"
                    ),
                    "judge_redundancy_safe_probability": self._weighted_mean(
                        group,
                        "judge_redundancy_safe_probability",
                        "accepted_merges",
                    ),
                    "same_final_answer_probability": self._ratio(
                        totals["same_final_answer_count"], known
                    ),
                    "different_final_answer_probability": self._ratio(
                        totals["different_final_answer_count"], known
                    ),
                    "both_answers_parse_valid_probability": self._ratio(
                        totals["both_answers_parse_valid_count"],
                        totals["accepted_merges"],
                    ),
                    "mean_merge_precision": self._mean(
                        [row["merge_precision"] for row in group]
                    ),
                    "mean_coverage_proxy": self._mean(
                        [row["coverage_proxy"] for row in group]
                    ),
                    "mean_graph_quality_score": self._mean(
                        [row["graph_quality_score"] for row in group]
                    ),
                }
            )
        return result

    @staticmethod
    def _f_score(precision: float, coverage: float, beta: float) -> float:
        beta_squared = beta * beta
        denominator = beta_squared * precision + coverage
        if denominator == 0:
            return 0.0
        return (1 + beta_squared) * precision * coverage / denominator

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _weighted_mean(rows, value_key: str, weight_key: str) -> float:
        weighted = [
            (row[value_key], row[weight_key])
            for row in rows
            if row[value_key] is not None and row[weight_key] > 0
        ]
        total_weight = sum(weight for _value, weight in weighted)
        if total_weight == 0:
            return 0.0
        return sum(value * weight for value, weight in weighted) / total_weight

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None