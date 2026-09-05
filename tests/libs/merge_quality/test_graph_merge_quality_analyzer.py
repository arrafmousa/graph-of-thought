"""Tests for continuation and whole-graph merge quality statistics."""
from __future__ import annotations

import pytest

from libs.merge_quality import GraphMergeQualityAnalyzer


def test_analyzer_reports_same_and_different_continuations():
    occurrences = [
        {
            "pair_id": "same",
            "graph_id": "g1",
            "dataset": "math",
            "question_id": "q1",
            "heuristic": "hidden_cosine",
            "threshold": 0.95,
            "predicted_a": "4",
            "predicted_b": "4",
            "correct_a": True,
            "correct_b": True,
            "answer_a_parse_valid": True,
            "answer_b_parse_valid": True,
            "same_final_answer": True,
        },
        {
            "pair_id": "different",
            "graph_id": "g1",
            "dataset": "math",
            "question_id": "q1",
            "heuristic": "hidden_cosine",
            "threshold": 0.95,
            "predicted_a": "4",
            "predicted_b": "5",
            "correct_a": True,
            "correct_b": False,
            "answer_a_parse_valid": True,
            "answer_b_parse_valid": True,
            "same_final_answer": False,
        },
    ]
    judgments = {
        "same": {
            "judgment": {
                "equivalence_label": "equivalent",
                "equivalence_score": 4,
                "redundancy_safe": True,
                "confidence": 0.9,
                "shared_information": "same sum",
                "critical_difference": "",
            }
        },
        "different": {
            "judgment": {
                "equivalence_label": "partial",
                "equivalence_score": 2,
                "redundancy_safe": False,
                "confidence": 0.8,
                "shared_information": "same setup",
                "critical_difference": "different arithmetic",
            }
        },
    }
    graphs = [
        {
            "graph_id": "g1",
            "dataset": "math",
            "question_id": "q1",
            "heuristic": "hidden_cosine",
            "threshold": 0.95,
            "stats": {"node_reduction": 0.25},
        }
    ]

    result = GraphMergeQualityAnalyzer().analyze(
        occurrences=occurrences,
        judgments=judgments,
        graphs=graphs,
        max_judge_score=4,
        semantic_weight=1.0,
        continuation_weight=1.0,
        quality_beta=1.0,
        missing_answer_agreement_score=0.0,
    )

    graph = result["graphs"][0]
    assert graph["same_final_answer_probability"] == 0.5
    assert graph["different_final_answer_probability"] == 0.5
    assert graph["mean_semantic_score"] == 0.75
    assert graph["merge_precision"] == 0.625
    assert graph["graph_quality_score"] == pytest.approx(2 * 0.625 * 0.25 / 0.875)
    assert result["configurations"][0]["accepted_merges"] == 2


def test_analyzer_handles_graph_without_merges():
    result = GraphMergeQualityAnalyzer().analyze(
        occurrences=[],
        judgments={},
        graphs=[
            {
                "graph_id": "empty",
                "dataset": "math",
                "question_id": "q1",
                "heuristic": "hidden_cosine",
                "threshold": 0.99,
                "stats": {"node_reduction": 0.0},
            }
        ],
        max_judge_score=4,
        semantic_weight=1.0,
        continuation_weight=1.0,
        quality_beta=1.0,
        missing_answer_agreement_score=0.0,
    )

    graph = result["graphs"][0]
    assert graph["accepted_merges"] == 0
    assert graph["same_final_answer_probability"] is None
    assert graph["graph_quality_score"] == 0.0