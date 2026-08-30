"""Tests for the graph HTML report library."""
from __future__ import annotations

from libs.graph_report import GraphHtmlReport
from libs.graph_report import report_html


def _graph() -> dict:
    return {
        "heuristic": "hidden_cosine",
        "threshold": 0.95,
        "root_id": -1,
        "dag_valid": True,
        "nodes": [
            {"cluster_id": 0, "depth": 0, "size": 2, "members": [[0, 0], [1, 0]], "texts": ["a"], "terminal": False},
            {"cluster_id": 1, "depth": 1, "size": 1, "members": [[0, 1]], "texts": ["b"], "terminal": True},
        ],
        "edges": [[-1, 0], [0, 1]],
        "merges": [
            {
                "cluster_id": 0,
                "node_a": [0, 0],
                "node_b": [1, 0],
                "similarity": 1.0,
                "winner_chain": 0,
                "loser_chain": 1,
                "winner_recent_mean_logprob": -0.1,
                "loser_recent_mean_logprob": -0.4,
                "token_a": "a",
                "token_b": "a",
            }
        ],
    }


def _lanes() -> list[dict]:
    return [
        {
            "chain_id": 0,
            "correct": True,
            "predicted": "4",
            "tokens": [
                {"index": 0, "text": "a", "cluster_id": 0, "join": True, "terminal": False},
                {"index": 1, "text": "b", "cluster_id": 1, "join": False, "terminal": True},
            ],
        },
        {
            "chain_id": 1,
            "correct": False,
            "predicted": "5",
            "tokens": [
                {"index": 0, "text": "a", "cluster_id": 0, "join": True, "terminal": True},
            ],
        },
    ]


def _stats() -> dict:
    return {
        "raw_token_nodes": 3,
        "consolidated_nodes": 2,
        "node_reduction": 0.33,
        "merge_events": 1,
        "join_nodes": 1,
        "branch_nodes": 1,
        "graph_depth": 1,
        "largest_cluster": 2,
        "histograms": {
            "in_degree": {"0": 1, "1": 1},
            "out_degree": {"1": 1},
            "cluster_size": {"1": 1, "2": 1},
        },
    }


def test_report_renders_all_sections():
    html = GraphHtmlReport().render(
        question={"question_id": "x", "question": "2+2?", "gold_answer": "4", "dataset": "d"},
        lanes=_lanes(),
        graph=_graph(),
        stats=_stats(),
    )
    assert "<svg" in html
    assert "chain-of-thought lanes" in html
    assert "consolidated graph" in html
    assert "merge diagnostics" in html
    assert "2+2?" in html


def test_color_for_root_is_neutral():
    assert report_html.color_for(-1) == "#5a6473"
    assert report_html.color_for(0).startswith("hsl(")
