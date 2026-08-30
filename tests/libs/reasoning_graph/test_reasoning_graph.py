"""Tests for the reasoning-graph consolidation library."""
from __future__ import annotations

import json

from libs.reasoning_graph import (
    CandidateFilter,
    GraphConsolidator,
    GraphStatistics,
    HiddenCosineMetric,
    LoadedChain,
    LoadedToken,
    MergeRegistry,
    RepresentativeSelector,
    TokenIdentityMetric,
    TraceLoader,
)
from libs.reasoning_graph import vector_math


def _chains() -> list[LoadedChain]:
    c0 = LoadedChain(chain_id=0, correct=True)
    c0.tokens = [
        LoadedToken(0, 0, 10, "a", -0.1, [1.0, 0.0]),
        LoadedToken(0, 1, 11, "b", -0.2, [0.0, 1.0]),
        LoadedToken(0, 2, 12, "c", -0.3, [1.0, 1.0]),
    ]
    c1 = LoadedChain(chain_id=1, correct=False)
    c1.tokens = [
        LoadedToken(1, 0, 10, "a", -0.4, [1.0, 0.0]),
        LoadedToken(1, 1, 21, "d", -0.5, [-1.0, 0.2]),
        LoadedToken(1, 2, 22, "e", -0.6, [0.9, 0.9]),
    ]
    return [c0, c1]


def _consolidator() -> GraphConsolidator:
    return GraphConsolidator(
        candidate_filter=CandidateFilter(depth_policy="same_depth", max_depth_difference=0),
        representative_selector=RepresentativeSelector(window=8),
    )


def test_consolidation_produces_join_and_valid_dag():
    graph = _consolidator().consolidate(
        chains=_chains(), metric=HiddenCosineMetric(), threshold=0.99, pooling_k=4
    )
    stats = GraphStatistics().compute(graph)
    assert graph.is_dag() is True
    assert stats["dag_valid"] is True
    assert stats["merge_events"] >= 1
    assert stats["join_nodes"] >= 1
    assert stats["raw_token_nodes"] == 6


def test_consolidation_is_deterministic():
    a = _consolidator().consolidate(
        chains=_chains(), metric=HiddenCosineMetric(), threshold=0.99, pooling_k=4
    )
    b = _consolidator().consolidate(
        chains=_chains(), metric=HiddenCosineMetric(), threshold=0.99, pooling_k=4
    )
    assert json.dumps(a.to_dict(), sort_keys=True) == json.dumps(b.to_dict(), sort_keys=True)


def test_high_threshold_prevents_merges():
    graph = _consolidator().consolidate(
        chains=_chains(), metric=HiddenCosineMetric(), threshold=1.01, pooling_k=4
    )
    stats = GraphStatistics().compute(graph)
    assert stats["merge_events"] == 0
    assert stats["consolidated_nodes"] == stats["raw_token_nodes"]


def test_token_identity_metric_needs_no_hidden():
    assert TokenIdentityMetric().requires_hidden() is False
    assert HiddenCosineMetric().requires_hidden() is True


def test_candidate_filter_blocks_same_chain_and_depth():
    filt = CandidateFilter(depth_policy="same_depth", max_depth_difference=0)
    token = LoadedToken(1, 2, 0, "x", -0.1, None)
    rep_same_chain = LoadedToken(1, 2, 0, "y", -0.1, None)
    rep_other = LoadedToken(0, 2, 0, "y", -0.1, None)
    rep_wrong_depth = LoadedToken(0, 3, 0, "y", -0.1, None)
    assert filt.admissible(token, rep_same_chain, {1}) is False
    assert filt.admissible(token, rep_other, {0}) is True
    assert filt.admissible(token, rep_wrong_depth, {0}) is False


def test_vector_math_cosine():
    assert vector_math.cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert vector_math.cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert vector_math.cosine(None, [1.0]) == 0.0


def test_registry_builds_metrics():
    registry = MergeRegistry.with_builtin_metrics()
    assert registry.create("hidden_cosine").name == "hidden_cosine"
    assert registry.create("token_identity").name == "token_identity"


def test_trace_loader_round_trip(tmp_path):
    question_dir = tmp_path / "question_x"
    question_dir.mkdir()
    (question_dir / "question.json").write_text(
        json.dumps({"entry": {"question_id": "x"}}), encoding="utf-8"
    )
    (question_dir / "raw_traces.jsonl").write_text(
        json.dumps(
            {
                "chain_id": 0,
                "completion_text": "hi",
                "terminated_reason": "eos",
                "predicted": "4",
                "correct": True,
                "tokens": [{"token_index": 0, "token_id": 1, "text": "a", "logprob": -0.1}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    from libs.generation import vector_codec

    (question_dir / "hidden_states.jsonl").write_text(
        json.dumps(
            {
                "chain_id": 0,
                "token_index": 0,
                "layer": -1,
                "dim": 2,
                "dtype": "float32",
                "data": vector_codec.encode([1.0, 2.0]),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _meta, chains = TraceLoader().load_question(question_dir)
    assert chains[0].tokens[0].hidden == [1.0, 2.0]
    assert chains[0].correct is True
