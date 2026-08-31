"""Reasoning-graph library object: consolidate raw token traces into a DAG.

Reads the on-disk trace artifacts (produced by the generation stage) and, under a
pluggable latent-merge heuristic and threshold, merges token states from different
chains that represent sufficiently equivalent reasoning states — producing branch
and join structure while preserving the DAG invariant (research plan sections 7-13).
The heuristic and threshold are configuration; generation is never re-run to try a
new merge policy.
"""
from .candidate_filter import CandidateFilter
from .depth_policy import DepthPolicy
from .graph_consolidator import GraphConsolidator
from .graph_node import GraphNode
from .graph_statistics import GraphStatistics
from .hidden_cosine_metric import HiddenCosineMetric
from .loaded_chain import LoadedChain
from .loaded_token import LoadedToken
from .merge_heuristic import MergeHeuristic
from .merge_metric import MergeMetric
from .merge_registry import MergeRegistry
from .pooled_hidden_cosine_metric import PooledHiddenCosineMetric
from .reasoning_graph import ReasoningGraph
from .representative_policy import RepresentativePolicy
from .representative_selector import RepresentativeSelector
from .representative_selector_registry import RepresentativeSelectorRegistry
from .token_identity_metric import TokenIdentityMetric
from .trace_loader import TraceLoader

__all__ = [
    "CandidateFilter",
    "DepthPolicy",
    "GraphConsolidator",
    "GraphNode",
    "GraphStatistics",
    "HiddenCosineMetric",
    "LoadedChain",
    "LoadedToken",
    "MergeHeuristic",
    "MergeMetric",
    "MergeRegistry",
    "PooledHiddenCosineMetric",
    "ReasoningGraph",
    "RepresentativePolicy",
    "RepresentativeSelector",
    "RepresentativeSelectorRegistry",
    "TokenIdentityMetric",
    "TraceLoader",
]
