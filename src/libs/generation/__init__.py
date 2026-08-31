"""Generation library object: sample multiple token-level reasoning traces.

A run selects a model provider by class name in configuration. A provider turns
a prompt into ``num_chains`` independent :class:`ChainTrace` values, where every
generated :class:`TokenNode` carries its hidden representation and sampled-token
log-probability (research plan sections 2, 5). Hidden states are detached to CPU
and written to disk by :class:`TraceStore` so they never accumulate on the GPU.
"""
from .chain_trace import ChainTrace
from .huggingface_model_provider import HuggingFaceModelProvider
from .model_provider import ModelProvider
from .model_provider_kind import ModelProviderKind
from .model_provider_registry import ModelProviderRegistry
from .synthetic_model_provider import SyntheticModelProvider
from .token_node import TokenNode
from .trace_store import TraceStore

__all__ = [
    "ChainTrace",
    "HuggingFaceModelProvider",
    "ModelProvider",
    "ModelProviderKind",
    "ModelProviderRegistry",
    "SyntheticModelProvider",
    "TokenNode",
    "TraceStore",
]
