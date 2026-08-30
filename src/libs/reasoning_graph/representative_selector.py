"""Deterministically choose which chain survives a merge (research plan section 9).

The first POC policy is recent mean sampled-token log-probability over a window;
ties break on the lower chain id so results are reproducible.
"""
from __future__ import annotations

from typing import Any

from .loaded_chain import LoadedChain


class RepresentativeSelector:
    """Select the higher-confidence chain at a merge and record the decision."""

    def __init__(self, *, window: int) -> None:
        self._window = window

    def select(
        self,
        chain_a: LoadedChain,
        index_a: int,
        chain_b: LoadedChain,
        index_b: int,
    ) -> dict[str, Any]:
        conf_a = chain_a.recent_mean_logprob(index_a, self._window)
        conf_b = chain_b.recent_mean_logprob(index_b, self._window)
        if conf_a > conf_b or (conf_a == conf_b and chain_a.chain_id <= chain_b.chain_id):
            winner, loser = chain_a, chain_b
            winner_conf, loser_conf = conf_a, conf_b
        else:
            winner, loser = chain_b, chain_a
            winner_conf, loser_conf = conf_b, conf_a
        return {
            "winner_chain": winner.chain_id,
            "loser_chain": loser.chain_id,
            "winner_recent_mean_logprob": winner_conf,
            "loser_recent_mean_logprob": loser_conf,
            "confidence_window": self._window,
        }
