"""Synthetic local model provider: deterministic token traces, no torch.

Produces reproducible chains whose hidden vectors cluster by concept so the
downstream latent-merge heuristics have real branch/join structure to consolidate.
Interchangeable with :class:`HuggingFaceModelProvider` purely through configuration;
used for local CPU development and tests (AGENTS.md local-cpu-dev skill).
"""
from __future__ import annotations

import random
import re

from .chain_trace import ChainTrace
from .model_provider import ModelProvider
from .token_node import TokenNode

_DIM = 16
_VOCAB = [
    "Let", "us", "count", "the", "apples", "carefully", "first", "we",
    "add", "them", "together", "step", "by", "so", "the", "total",
]
_SHARED_PREFIX_LEN = 4
_NUMBER = re.compile(r"-?\d+")


def _base_vector(concept: int) -> list[float]:
    rng = random.Random(1000 + concept)
    vector = [0.0] * _DIM
    for _ in range(3):
        vector[rng.randrange(_DIM)] = rng.uniform(0.8, 1.2)
    return vector


class SyntheticModelProvider(ModelProvider):
    """Generate clustering token traces without any machine-learning dependency."""

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        dtype: str,
        device: str,
        hidden_layer: int,
    ) -> None:
        self._model_id = model_id
        self._model_revision = model_revision
        self._dtype = dtype
        self._device = device
        self._hidden_layer = hidden_layer
        self._base_vectors = [_base_vector(i) for i in range(len(_VOCAB))]

    def provides_hidden_states(self) -> bool:
        return True

    def release(self) -> None:
        return None

    def generate(
        self,
        *,
        prompt: str,
        num_chains: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        batch_size: int,
        seed: int,
    ) -> list[ChainTrace]:
        gold = self._gold_from_prompt(prompt)
        shared = self._shared_prefix(seed, max_new_tokens)
        traces: list[ChainTrace] = []
        for chain_id in range(num_chains):
            traces.append(
                self._sample_chain(chain_id, seed, gold, shared, max_new_tokens, temperature)
            )
        return traces

    def _sample_chain(self, chain_id, seed, gold, shared, max_new_tokens, temperature):
        rng = random.Random(seed * 1000 + chain_id)
        correct = rng.random() < 0.6
        answer = gold if correct else gold + rng.choice([-3, -2, -1, 1, 2, 3])
        trace = ChainTrace(chain_id=chain_id, terminated_reason="eos")
        index = 0
        body_len = max(1, max_new_tokens - 3)
        for concept in shared:
            if index >= body_len:
                break
            trace.tokens.append(self._token(index, concept, rng, confident=correct, temperature=temperature))
            index += 1
        while index < body_len:
            concept = rng.randrange(len(_VOCAB))
            trace.tokens.append(self._token(index, concept, rng, confident=correct, temperature=temperature))
            index += 1
            if rng.random() < 0.25:
                break
        trace.tokens.append(self._answer_token(index, "####", 900, rng, correct))
        trace.tokens.append(self._answer_token(index + 1, str(answer), 901, rng, correct))
        trace.tokens.append(self._answer_token(index + 2, "####", 902, rng, correct))
        trace.completion_text = self._detokenize(trace.tokens)
        return trace

    def _token(self, index, concept, rng, *, confident, temperature):
        scale = 0.02 * max(temperature, 0.1)
        vector = [v + rng.uniform(-scale, scale) for v in self._base_vectors[concept]]
        logprob = -rng.uniform(0.05, 0.5) if confident else -rng.uniform(0.5, 2.0)
        return TokenNode(
            token_index=index,
            token_id=concept,
            text=_VOCAB[concept],
            logprob=logprob,
            hidden=vector,
        )

    def _answer_token(self, index, text, token_id, rng, correct):
        base = _base_vector(token_id % len(_VOCAB))
        vector = [v + rng.uniform(-0.02, 0.02) for v in base]
        logprob = -rng.uniform(0.02, 0.2) if correct else -rng.uniform(0.3, 1.0)
        return TokenNode(
            token_index=index, token_id=token_id, text=text, logprob=logprob, hidden=vector
        )

    def _shared_prefix(self, seed, max_new_tokens):
        rng = random.Random(seed)
        length = min(_SHARED_PREFIX_LEN, max(1, max_new_tokens - 2))
        return [rng.randrange(len(_VOCAB)) for _ in range(length)]

    def _gold_from_prompt(self, prompt: str) -> int:
        numbers = [int(m.group(0)) for m in _NUMBER.finditer(prompt)]
        if len(numbers) >= 2:
            return numbers[0] + numbers[1]
        if numbers:
            return numbers[0]
        return 0

    @staticmethod
    def _detokenize(tokens) -> str:
        return " ".join(token.text for token in tokens).strip()
