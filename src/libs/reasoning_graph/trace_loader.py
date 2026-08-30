"""Load raw trace artifacts back into graph-library input types.

Reads the documented on-disk format written by the generation stage's
``TraceStore`` (see that module for the layout). This is the file-based contract
between the two isolated libraries — no cross-library imports.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import vector_math
from .loaded_chain import LoadedChain
from .loaded_token import LoadedToken


class TraceLoader:
    """Reconstruct ``LoadedChain`` values from a question's trace directory."""

    def load_question(self, question_dir: Path) -> tuple[dict[str, Any], list[LoadedChain]]:
        question_dir = Path(question_dir)
        question_meta = json.loads(
            (question_dir / "question.json").read_text(encoding="utf-8")
        )
        hidden = self._load_hidden(question_dir / "hidden_states.jsonl")
        chains = self._load_chains(question_dir / "raw_traces.jsonl", hidden)
        return question_meta, chains

    def _load_hidden(self, path: Path) -> dict[tuple[int, int], list[float]]:
        vectors: dict[tuple[int, int], list[float]] = {}
        if not path.is_file():
            return vectors
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = (int(record["chain_id"]), int(record["token_index"]))
            vectors[key] = vector_math.decode(record["data"])
        return vectors

    def _load_chains(
        self, path: Path, hidden: dict[tuple[int, int], list[float]]
    ) -> list[LoadedChain]:
        chains: list[LoadedChain] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            chain_id = int(record["chain_id"])
            chain = LoadedChain(
                chain_id=chain_id,
                completion_text=record.get("completion_text", ""),
                terminated_reason=record.get("terminated_reason", "max_tokens"),
                predicted=record.get("predicted"),
                correct=record.get("correct"),
            )
            for token in record.get("tokens", []):
                index = int(token["token_index"])
                chain.tokens.append(
                    LoadedToken(
                        chain_id=chain_id,
                        token_index=index,
                        token_id=int(token["token_id"]),
                        text=token["text"],
                        logprob=float(token["logprob"]),
                        hidden=hidden.get((chain_id, index)),
                    )
                )
            chains.append(chain)
        return chains
