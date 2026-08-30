"""Persist raw traces as the documented hand-off between pipeline stages.

Graph generation is only the first stage of the larger pipeline (research plan
sections 6, 18). This store writes a stable, inspectable on-disk format under
``output/<run_id>/artifacts/traces/`` so consolidation, SBC labeling, and the
detector can be re-run later without regenerating expensive LLM traces:

    traces/
      index.json                    # questions + generation settings summary
      question_<qid>/
        question.json               # entry, prompt, generation params, hidden layer
        raw_traces.jsonl            # one JSON line per chain (tokens without hidden)
        hidden_states.jsonl         # one JSON line per token: base64 float32 vector
        raw_stats.json              # per-question generation statistics

Hidden vectors are stored separately (base64 float32) so the lightweight trace
metadata stays human-readable and the latent tensors can be streamed on demand.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import vector_codec
from .chain_trace import ChainTrace

TRACE_SCHEMA_VERSION = "1.0.0"


class TraceStore:
    """Write per-question trace artifacts under a traces root directory."""

    def __init__(self, traces_root: Path) -> None:
        self._root = Path(traces_root)
        self._root.mkdir(parents=True, exist_ok=True)

    def question_dir(self, question_id: str) -> Path:
        safe = question_id.replace("/", "-")
        return self._root / f"question_{safe}"

    def write_question(
        self,
        *,
        question_id: str,
        entry: dict[str, Any],
        prompt: str,
        generation_params: dict[str, Any],
        hidden_layer: int,
        traces: list[ChainTrace],
        stats: dict[str, Any],
    ) -> Path:
        directory = self.question_dir(question_id)
        directory.mkdir(parents=True, exist_ok=True)

        (directory / "question.json").write_text(
            json.dumps(
                {
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "entry": entry,
                    "prompt": prompt,
                    "generation": generation_params,
                    "hidden_layer": hidden_layer,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        with (directory / "raw_traces.jsonl").open("w", encoding="utf-8") as handle:
            for trace in traces:
                handle.write(json.dumps(trace.metadata()) + "\n")

        with (directory / "hidden_states.jsonl").open("w", encoding="utf-8") as handle:
            for trace in traces:
                for token in trace.tokens:
                    if token.hidden is None:
                        continue
                    record = {
                        "chain_id": trace.chain_id,
                        "token_index": token.token_index,
                        "layer": hidden_layer,
                        "dim": len(token.hidden),
                        "dtype": "float32",
                        "data": vector_codec.encode(token.hidden),
                    }
                    handle.write(json.dumps(record) + "\n")

        (directory / "raw_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        return directory

    def write_index(self, index: dict[str, Any]) -> Path:
        path = self._root / "index.json"
        path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        return path
