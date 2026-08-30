"""A consolidated graph node: a cluster of equivalent token states."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphNode:
    """One node of the consolidated DAG (research plan sections 7, 15).

    A node groups one or more raw token states (``members`` as ``(chain, index)``
    pairs) judged equivalent by the merge heuristic. ``depth`` is the minimum token
    index among members; ``texts`` collects the distinct decoded tokens.
    """

    cluster_id: int
    depth: int
    members: list[tuple[int, int]] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    terminal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "depth": self.depth,
            "size": len(self.members),
            "members": [list(member) for member in self.members],
            "texts": list(self.texts),
            "terminal": self.terminal,
        }
