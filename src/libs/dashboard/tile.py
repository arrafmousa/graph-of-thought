"""Tile contract: a named dashboard component (AGENTS.md sections 5, 13).

A tile is created with its own settings and exposes a typed ``update`` method
that defines exactly how it accepts a new piece of information (a table appends
a row; a graph appends a point). ``render`` returns its HTML. New tile types are
added by implementing this contract in a new file.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class Tile(ABC):
    """A dashboard component that accepts typed updates and renders HTML."""

    _listener: Optional[Callable[[], None]] = None

    def set_listener(self, listener: Callable[[], None]) -> None:
        """Register a callback invoked after each update (used for live writes)."""
        self._listener = listener

    def _notify(self) -> None:
        if self._listener is not None:
            self._listener()

    @abstractmethod
    def update(self, *args: Any) -> None:
        """Accept a new piece of information; signature is tile-specific."""

    @abstractmethod
    def render(self) -> str:
        """Return an HTML fragment representing the tile's current state."""

    @abstractmethod
    def render_text(self) -> str:
        """Return a plain-text representation for terminal output."""
