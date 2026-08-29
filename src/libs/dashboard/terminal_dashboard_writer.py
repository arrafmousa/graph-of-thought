"""Terminal writer: print the dashboard as text whenever a component updates.

Attaches to a dashboard as an update listener and renders its tiles (ASCII
tables and sparkline graphs) to a text stream. On an interactive terminal it
redraws in place; on a non-TTY stream (e.g. a Colab cell or a file) it appends
each snapshot. Useful for headless environments with no browser.
"""
from __future__ import annotations

from typing import TextIO

from .base_dashboard import BaseDashboard


class TerminalDashboardWriter:
    """Render a dashboard's text form to a stream after every update."""

    def __init__(self, dashboard: BaseDashboard, stream: TextIO) -> None:
        self._dashboard = dashboard
        self._stream = stream
        # Prefer UTF-8 so sparkline block glyphs render on consoles that
        # otherwise default to a narrow encoding (e.g. cp1252 on Windows).
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass
        dashboard.add_update_listener(self.write)

    def write(self) -> None:
        text = self._dashboard.render_text()
        is_tty = getattr(self._stream, "isatty", lambda: False)()
        prefix = "\x1b[2J\x1b[H" if is_tty else ""  # clear screen + cursor home on a TTY
        payload = prefix + text + "\n"
        try:
            self._stream.write(payload)
        except UnicodeEncodeError:
            # Never let a console encoding limitation break the run.
            encoding = getattr(self._stream, "encoding", None) or "ascii"
            self._stream.write(payload.encode(encoding, errors="replace").decode(encoding))
        self._stream.flush()
