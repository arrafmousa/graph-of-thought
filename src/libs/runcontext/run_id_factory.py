"""Deterministic, collision-resistant run identifiers (AGENTS.md section 21)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone


class RunIdFactory:
    """Create readable, unique run identifiers combining a UTC stamp and a suffix."""

    def create(self, task_name: str) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = uuid.uuid4().hex[:8]
        safe_task = "".join(c if c.isalnum() or c in "-_" else "-" for c in task_name)
        return f"{stamp}_{safe_task}_{suffix}"
