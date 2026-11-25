"""
Audit logging module

Records raw JSON input/output payloads passing through the Gemini CLI (agy)
to a separate JSONL file for inspection and debugging.
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from .config import config

logger = logging.getLogger("gemini_cli_proxy.audit")


class AuditLogger:
    """Thread-safe JSONL audit logger for raw agy input/output."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def record(
        self,
        direction: str,
        data: Any,
        *,
        mode: str = "acp",
        pid: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Append a single audit record as a JSON line if auditing is enabled."""
        if not config.audit_enabled:
            return

        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": direction,
            "mode": mode,
        }
        if pid is not None:
            entry["pid"] = pid
        if session_id is not None:
            entry["session_id"] = session_id
        entry["data"] = data

        audit_file = config.audit_file
        try:
            line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
            parent = os.path.dirname(audit_file)
            with self._lock:
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(audit_file, "a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush()
        except Exception as exc:
            logger.warning(f"Failed to write audit log to '{audit_file}': {exc}")


audit_logger = AuditLogger()


def record_audit(
    direction: str,
    data: Any,
    *,
    mode: str = "acp",
    pid: Optional[int] = None,
    session_id: Optional[str] = None,
) -> None:
    """Record an audit entry using the global AuditLogger singleton."""
    audit_logger.record(
        direction,
        data,
        mode=mode,
        pid=pid,
        session_id=session_id,
    )
