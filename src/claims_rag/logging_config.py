from __future__ import annotations

import json
import logging
import os


class JsonFormatter(logging.Formatter):
    """One JSON event per line, for log collectors."""

    def format(self, record: logging.LogRecord) -> str:
        event = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            event["exception"] = self.formatException(record.exc_info)
        return json.dumps(event)


def configure_logging() -> None:
    """With LOG_FORMAT=json, switch the root and uvicorn handlers to JSON. Logs always go to stdout/stderr."""
    if os.environ.get("LOG_FORMAT", "").lower() != "json":
        return
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(logging.StreamHandler())
    for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access"):
        for handler in logging.getLogger(name).handlers:
            handler.setFormatter(JsonFormatter())
