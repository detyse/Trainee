from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any


def configure_logging() -> None:
    level_name = os.getenv("TRAINEE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger("trainee")
    root.setLevel(level)
    if root.handlers:
        for handler in root.handlers:
            handler.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"trainee.{name}")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key.startswith("_") and key not in {"_trainee_extra"}:
                continue
        extra = getattr(record, "_trainee_extra", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)
