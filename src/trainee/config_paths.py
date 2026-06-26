from __future__ import annotations

from typing import Any


def get_config_value(payload: Any, raw_path: str) -> Any:
    current = payload
    for part in _path_parts(raw_path):
        current = _get_child(current, part, raw_path)
    return current


def set_config_value(payload: Any, raw_path: str, value: Any) -> None:
    parts = _path_parts(raw_path)
    current = payload
    for part in parts[:-1]:
        current = _get_child(current, part, raw_path)

    last = parts[-1]
    if isinstance(current, dict):
        if last not in current:
            raise ValueError(f"config path does not exist: {raw_path}")
        current[last] = value
        return
    if isinstance(current, list) and last.isdigit():
        index = int(last)
        if index >= len(current):
            raise ValueError(f"config path does not exist: {raw_path}")
        current[index] = value
        return
    raise ValueError(f"cannot set config path through non-container segment: {raw_path}")


def _path_parts(raw_path: str) -> list[str]:
    parts = raw_path.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"config_path must use non-empty dot-separated segments: {raw_path}")
    return parts


def _get_child(current: Any, part: str, raw_path: str) -> Any:
    if isinstance(current, dict):
        if part not in current:
            raise ValueError(f"config path does not exist: {raw_path}")
        return current[part]
    if isinstance(current, list) and part.isdigit():
        index = int(part)
        if index >= len(current):
            raise ValueError(f"config path does not exist: {raw_path}")
        return current[index]
    raise ValueError(f"cannot traverse config path through non-container segment: {raw_path}")
