from __future__ import annotations

import glob
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from trainee.models import MetricSpec, ProjectSpec

LOSS_PATTERNS = {
    "total_loss": re.compile(r"(?i)total[_ ]?loss\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "loss": re.compile(r"(?i)\bloss\b\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
}
WANDB_URL_RE = re.compile(r"https://wandb\.ai/\S+")


def extract_wandb_url(text: str) -> Optional[str]:
    match = WANDB_URL_RE.search(text)
    return match.group(0) if match else None


def parse_metrics_from_logs(log_text: str, spec: ProjectSpec, wandb_summary: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    for name, pattern in LOSS_PATTERNS.items():
        value = _extract_last_match(pattern, log_text)
        if value is not None:
            metrics[name] = value

    for metric in spec.metric_specs:
        value = _resolve_metric_from_text(metric, log_text, wandb_summary or {})
        if value is not None:
            metrics[metric.name] = value

    return metrics


def parse_metrics_from_sources(
    internal_log_path: str,
    spec: ProjectSpec,
    wandb_summary: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], List[str]]:
    internal_log_text = _read_text(Path(internal_log_path))
    metrics: Dict[str, float] = {}
    source_paths: List[str] = []

    legacy_log_paths = _resolve_existing_paths(spec.legacy_log_paths_for_metrics(), spec)
    legacy_log_text = "\n".join(_read_text(path) for path in legacy_log_paths)
    if legacy_log_paths:
        source_paths.extend(str(path) for path in legacy_log_paths)

    default_log_text = "\n".join(text for text in (internal_log_text, legacy_log_text) if text)
    for name, pattern in LOSS_PATTERNS.items():
        value = _extract_last_match(pattern, default_log_text)
        if value is not None:
            metrics[name] = value

    for metric in spec.metric_specs:
        value, metric_paths = _resolve_metric_from_sources(metric, internal_log_text, spec, wandb_summary or {})
        source_paths.extend(str(path) for path in metric_paths)
        if value is not None:
            metrics[metric.name] = value

    return metrics, _dedupe(source_paths)


def missing_required_metrics(spec: ProjectSpec, metrics: Dict[str, Any]) -> list[str]:
    missing = [item.name for item in spec.metric_specs if item.required and item.name not in metrics]
    if spec.metric_specs:
        return missing
    return [] if any(key in metrics for key in {"loss", "total_loss"}) else ["loss"]


def discover_wandb_summary(spec: ProjectSpec, round_started_at: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if not spec.wandb_enabled:
        return None, None

    project_root = Path(spec.project_root).expanduser().resolve()
    candidates = []
    for path in _summary_search_roots(spec, project_root):
        if not path.exists():
            continue
        if path.is_file() and path.name == "wandb-summary.json":
            candidates.append(path)
            continue
        for summary in path.rglob("wandb-summary.json"):
            candidates.append(summary)

    if not candidates:
        return None, None

    started = datetime.fromisoformat(round_started_at)
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if datetime.fromtimestamp(candidate.stat().st_mtime, tz=started.tzinfo) < started:
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        return str(candidate), payload
    return None, None


def _resolve_metric_from_sources(
    metric: MetricSpec,
    internal_log_text: str,
    spec: ProjectSpec,
    wandb_summary: Dict[str, Any],
) -> Tuple[Optional[float], List[Path]]:
    if metric.source == "stdout_regex":
        return _resolve_metric_from_text(metric, internal_log_text, wandb_summary), []
    if metric.source == "log_file_regex":
        paths = _resolve_existing_paths(_metric_configured_paths(metric), spec)
        log_text = "\n".join(_read_text(path) for path in paths)
        return _resolve_metric_from_text(metric, log_text, wandb_summary), paths
    if metric.source == "jsonl":
        paths = _resolve_existing_paths(_metric_configured_paths(metric), spec)
        return _extract_jsonl_metric(paths, metric.key_or_pattern or metric.name), paths
    if metric.source == "log_regex":
        paths = _resolve_existing_paths(spec.legacy_log_paths_for_metrics(), spec)
        log_text = "\n".join(text for text in [internal_log_text, *[_read_text(path) for path in paths]] if text)
        return _resolve_metric_from_text(metric, log_text, wandb_summary), paths
    return _resolve_metric_from_text(metric, internal_log_text, wandb_summary), []


def _resolve_metric_from_text(metric: MetricSpec, log_text: str, wandb_summary: Dict[str, Any]) -> Optional[float]:
    if metric.source == "wandb_summary":
        value = wandb_summary.get(metric.key_or_pattern)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    try:
        pattern = re.compile(metric.key_or_pattern)
    except re.error:
        return None
    return _extract_last_match(pattern, log_text)


def _extract_last_match(pattern: re.Pattern[str], text: str) -> Optional[float]:
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    if "value" in match.groupdict():
        candidate = match.group("value")
    else:
        candidate = match.group(1)
    try:
        return float(candidate)
    except (TypeError, ValueError):
        return None


def _extract_jsonl_metric(paths: Iterable[Path], key: str) -> Optional[float]:
    for path in sorted(paths, key=lambda item: item.stat().st_mtime if item.exists() else 0, reverse=True):
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = _lookup_key(payload, key)
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _lookup_key(payload: Any, key: str) -> Any:
    current = payload
    for part in key.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _metric_configured_paths(metric: MetricSpec) -> List[str]:
    paths = list(metric.paths)
    if metric.path:
        paths.append(metric.path)
    return paths


def _resolve_existing_paths(raw_paths: Iterable[str], spec: ProjectSpec) -> List[Path]:
    working_dir = Path(spec.working_dir).expanduser().resolve()
    paths: List[Path] = []
    for raw_path in raw_paths:
        base_candidate = Path(raw_path).expanduser()
        if not base_candidate.is_absolute():
            base_candidate = (working_dir / raw_path).resolve()
        matches = glob.glob(str(base_candidate), recursive=True)
        if matches:
            paths.extend(Path(item).resolve() for item in matches if Path(item).is_file())
        elif base_candidate.is_file():
            paths.append(base_candidate.resolve())
    return _dedupe_paths(paths)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _dedupe(values: Iterable[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    deduped: List[Path] = []
    seen = set()
    for path in paths:
        marker = str(path)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(path)
    return deduped


def _summary_search_roots(spec: ProjectSpec, project_root: Path) -> list[Path]:
    roots = [project_root, Path(spec.working_dir).expanduser().resolve()]
    for raw_path in spec.log_paths:
        expanded = (Path(spec.working_dir).expanduser().resolve() / raw_path).expanduser()
        roots.append(expanded if expanded.exists() else project_root / raw_path)
    deduped = []
    seen = set()
    for path in roots:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if str(resolved) in seen:
            continue
        seen.add(str(resolved))
        deduped.append(resolved)
    return deduped
