from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
        value = _resolve_metric(metric, log_text, wandb_summary or {})
        if value is not None:
            metrics[metric.name] = value

    return metrics


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


def _resolve_metric(metric: MetricSpec, log_text: str, wandb_summary: Dict[str, Any]) -> Optional[float]:
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
