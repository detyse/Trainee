from __future__ import annotations

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from trainee.models import MetricSpec, ProjectSpec, RoundRecord


class ResearchRoundState(BaseModel):
    round_id: Optional[int] = None
    session_id: Optional[int] = None
    round_index: int
    status: str
    param_values: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    primary_metric_value: Optional[float] = None
    hypothesis: str = ""
    change_summary: str = ""
    param_diff: dict[str, dict[str, Any]] = Field(default_factory=dict)
    change_signature: str = ""
    latest_round_judgement: str = "inconclusive"
    error: Optional[str] = None


class ResearchState(BaseModel):
    primary_metric_name: str = ""
    primary_metric_goal: str = "min"
    baseline_round: Optional[ResearchRoundState] = None
    best_so_far_round: Optional[ResearchRoundState] = None
    latest_round: Optional[ResearchRoundState] = None
    recent_rounds: list[ResearchRoundState] = Field(default_factory=list)
    tried_change_signatures: list[str] = Field(default_factory=list)
    rejected_change_signatures: list[str] = Field(default_factory=list)


class ResearchStateBuilder:
    recent_round_limit = 5

    def build(self, spec: ProjectSpec, rounds: list[RoundRecord]) -> ResearchState:
        metric = self.primary_metric(spec, rounds)
        states: list[ResearchRoundState] = []
        tried: list[str] = []
        rejected: list[str] = []

        for index, record in enumerate(rounds):
            previous = rounds[index - 1] if index > 0 else None
            param_diff = self.param_diff(previous.param_values, record.param_values) if previous else {}
            signature = self.change_signature(param_diff)
            hypothesis, change_summary = self._incoming_experiment(spec, previous, record, index)
            value = self.metric_value(record, metric)
            judgement = (
                record.agent_decision.latest_round_judgement
                if record.agent_decision is not None
                else "inconclusive"
            )
            state = ResearchRoundState(
                round_id=record.id,
                session_id=record.session_id,
                round_index=record.round_index,
                status=record.status,
                param_values=record.param_values,
                metrics=record.metrics,
                primary_metric_value=value,
                hypothesis=hypothesis,
                change_summary=change_summary,
                param_diff=param_diff,
                change_signature=signature,
                latest_round_judgement=judgement,
                error=record.error,
            )
            states.append(state)
            if signature:
                self._append_unique(tried, signature)
                if record.status != "completed" or self._is_rejected_judgement(judgement):
                    self._append_unique(rejected, signature)

        best = self._best_round(states, metric)
        return ResearchState(
            primary_metric_name=metric.name if metric else "",
            primary_metric_goal=metric.goal if metric else "min",
            baseline_round=states[0] if states else None,
            best_so_far_round=best,
            latest_round=states[-1] if states else None,
            recent_rounds=states[-self.recent_round_limit :],
            tried_change_signatures=tried,
            rejected_change_signatures=rejected,
        )

    def primary_metric(self, spec: ProjectSpec, rounds: list[RoundRecord]) -> Optional[MetricSpec]:
        if spec.metric_specs:
            for metric in spec.metric_specs:
                if any(self.metric_value(item, metric) is not None for item in rounds):
                    return metric
            return spec.metric_specs[0]
        for name in ("total_loss", "loss"):
            if any(self._numeric(item.metrics.get(name)) is not None for item in rounds):
                return MetricSpec(
                    name=name,
                    key_or_pattern=r"(?P<value>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
                    goal="min",
                    required=False,
                )
        names = sorted({name for item in rounds for name in item.metrics})
        return (
            MetricSpec(
                name=names[0],
                key_or_pattern=r"(?P<value>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
                goal="min",
                required=False,
            )
            if names
            else None
        )

    def metric_value(self, record: RoundRecord, metric: Optional[MetricSpec]) -> Optional[float]:
        if metric is None:
            return None
        return self._numeric(record.metrics.get(metric.name))

    def param_diff(
        self,
        previous: dict[str, Any],
        current: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        diff: dict[str, dict[str, Any]] = {}
        for name in sorted(set(previous) | set(current)):
            before = previous.get(name)
            after = current.get(name)
            if before != after:
                diff[name] = {"from": before, "to": after}
        return diff

    def change_signature(self, diff: dict[str, dict[str, Any]]) -> str:
        if not diff:
            return ""
        return json.dumps(diff, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def is_better(self, candidate: float, reference: float, goal: str) -> bool:
        return candidate < reference if goal == "min" else candidate > reference

    def _incoming_experiment(
        self,
        spec: ProjectSpec,
        previous: Optional[RoundRecord],
        current: RoundRecord,
        index: int,
    ) -> tuple[str, str]:
        if index == 0:
            return "Establish the baseline with the original parameter configuration.", "Run the baseline configuration."
        if previous is None or previous.agent_decision is None:
            return "", ""
        try:
            expected = spec.merge_param_values(previous.agent_decision.next_params, base=previous.param_values)
            actual = spec.merge_param_values(current.param_values)
        except ValueError:
            return "", ""
        if expected != actual:
            return "", ""
        return previous.agent_decision.hypothesis, previous.agent_decision.change_summary

    def _best_round(
        self,
        states: list[ResearchRoundState],
        metric: Optional[MetricSpec],
    ) -> Optional[ResearchRoundState]:
        if metric is None:
            return None
        best: Optional[ResearchRoundState] = None
        for state in states:
            if state.status != "completed" or state.primary_metric_value is None:
                continue
            if best is None or self.is_better(
                state.primary_metric_value,
                best.primary_metric_value,  # type: ignore[arg-type]
                metric.goal,
            ):
                best = state
        return best

    def _is_rejected_judgement(self, value: str) -> bool:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        return normalized in {"worse", "rejected"}

    def _numeric(self, value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _append_unique(self, values: list[str], value: str) -> None:
        if value not in values:
            values.append(value)
