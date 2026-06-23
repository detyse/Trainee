from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from trainee.models import MetricSpec, RoundRecord, RunSession
from trainee.storage import Storage


class ReportGenerator:
    def __init__(self, storage: Storage):
        self.storage = storage

    def generate_session_report(self, session_id: int) -> str:
        session = self.storage.get_session(session_id)
        if session is None:
            raise ValueError(f"session {session_id} not found")
        rounds = list(reversed(self.storage.list_rounds(session_id)))
        metric = self._primary_metric(session, rounds)
        best_round = self._best_round(rounds, metric)

        sections = [
            f"# Trainee Session Report #{session_id}",
            self._session_summary(session, rounds, metric, best_round),
            self._params_table(rounds),
            self._metrics_table(rounds, metric, best_round),
            self._decision_log(rounds),
            self._final_conclusion(metric, best_round),
        ]
        return "\n\n".join(section for section in sections if section).rstrip() + "\n"

    def save_report(self, session_id: int, output_dir: Path) -> Path:
        session_dir = output_dir / f"session-{session_id:04d}"
        session_dir.mkdir(parents=True, exist_ok=True)
        report_path = session_dir / "report.md"
        report_path.write_text(self.generate_session_report(session_id), encoding="utf-8")
        return report_path

    def _session_summary(
        self,
        session: RunSession,
        rounds: list[RoundRecord],
        metric: Optional[MetricSpec],
        best_round: Optional[RoundRecord],
    ) -> str:
        spec = session.project_spec
        rows = [
            ("Status", session.status),
            ("Started At", session.started_at),
            ("Ended At", session.ended_at or ""),
            ("Stop Reason", session.stop_reason or ""),
            ("Total Rounds", str(len(rounds))),
            ("Project Root", spec.project_root if spec else ""),
            ("Resumed From", str(session.resumed_from or "")),
        ]
        if metric and best_round:
            rows.append(("Best Metric", f"{metric.name}={best_round.metrics.get(metric.name)} at round {best_round.round_index}"))
        return "## Summary\n\n" + "\n".join(f"- **{name}**: {value}" for name, value in rows)

    def _params_table(self, rounds: list[RoundRecord]) -> str:
        param_names = sorted({name for item in rounds for name in item.param_values})
        if not rounds or not param_names:
            return "## Parameter Trajectory\n\nNo parameter values were recorded."
        header = ["Round", "Status", *param_names]
        lines = [self._markdown_row(header), self._markdown_separator(len(header))]
        for item in rounds:
            lines.append(
                self._markdown_row(
                    [
                        item.round_index,
                        item.status,
                        *[item.param_values.get(name, "") for name in param_names],
                    ]
                )
            )
        return "## Parameter Trajectory\n\n" + "\n".join(lines)

    def _metrics_table(
        self,
        rounds: list[RoundRecord],
        primary_metric: Optional[MetricSpec],
        best_round: Optional[RoundRecord],
    ) -> str:
        metric_names = sorted({name for item in rounds for name in item.metrics})
        if not rounds or not metric_names:
            return "## Metric Trend\n\nNo metrics were recorded."
        header = ["Round", "Status", *metric_names, "Best"]
        lines = [self._markdown_row(header), self._markdown_separator(len(header))]
        for item in rounds:
            marker = ""
            if primary_metric and best_round and item.id == best_round.id:
                marker = f"best {primary_metric.name}"
            lines.append(
                self._markdown_row(
                    [
                        item.round_index,
                        item.status,
                        *[item.metrics.get(name, "") for name in metric_names],
                        marker,
                    ]
                )
            )
        return "## Metric Trend\n\n" + "\n".join(lines)

    def _decision_log(self, rounds: list[RoundRecord]) -> str:
        rows = [item for item in rounds if item.agent_decision is not None]
        if not rows:
            return "## Decision Log\n\nNo agent decisions were recorded."
        lines = [self._markdown_row(["Round", "Action", "Reason"]), self._markdown_separator(3)]
        for item in rows:
            decision = item.agent_decision
            assert decision is not None
            lines.append(self._markdown_row([item.round_index, decision.action, decision.reason]))
        return "## Decision Log\n\n" + "\n".join(lines)

    def _final_conclusion(self, metric: Optional[MetricSpec], best_round: Optional[RoundRecord]) -> str:
        if metric is None or best_round is None:
            return "## Final Conclusion\n\nNo best round could be selected because no primary metric was available."
        params = ", ".join(f"{name}={value}" for name, value in sorted(best_round.param_values.items()))
        value = best_round.metrics.get(metric.name)
        return (
            "## Final Conclusion\n\n"
            f"Best observed round: **{best_round.round_index}** with **{metric.name}={value}**. "
            f"Recommended parameter set: `{params}`."
        )

    def _primary_metric(self, session: RunSession, rounds: list[RoundRecord]) -> Optional[MetricSpec]:
        if session.project_spec and session.project_spec.metric_specs:
            for metric in session.project_spec.metric_specs:
                if any(metric.name in item.metrics for item in rounds):
                    return metric
            return session.project_spec.metric_specs[0]
        names = sorted({name for item in rounds for name in item.metrics})
        if not names:
            return None
        return MetricSpec(name=names[0], key_or_pattern=names[0], goal="min", required=False)

    def _best_round(self, rounds: list[RoundRecord], metric: Optional[MetricSpec]) -> Optional[RoundRecord]:
        if metric is None:
            return None
        candidates = [item for item in rounds if metric.name in item.metrics]
        if not candidates:
            return None
        reverse = metric.goal == "max"
        return sorted(candidates, key=lambda item: float(item.metrics[metric.name]), reverse=reverse)[0]

    def _markdown_row(self, values: list[Any]) -> str:
        return "| " + " | ".join(self._cell(value) for value in values) + " |"

    def _markdown_separator(self, count: int) -> str:
        return "| " + " | ".join("---" for _ in range(count)) + " |"

    def _cell(self, value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
