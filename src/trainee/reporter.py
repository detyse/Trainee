from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from trainee.ledger import LedgerExporter
from trainee.models import ProjectSpec, RoundRecord, RunSession
from trainee.research_state import ResearchRoundState, ResearchStateBuilder
from trainee.storage import Storage


class ReportGenerator:
    def __init__(
        self,
        storage: Storage,
        research_state_builder: Optional[ResearchStateBuilder] = None,
        ledger_exporter: Optional[LedgerExporter] = None,
    ):
        self.storage = storage
        self.research_state_builder = research_state_builder or ResearchStateBuilder()
        self.ledger_exporter = ledger_exporter or LedgerExporter(self.research_state_builder)

    def generate_session_report(self, session_id: int) -> str:
        session = self.storage.get_session(session_id)
        if session is None:
            raise ValueError(f"session {session_id} not found")
        if session.project_spec is None:
            raise ValueError(f"session {session_id} has no project spec")
        rounds = self.storage.list_research_rounds(session_id)
        state = self.research_state_builder.build(session.project_spec, rounds)
        ledger_rows = self.ledger_exporter.build_rows(session.project_spec, rounds, state)

        sections = [
            f"# Trainee Session Report #{session_id}",
            self._session_summary(session, rounds, state.best_so_far_round, state.primary_metric_name),
            self._params_table(rounds),
            self._metrics_table(rounds, ledger_rows, state.primary_metric_name, state.best_so_far_round),
            self._decision_log(rounds),
            self._final_conclusion(session.project_spec, state.primary_metric_name, state.best_so_far_round),
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
        best_round: Optional[ResearchRoundState],
        metric_name: str,
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
        if metric_name and best_round is not None:
            rows.append(
                (
                    "Best Metric",
                    f"{metric_name}={best_round.primary_metric_value} at round {best_round.round_index}",
                )
            )
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
        ledger_rows: list[dict[str, Any]],
        metric_name: str,
        best_round: Optional[ResearchRoundState],
    ) -> str:
        metric_names = sorted({name for item in rounds for name in item.metrics})
        if not rounds or not metric_names:
            return "## Metric Trend\n\nNo metrics were recorded."
        header = ["Round", "Status", *metric_names, "Best"]
        lines = [self._markdown_row(header), self._markdown_separator(len(header))]
        best_id = best_round.round_id if best_round else None
        for item, ledger_row in zip(rounds, ledger_rows):
            marker = f"best {metric_name}" if metric_name and item.id == best_id else ""
            lines.append(
                self._markdown_row(
                    [
                        item.round_index,
                        item.status,
                        *[item.metrics.get(name, "") for name in metric_names],
                        marker or ledger_row["best_so_far"],
                    ]
                )
            )
        return "## Metric Trend\n\n" + "\n".join(lines)

    def _decision_log(self, rounds: list[RoundRecord]) -> str:
        rows = [item for item in rounds if item.agent_decision is not None]
        if not rows:
            return "## Decision Log\n\nNo agent decisions were recorded."
        lines = [
            self._markdown_row(["Round", "Action", "Hypothesis", "Change", "Reason"]),
            self._markdown_separator(5),
        ]
        for item in rows:
            decision = item.agent_decision
            assert decision is not None
            lines.append(
                self._markdown_row(
                    [
                        item.round_index,
                        decision.action,
                        decision.hypothesis,
                        decision.change_summary,
                        decision.reason,
                    ]
                )
            )
        return "## Decision Log\n\n" + "\n".join(lines)

    def _final_conclusion(
        self,
        spec: ProjectSpec,
        metric_name: str,
        best_round: Optional[ResearchRoundState],
    ) -> str:
        if not metric_name or best_round is None:
            return "## Final Conclusion\n\nNo best round could be selected because no primary metric was available."
        params = ", ".join(f"{name}={value}" for name, value in sorted(best_round.param_values.items()))
        return (
            "## Final Conclusion\n\n"
            f"Best observed round: **{best_round.round_index}** with "
            f"**{metric_name}={best_round.primary_metric_value}**. "
            f"Recommended parameter set: `{params}`."
        )

    def _markdown_row(self, values: list[Any]) -> str:
        return "| " + " | ".join(self._cell(value) for value in values) + " |"

    def _markdown_separator(self, count: int) -> str:
        return "| " + " | ".join("---" for _ in range(count)) + " |"

    def _cell(self, value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
