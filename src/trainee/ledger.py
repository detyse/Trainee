from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Optional

from trainee.models import ProjectSpec, RoundRecord
from trainee.research_state import ResearchState, ResearchStateBuilder


LEDGER_FIELDS = [
    "round",
    "status",
    "primary_metric",
    "delta_vs_baseline",
    "delta_vs_best",
    "best_so_far",
    "hypothesis",
    "change_summary",
    "param_diff",
    "error",
    "decision_reason",
]


class LedgerExporter:
    def __init__(self, research_state_builder: Optional[ResearchStateBuilder] = None):
        self.research_state_builder = research_state_builder or ResearchStateBuilder()

    def build_rows(
        self,
        spec: ProjectSpec,
        rounds: list[RoundRecord],
        research_state: Optional[ResearchState] = None,
    ) -> list[dict[str, Any]]:
        state = research_state or self.research_state_builder.build(spec, rounds)
        round_states = self._all_round_states(spec, rounds)
        baseline_value = (
            state.baseline_round.primary_metric_value
            if state.baseline_round is not None
            else None
        )
        best_value: Optional[float] = None
        rows: list[dict[str, Any]] = []

        for record, round_state in zip(rounds, round_states):
            value = round_state.primary_metric_value
            delta_vs_best = value - best_value if value is not None and best_value is not None else None
            if record.status == "completed" and value is not None:
                if best_value is None or self.research_state_builder.is_better(
                    value,
                    best_value,
                    state.primary_metric_goal,
                ):
                    best_value = value
            rows.append(
                {
                    "round": record.round_index,
                    "status": record.status,
                    "primary_metric": value,
                    "delta_vs_baseline": (
                        value - baseline_value
                        if value is not None and baseline_value is not None
                        else None
                    ),
                    "delta_vs_best": delta_vs_best,
                    "best_so_far": best_value,
                    "hypothesis": round_state.hypothesis,
                    "change_summary": round_state.change_summary,
                    "param_diff": round_state.param_diff,
                    "error": record.error or "",
                    "decision_reason": record.agent_decision.reason if record.agent_decision else "",
                }
            )
        return rows

    def export(
        self,
        session_id: int,
        spec: ProjectSpec,
        rounds: list[RoundRecord],
        output_dir: Path,
    ) -> dict[str, Path]:
        state = self.research_state_builder.build(spec, rounds)
        rows = self.build_rows(spec, rounds, state)
        session_dir = output_dir / f"session-{session_id:04d}"
        session_dir.mkdir(parents=True, exist_ok=True)
        csv_path = session_dir / "result_ledger.csv"
        jsonl_path = session_dir / "result_ledger.jsonl"
        state_path = session_dir / "research_state.json"

        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["param_diff"] = json.dumps(
                row["param_diff"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            writer.writerow(csv_row)
        csv_path.write_text(csv_buffer.getvalue(), encoding="utf-8")

        jsonl_text = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
        jsonl_path.write_text(jsonl_text, encoding="utf-8")
        state_path.write_text(
            json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "csv": csv_path,
            "jsonl": jsonl_path,
            "research_state": state_path,
        }

    def _all_round_states(self, spec: ProjectSpec, rounds: list[RoundRecord]):
        states = []
        for end in range(1, len(rounds) + 1):
            state = self.research_state_builder.build(spec, rounds[:end])
            if state.latest_round is not None:
                states.append(state.latest_round)
        return states
