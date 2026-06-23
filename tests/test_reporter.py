from __future__ import annotations

from pathlib import Path

from trainee.models import AgentDecision, MetricSpec, ProjectContext, ProjectSpec, RoundRecord, RunSession, TunableParam
from trainee.reporter import ReportGenerator
from trainee.storage import Storage


def test_reporter_generates_markdown_and_saves_file(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "runtime.sqlite3")
    try:
        spec = ProjectSpec(
            project_root=str(tmp_path / "project"),
            working_dir=str(tmp_path / "project"),
            launcher_template="python train.py {extra_args}",
            tunable_params=[
                TunableParam(name="lr", flag="--lr", type="float", default=0.2, min_value=0.05, max_value=0.4),
            ],
            metric_specs=[
                MetricSpec(
                    name="total_loss",
                    source="log_regex",
                    key_or_pattern=r"total_loss=(?P<value>-?\d+(?:\.\d+)?)",
                    goal="min",
                    required=True,
                )
            ],
        )
        session = storage.create_session(
            RunSession(
                status="stopped",
                stop_reason="Reached max_rounds.",
                project_spec=spec,
                project_context=ProjectContext(project_summary="Fake trainer"),
            )
        )
        assert session.id is not None
        storage.create_round(
            RoundRecord(
                session_id=session.id,
                round_index=1,
                resolved_command="python train.py --lr 0.2",
                param_values={"lr": 0.2},
                status="completed",
                exit_code=0,
                metrics={"total_loss": 0.8},
                agent_decision=AgentDecision(
                    action="continue",
                    next_params={"lr": 0.1},
                    reason="Lower learning rate.",
                    focus_metrics=["total_loss"],
                ),
            )
        )
        storage.create_round(
            RoundRecord(
                session_id=session.id,
                round_index=2,
                resolved_command="python train.py --lr 0.1",
                param_values={"lr": 0.1},
                status="completed",
                exit_code=0,
                metrics={"total_loss": 0.4},
            )
        )

        generator = ReportGenerator(storage)
        report = generator.generate_session_report(session.id)
        report_path = generator.save_report(session.id, tmp_path / "artifacts")

        assert "# Trainee Session Report" in report
        assert "Best observed round: **2**" in report
        assert "total_loss=0.4" in report
        assert "Lower learning rate." in report
        assert report_path.read_text(encoding="utf-8") == report
    finally:
        storage.close()
