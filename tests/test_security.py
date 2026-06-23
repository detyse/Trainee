from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from trainee.decision import DecisionEngine
from trainee.executor import TrainingExecutor
from trainee.models import AgentDecision, MetricSpec, ProjectContext, ProjectSpec, RoundRecord, TunableParam
from trainee.settings import Settings


def test_string_param_choices_reject_invalid_values() -> None:
    param = TunableParam(
        name="optimizer",
        flag="--optimizer",
        type="str",
        default="adam",
        choices=["adam", "sgd"],
    )

    assert param.normalize_value("sgd") == "sgd"
    with pytest.raises(ValueError, match="must be one of"):
        param.normalize_value("rm -rf /")


def test_project_merge_rejects_unknown_params() -> None:
    spec = _spec(Path("/tmp/project"))

    with pytest.raises(ValueError, match="unknown tunable params"):
        spec.merge_param_values({"unknown": 1})


def test_executor_rejects_paths_outside_project_root(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    spec = _spec(project).model_copy(update={"log_paths": [str(outside / "*.log")]})

    with pytest.raises(ValueError, match="log_paths must stay within project_root"):
        TrainingExecutor().validate_paths(spec, tmp_path / "artifacts")


def test_executor_accepts_relative_globs_inside_project_root(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "logs").mkdir(parents=True)
    spec = _spec(project).model_copy(update={"log_paths": ["logs/*.log"]})

    TrainingExecutor().validate_paths(spec, tmp_path / "artifacts")


def test_invalid_provider_params_fall_back_without_unknown_values(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    engine = DecisionEngine(settings)
    spec = _spec(tmp_path / "project")
    context = ProjectContext(project_summary="Fake project")
    history = [
        RoundRecord(
            session_id=1,
            round_index=1,
            resolved_command="python train.py --lr 0.2",
            param_values={"lr": 0.2},
            status="completed",
            metrics={"total_loss": 1.0},
            exit_code=0,
        )
    ]

    async def invalid_provider(*args, **kwargs):
        return AgentDecision(action="continue", next_params={"unknown": "bad"}, reason="bad"), None

    engine._provider_decision = invalid_provider  # type: ignore[method-assign]

    result = asyncio.run(engine.decide_with_prompt(spec, context, history, {"lr": 0.2}))

    assert "unknown" not in result.decision.next_params
    assert set(result.decision.next_params) <= {"lr"}


def _spec(project_root: Path) -> ProjectSpec:
    return ProjectSpec(
        project_root=str(project_root),
        working_dir=str(project_root),
        launcher_template="python train.py {extra_args}",
        data_paths=[],
        log_paths=[],
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


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        repo_root=tmp_path,
        project_root=None,
        data_dir=tmp_path / ".trainee",
        database_path=tmp_path / ".trainee" / "runtime.sqlite3",
        artifacts_dir=tmp_path / ".trainee" / "artifacts",
        template_dir=tmp_path,
        static_dir=tmp_path,
        config_path=tmp_path / ".trainee" / "config.json",
        dotenv_path=tmp_path / ".env",
        llm_provider="none",
        llm_timeout_sec=5.0,
        openai_api_key=None,
        openai_base_url="https://api.openai.com/v1",
        openai_model="gpt-4o-mini",
        anthropic_api_key=None,
        anthropic_base_url="https://api.anthropic.com",
        anthropic_model="claude-3-5-haiku-latest",
        anthropic_version="2023-06-01",
        anthropic_max_tokens=1024,
    )
