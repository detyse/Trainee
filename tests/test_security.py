from __future__ import annotations

import asyncio
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from trainee.decision import DecisionEngine
from trainee.executor import TrainingExecutor
from trainee.models import AgentDecision, MetricSpec, ProjectContext, ProjectSpec, RoundRecord, TunableParam
from trainee.security import build_secure_command
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


def test_guarded_executor_rejects_log_paths_outside_trainee_dir(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "logs").mkdir(parents=True)
    spec = _spec(project).model_copy(update={"log_paths": ["logs/*.log"]})

    with pytest.raises(ValueError, match="log_paths must stay within project_root"):
        TrainingExecutor().validate_paths(spec, tmp_path / "artifacts")


def test_unsafe_executor_accepts_relative_globs_inside_project_root(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "logs").mkdir(parents=True)
    spec = _spec(project).model_copy(update={"security_mode": "unsafe", "log_paths": ["logs/*.log"]})

    TrainingExecutor().validate_paths(spec, tmp_path / "artifacts")


def test_guarded_executor_accepts_log_paths_inside_trainee_dir(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / ".trainee" / "logs").mkdir(parents=True)
    spec = _spec(project).model_copy(update={"log_paths": [".trainee/logs/*.log"]})

    TrainingExecutor().validate_paths(spec, tmp_path / "artifacts")


def test_secure_command_builds_bwrap_launcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr("trainee.security.shutil.which", lambda name: "/usr/bin/bwrap")

    secure = build_secure_command(
        project_root=project,
        working_dir=project,
        command="python train.py",
        security_mode="guarded",
        base_env={"PATH": "/bin"},
    )

    trainee_dir = project / ".trainee"
    assert secure.cwd is None
    assert secure.argv[:5] == ["/usr/bin/bwrap", "--die-with-parent", "--ro-bind", "/", "/"]
    bind_index = secure.argv.index("--bind")
    assert secure.argv[bind_index : bind_index + 3] == ["--bind", str(trainee_dir), str(trainee_dir)]
    assert "--tmpfs" in secure.argv
    assert "--dev-bind" in secure.argv
    assert secure.argv[-4:] == ["--chdir", str(project), "/bin/bash", "-lc", "python train.py"][-4:]
    assert secure.env["HOME"] == str(trainee_dir / "home")
    assert secure.env["WANDB_DIR"] == str(trainee_dir / "wandb")
    assert (trainee_dir / "runs").is_dir()
    assert (trainee_dir / "logs").is_dir()


def test_secure_command_unsafe_uses_plain_shell_with_redirected_env(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()

    secure = build_secure_command(
        project_root=project,
        working_dir=project,
        command="python train.py",
        security_mode="unsafe",
        base_env={},
    )

    assert secure.argv == ["/bin/bash", "-lc", "python train.py"]
    assert secure.cwd == project
    assert secure.env["XDG_CACHE_HOME"] == str(project / ".trainee" / "cache")


def test_guarded_secure_command_fails_closed_without_bwrap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr("trainee.security.shutil.which", lambda name: None)

    with pytest.raises(RuntimeError, match="requires bubblewrap"):
        build_secure_command(
            project_root=project,
            working_dir=project,
            command="python train.py",
            security_mode="guarded",
        )


def test_bwrap_allows_only_trainee_writes(tmp_path: Path) -> None:
    if shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is not installed")

    project = tmp_path / "project"
    project.mkdir()
    probe = build_secure_command(
        project_root=project,
        working_dir=project,
        command="true",
        security_mode="guarded",
    )
    probe_result = subprocess.run(probe.argv, env=probe.env, capture_output=True, text=True, timeout=5)
    if probe_result.returncode != 0:
        pytest.skip(f"bubblewrap unavailable in this environment: {probe_result.stderr.strip()}")

    script = (
        "from pathlib import Path\n"
        f"project = Path({str(project)!r})\n"
        "source_blocked = False\n"
        "try:\n"
        "    (project / 'blocked.txt').write_text('x', encoding='utf-8')\n"
        "except OSError:\n"
        "    source_blocked = True\n"
        "(project / '.trainee' / 'allowed.txt').write_text('ok', encoding='utf-8')\n"
        "raise SystemExit(0 if source_blocked else 1)\n"
    )
    secure = build_secure_command(
        project_root=project,
        working_dir=project,
        command=f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}",
        security_mode="guarded",
    )

    result = subprocess.run(secure.argv, env=secure.env, capture_output=True, text=True, timeout=5)

    assert result.returncode == 0, result.stderr + result.stdout
    assert not (project / "blocked.txt").exists()
    assert (project / ".trainee" / "allowed.txt").read_text(encoding="utf-8") == "ok"


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
