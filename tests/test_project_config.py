from __future__ import annotations

from pathlib import Path

import yaml

from trainee.cli import init_project, main
from trainee.executor import TrainingExecutor
from trainee.project_config import (
    CommandArg,
    DataInput,
    LaunchConfig,
    ProjectConfig,
    RunConfig,
    TuningConfig,
    compile_project_spec,
    detect_project,
)


def test_structured_launchers_cover_supported_environments(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / ".venv" / "bin").mkdir(parents=True)
    (project / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
    (project / "train.py").write_text("", encoding="utf-8")
    (project / "data").mkdir()

    expected_prefixes = {
        "system": "python train.py",
        "uv": "uv run python train.py",
        "venv": f"{project}/.venv/bin/python train.py",
        "conda": "conda run -n trainer python train.py",
    }
    for environment, expected in expected_prefixes.items():
        config = ProjectConfig(
            data=[DataInput(path="data")],
            launch=LaunchConfig(
                environment=environment,
                env_name="trainer" if environment == "conda" else None,
                command=["python", "train.py"],
            ),
        )
        command = TrainingExecutor().render_command(compile_project_spec(project, config), {})
        assert command.startswith(expected)


def test_data_fixed_args_and_tunable_args_have_stable_order(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "data set").mkdir()
    config = ProjectConfig(
        data=[DataInput(path="data set", flag="--data-root")],
        launch=LaunchConfig(
            command=["python", "train.py"],
            args=[CommandArg(flag="--config", value="configs/base config.yaml")],
        ),
        run=RunConfig(
            max_rounds=2,
            fixed_args=[CommandArg(flag="--max-iter", value=1000)],
        ),
        tuning=TuningConfig(
            params=[
                {
                    "name": "lr",
                    "flag": "--lr",
                    "type": "float",
                    "default": 0.001,
                    "min_value": 0.00001,
                    "max_value": 0.1,
                }
            ]
        ),
    )

    spec = compile_project_spec(project, config)
    command = TrainingExecutor().render_command(spec, {"lr": 0.01})

    assert command.index("--config") < command.index("--data-root")
    assert command.index("--data-root") < command.index("--max-iter")
    assert command.index("--max-iter") < command.index("--lr")
    assert str(project / "data set") in command
    assert [item.name for item in spec.tunable_params] == ["lr"]
    assert "--max-iter" not in [item.flag for item in spec.tunable_params]


def test_detection_reports_conda_entrypoints_data_configs_and_limit(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "data").mkdir(parents=True)
    (project / "scripts").mkdir()
    (project / "configs").mkdir()
    (project / "environment.yml").write_text("name: trainer\n", encoding="utf-8")
    (project / "train.py").write_text(
        'parser.add_argument("--max-iter", type=int, default=200)\n',
        encoding="utf-8",
    )
    (project / "scripts" / "train_alt.py").write_text("", encoding="utf-8")
    (project / "configs" / "base.yaml").write_text("lr: 0.1\n", encoding="utf-8")

    discovery = detect_project(project)

    assert discovery.environment == "conda"
    assert discovery.env_name == "trainer"
    assert discovery.entrypoints[:2] == ["train.py", "scripts/train_alt.py"]
    assert discovery.data_dirs == ["data"]
    assert "configs/base.yaml" in discovery.config_files
    assert [(item.flag, item.value) for item in discovery.limit_flags] == [("--max-iter", 200)]


def test_dry_run_does_not_create_runtime_database(tmp_path: Path, capsys) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "data").mkdir()
    (project / "train.py").write_text("print('total_loss=1')\n", encoding="utf-8")
    init_project(project)

    exit_code = main(["run", str(project), "--unsafe", "--dry-run"])

    assert exit_code == 0
    assert "Baseline command" in capsys.readouterr().out
    assert not (project / ".trainee" / "runtime.sqlite3").exists()


def test_failed_preflight_does_not_create_runtime_database(tmp_path: Path, capsys) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "train.py").write_text("print('total_loss=1')\n", encoding="utf-8")
    init_project(project)

    exit_code = main(["run", str(project), "--unsafe"])

    assert exit_code == 1
    assert "no data paths configured" in capsys.readouterr().err
    assert not (project / ".trainee" / "runtime.sqlite3").exists()


def test_web_ui_saves_the_same_project_yaml(runtime_env) -> None:
    client = runtime_env["client"]
    project = runtime_env["external_project"]
    response = client.post(
        "/ui/project/register",
        data={
            "project_root": str(project),
            "data_lines": "data | --data-root",
            "launch_environment": "system",
            "launch_env_name": "",
            "launch_command": "python train.py",
            "launch_args_lines": "--log-file=.trainee/logs/train.log",
            "max_rounds": "2",
            "timeout_minutes": "5",
            "fixed_args_lines": "--max-iter=10",
            "tunable_params_yaml": "[]",
            "metric_specs_yaml": "[]",
            "metric_prompt": "",
            "tuning_prompt": "",
            "working_dir": ".",
            "security_mode": "unsafe",
            "advanced_yaml": "{}",
        },
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 200
    payload = yaml.safe_load((project / ".trainee" / "project.yaml").read_text(encoding="utf-8"))
    assert payload["data"] == [{"path": "data", "flag": "--data-root"}]
    assert payload["run"]["fixed_args"] == [{"flag": "--max-iter", "value": 10}]
    assert client.get("/api/project").json()["config"]["run"]["max_rounds"] == 2
