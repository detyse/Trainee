from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from trainee.cli import init_project, main
from trainee.executor import TrainingExecutor
from trainee.models import LoopSnapshot, ProjectContext, ProjectSpec
from trainee.project_config import (
    CommandArg,
    DataInput,
    LaunchConfig,
    ProjectConfig,
    RunConfig,
    TuningConfig,
    compile_project_spec,
    default_project_config,
    detect_project,
    load_project_config,
    save_project_config,
    tuning_config_path,
)
from trainee.storage import Storage


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
    (project / "configs").mkdir()
    (project / "configs" / "base config.yaml").write_text("lr: 0.001\n", encoding="utf-8")
    config = ProjectConfig(
        data=[DataInput(path="data set", flag="--data-root")],
        launch=LaunchConfig(
            command=["python", "train.py"],
            baseline_config="configs/base config.yaml",
            args=[CommandArg(flag="--seed", value=7)],
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

    assert command.index("--config") < command.index("--seed")
    assert command.index("--seed") < command.index("--data-root")
    assert command.index("--data-root") < command.index("--max-iter")
    assert command.index("--max-iter") < command.index("--lr")
    assert str(project / "data set") in command
    assert [item.name for item in spec.tunable_params] == ["lr"]
    assert "--max-iter" not in [item.flag for item in spec.tunable_params]


def test_config_backed_tunables_use_generated_round_config(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "configs").mkdir()
    (project / "configs" / "base.yaml").write_text(
        """
data:
  max_frames: null
fit:
  term_weights:
    theta: 8.0
""".lstrip(),
        encoding="utf-8",
    )
    config = ProjectConfig(
        launch=LaunchConfig(
            command=["python", "-m", "src.run_fitting"],
            baseline_config="configs/base.yaml",
        ),
        tuning=TuningConfig(
            params=[
                {
                    "name": "theta_weight",
                    "type": "float",
                    "default": 9.0,
                    "min_value": 1.0,
                    "max_value": 15.0,
                    "config_path": "fit.term_weights.theta",
                }
            ]
        ),
    )

    spec = compile_project_spec(project, config)
    command = TrainingExecutor().render_command(spec, {"theta_weight": 10.0}, session_id=3, round_index=2)

    assert spec.uses_generated_config()
    assert spec.default_params()["theta_weight"] == 8.0
    assert f"--config {project}/.trainee/runs/session-0003/round-0002/config.yaml" in command
    assert str(project / "configs" / "base.yaml") not in command
    assert "theta_weight" not in command


def test_baseline_config_without_tunables_still_uses_generated_round_config(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "configs").mkdir()
    baseline = project / "configs" / "base.yaml"
    baseline.write_text(
        """
data:
  max_frames: 5
fit:
  term_weights:
    theta: 9.0
""".lstrip(),
        encoding="utf-8",
    )
    config = ProjectConfig(
        launch=LaunchConfig(
            command=["python", "-m", "src.run_fitting"],
            baseline_config="configs/base.yaml",
        ),
    )

    spec = compile_project_spec(project, config)
    executor = TrainingExecutor()
    workspace = executor.round_workspace(spec, session_id=1, round_index=1)
    command = executor.render_command(spec, {}, session_id=1, round_index=1)
    written = executor.write_round_config(spec, {}, workspace)

    assert spec.uses_generated_config()
    assert f"--config {project}/.trainee/runs/session-0001/round-0001/config.yaml" in command
    assert str(baseline) not in command
    assert written == workspace.config_path
    assert yaml.safe_load(workspace.config_path.read_text(encoding="utf-8")) == yaml.safe_load(
        baseline.read_text(encoding="utf-8")
    )


def test_output_config_rewrites_generated_round_config(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "configs").mkdir()
    baseline = project / "configs" / "base.yaml"
    baseline.write_text(
        """
data:
  max_frames: 5
output:
  root: outputs
  run_name: baseline
""".lstrip(),
        encoding="utf-8",
    )
    config = ProjectConfig(
        launch=LaunchConfig(
            command=["python", "-m", "src.run_fitting"],
            baseline_config="configs/base.yaml",
        ),
        output={"config_path": "output.root"},
    )

    spec = compile_project_spec(project, config)
    executor = TrainingExecutor()
    workspace = executor.round_workspace(spec, session_id=3, round_index=2)
    executor.write_round_config(spec, {}, workspace)
    generated = yaml.safe_load(workspace.config_path.read_text(encoding="utf-8"))

    assert generated["output"]["root"] == str(workspace.round_dir / "outputs")
    assert generated["output"]["run_name"] == "baseline"
    assert yaml.safe_load(baseline.read_text(encoding="utf-8"))["output"]["root"] == "outputs"


def test_output_config_requires_baseline_config(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = ProjectConfig(
        launch=LaunchConfig(command=["python", "train.py"]),
        output={"config_path": "output.root"},
    )

    with pytest.raises(ValueError, match="output.config_path requires launch.baseline_config"):
        compile_project_spec(project, config)


def test_fixed_args_exclude_matching_tunable_flags_and_names() -> None:
    for tunable in (
        {"name": "max_iter", "flag": "--train-max-iter", "type": "int"},
        {"name": "other", "flag": "--max-iter", "type": "int"},
        {"name": "global_iterations", "config_path": "fit.stages.global.max_iters", "type": "int"},
    ):
        with pytest.raises(ValueError, match="tuning.yaml params must not include fixed launch/run args"):
            ProjectConfig(
                launch=LaunchConfig(command=["python", "train.py"]),
                run=RunConfig(fixed_args=[CommandArg(flag="--max-iter", value=1000)]),
                tuning=TuningConfig(params=[tunable]),
            )


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
    assert "environment.yml" not in discovery.config_files
    assert [(item.flag, item.value) for item in discovery.limit_flags] == [("--max-iter", 200)]


def test_default_config_requires_user_to_select_baseline(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "train.py").write_text("", encoding="utf-8")
    (project / "config.yaml").write_text("epochs: 1\n", encoding="utf-8")

    config = default_project_config(project)

    assert config.launch.baseline_config is None
    assert config.launch.args == []


def test_save_project_config_uses_atomic_replace_and_keeps_null_baseline(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "train.py").write_text("", encoding="utf-8")
    calls: list[tuple[Path, Path]] = []
    real_replace = __import__("os").replace

    def record_replace(source, destination):
        calls.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr("trainee.project_config.os.replace", record_replace)
    path = save_project_config(
        project,
        ProjectConfig(launch=LaunchConfig(command=["python", "train.py"])),
    )

    assert calls
    assert path in {destination for _, destination in calls}
    assert tuning_config_path(project) in {destination for _, destination in calls}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["launch"]["baseline_config"] is None
    assert "tuning" not in payload
    tuning_payload = yaml.safe_load(tuning_config_path(project).read_text(encoding="utf-8"))
    assert tuning_payload["params"] == []


def test_project_yaml_must_not_embed_tuning(tmp_path: Path) -> None:
    project = tmp_path / "project"
    trainee_dir = project / ".trainee"
    trainee_dir.mkdir(parents=True)
    (trainee_dir / "project.yaml").write_text(
        """
version: 1
launch:
  environment: system
  command: [python, train.py]
tuning:
  params: []
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not contain tuning"):
        load_project_config(project)


def test_baseline_config_must_exist_inside_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text("lr: 0.1\n", encoding="utf-8")

    for value in ("missing.yaml", str(outside)):
        config = ProjectConfig(
            launch=LaunchConfig(
                command=["python", "train.py"],
                baseline_config=value,
            )
        )
        try:
            compile_project_spec(project, config)
        except ValueError as exc:
            assert "launch.baseline_config" in str(exc)
        else:
            raise AssertionError(f"expected invalid baseline config: {value}")


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
    (project / "configs").mkdir(exist_ok=True)
    (project / "configs" / "base.yaml").write_text("lr: 0.1\n", encoding="utf-8")
    form = {
        "project_root": str(project),
        "data_lines": "data | --data-root",
        "launch_environment": "system",
        "launch_env_name": "",
        "launch_command": "python train.py",
        "baseline_config": "configs/base.yaml",
        "output_config_path": "",
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
    }
    response = client.post(
        "/ui/project/register",
        data=form,
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 200
    assert "Suggested 1 tunable parameter" in response.text
    assert not (project / ".trainee" / "project.yaml").exists()

    form.update(
        {
            "tunable_reviewed": "1",
            "tunable_params_yaml": """
- name: lr
  config_path: lr
  type: float
  default: 0.1
""".lstrip(),
        }
    )
    response = client.post(
        "/ui/project/register",
        data=form,
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 200
    payload = yaml.safe_load((project / ".trainee" / "project.yaml").read_text(encoding="utf-8"))
    assert payload["data"] == [{"path": "data", "flag": "--data-root"}]
    assert payload["launch"]["baseline_config"] == "configs/base.yaml"
    assert payload["output"] == {"config_path": None}
    assert payload["run"]["fixed_args"] == [{"flag": "--max-iter", "value": 10}]
    assert "tuning" not in payload
    tuning_payload = yaml.safe_load((project / ".trainee" / "tuning.yaml").read_text(encoding="utf-8"))
    assert tuning_payload["params"][0]["config_path"] == "lr"
    assert "default" not in tuning_payload["params"][0]
    assert client.get("/api/project").json()["config"]["run"]["max_rounds"] == 2


def test_api_registration_failure_restores_previous_project_yaml(runtime_env, monkeypatch) -> None:
    client = runtime_env["client"]
    runtime = client.app.state.runtime
    project = runtime_env["external_project"]
    registration = {
        "project_root": str(project),
        "version": 1,
        "data": [],
        "launch": {
            "environment": "system",
            "command": ["python", "train.py"],
            "baseline_config": None,
            "args": [],
        },
        "run": {
            "max_rounds": 2,
            "timeout_minutes": None,
            "fixed_args": [],
        },
        "tuning": {"params": []},
        "metrics": {"specs": [], "prompt": ""},
        "advanced": {
            "security_mode": "unsafe",
            "working_dir": ".",
            "signal_sources": [],
            "log_paths": [],
        },
    }
    assert client.post("/api/project/register", json=registration).status_code == 200
    path = project / ".trainee" / "project.yaml"
    tuning_path = project / ".trainee" / "tuning.yaml"
    previous = path.read_bytes()
    previous_tuning = tuning_path.read_bytes()

    def fail_registration(*args, **kwargs):
        raise ValueError("simulated registration failure")

    monkeypatch.setattr(runtime.storage, "save_project_registration", fail_registration)
    changed = registration | {"run": registration["run"] | {"max_rounds": 7}}
    response = client.post("/api/project/register", json=changed)

    assert response.status_code == 400
    assert path.read_bytes() == previous
    assert tuning_path.read_bytes() == previous_tuning
    assert runtime.storage.get_project_spec().max_rounds == 2


def test_project_registration_storage_is_transactional(tmp_path: Path, monkeypatch) -> None:
    storage = Storage(tmp_path / "runtime.sqlite3")
    old_spec = ProjectSpec(
        project_root=str(tmp_path),
        working_dir=str(tmp_path),
        launcher_template="python train.py",
        max_rounds=2,
    )
    old_context = ProjectContext(project_summary="old")
    old_snapshot = LoopSnapshot(status="ready", message="old")
    storage.save_project_registration(old_spec, old_context, old_snapshot)
    original_write = storage._write_setting

    def fail_on_context(key, value):
        if key == "project_context":
            raise RuntimeError("simulated transaction failure")
        original_write(key, value)

    monkeypatch.setattr(storage, "_write_setting", fail_on_context)
    new_spec = old_spec.model_copy(update={"max_rounds": 7})

    with pytest.raises(RuntimeError, match="transaction failure"):
        storage.save_project_registration(
            new_spec,
            ProjectContext(project_summary="new"),
            LoopSnapshot(status="ready", message="new"),
        )

    assert storage.get_project_spec().max_rounds == 2
    assert storage.get_project_context().project_summary == "old"
    assert storage.get_loop_snapshot().message == "old"
    storage.close()
