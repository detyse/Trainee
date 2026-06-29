from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from trainee.cli import init_project, main
from trainee.doctor import _check_environment, _check_launcher, format_doctor_report, run_doctor
from trainee.models import OutputConfig, ProjectSpec, TunableParam
from trainee.provider_probe import ProviderProbeResult


def test_doctor_reports_missing_project_scaffold(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()

    report = run_doctor(project, skip_provider_test=True)

    assert report.has_failures
    text = format_doctor_report(report)
    assert "Trainee doctor" in text
    assert "[fail] .trainee exists: missing" in text
    assert "trainee init" in text


def test_init_project_creates_doctor_required_directories(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "train.py").write_text("print('train')\n", encoding="utf-8")

    init_project(project, skip_provider_test=True)

    assert (project / ".trainee" / "runs").is_dir()
    assert (project / ".trainee" / "logs").is_dir()


def test_doctor_reports_invalid_project_yaml(tmp_path: Path) -> None:
    project = tmp_path / "project"
    trainee_dir = project / ".trainee"
    (trainee_dir / "runs").mkdir(parents=True)
    (trainee_dir / "logs").mkdir()
    (trainee_dir / "project.yaml").write_text("launch: [", encoding="utf-8")

    report = run_doctor(project, skip_provider_test=True)

    assert report.has_failures
    assert "project.yaml is invalid" in format_doctor_report(report)


def test_doctor_fails_when_provider_live_test_fails(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "train.py").write_text("print('train')\n", encoding="utf-8")
    init_project(project, skip_provider_test=True)

    async def fake_probe(settings):
        return ProviderProbeResult(
            provider="moonshot",
            model="kimi-test",
            ok=False,
            status="request_failed",
            http_status=401,
            error_message="Provider returned HTTP 401.",
        )

    monkeypatch.setattr("trainee.doctor.probe_provider", fake_probe)

    report = run_doctor(project)

    assert report.has_failures
    assert "LLM provider live test failed" in format_doctor_report(report)


def test_launcher_analysis_warns_on_unsafe_outputs_and_unbounded_params(tmp_path: Path) -> None:
    spec = ProjectSpec(
        project_root=str(tmp_path),
        working_dir=str(tmp_path),
        launcher_template="python train.py --save_dir outputs > train.log {extra_args}",
        tunable_params=[TunableParam(name="lr", flag="--lr", type="float")],
    )

    section = _check_launcher(tmp_path, spec)
    messages = [finding.message for finding in section.findings]

    assert "launcher has no output_dir" in messages
    assert "launcher contains suspicious shell operator: >" in messages
    assert "save_dir may write outside .trainee: outputs" in messages
    assert "tunable params are not bounded: lr" in messages


def test_launcher_analysis_accepts_trainee_output_and_bounded_params(tmp_path: Path) -> None:
    spec = ProjectSpec(
        project_root=str(tmp_path),
        working_dir=str(tmp_path),
        launcher_template="python train.py --output_dir .trainee/runs/latest {extra_args}",
        tunable_params=[
            TunableParam(name="lr", flag="--lr", type="float", min_value=0.0, max_value=1.0),
            TunableParam(name="optimizer", flag="--optimizer", type="str", choices=["adam", "sgd"]),
            TunableParam(name="debug", flag="--debug", type="bool"),
        ],
    )

    section = _check_launcher(tmp_path, spec)
    messages = [finding.message for finding in section.findings]

    assert "output_dir points to .trainee/runs" in messages
    assert "tunable params are bounded" in messages


def test_launcher_analysis_accepts_round_workspace_output(tmp_path: Path) -> None:
    spec = ProjectSpec(
        project_root=str(tmp_path),
        working_dir=str(tmp_path),
        launcher_template="python scripts/trainee_launch.py --config-out {config_path} --output_dir {round_dir} {extra_args}",
        tunable_params=[
            TunableParam(name="lr", flag="--lr", type="float", min_value=0.0, max_value=1.0),
        ],
    )

    section = _check_launcher(tmp_path, spec)
    messages = [finding.message for finding in section.findings]

    assert "output_dir points to .trainee/runs" in messages
    assert not any("may write outside .trainee" in message for message in messages)


def test_launcher_analysis_accepts_config_output_path(tmp_path: Path) -> None:
    spec = ProjectSpec(
        project_root=str(tmp_path),
        working_dir=str(tmp_path),
        launcher_template="python train.py --config {config_path} {extra_args}",
        output=OutputConfig(config_path="output.root"),
        tunable_params=[
            TunableParam(name="lr", flag="--lr", type="float", min_value=0.0, max_value=1.0),
        ],
    )

    section = _check_launcher(tmp_path, spec)
    messages = [finding.message for finding in section.findings]

    assert any(message.startswith("config output path: output.root ->") for message in messages)
    assert "launcher has no output_dir" not in messages


def test_launcher_analysis_accepts_local_python_module(tmp_path: Path) -> None:
    module_dir = tmp_path / "pkg"
    module_dir.mkdir()
    (module_dir / "train.py").write_text("", encoding="utf-8")
    spec = ProjectSpec(
        project_root=str(tmp_path),
        working_dir=str(tmp_path),
        launcher_template="conda run -n trainer python -u -m pkg.train {extra_args}",
    )

    section = _check_launcher(tmp_path, spec)

    assert any(
        finding.status == "ok" and finding.message == "train module: pkg.train"
        for finding in section.findings
    )
    assert not any(finding.status == "fail" for finding in section.findings)


def test_launcher_analysis_warns_for_external_python_module(tmp_path: Path) -> None:
    spec = ProjectSpec(
        project_root=str(tmp_path),
        working_dir=str(tmp_path),
        launcher_template="uv run python -m installed_pkg.train {extra_args}",
    )

    section = _check_launcher(tmp_path, spec)

    assert any(
        finding.status == "warn" and "must be installed" in finding.message
        for finding in section.findings
    )
    assert not any(finding.status == "fail" for finding in section.findings)


def test_launcher_analysis_rejects_invalid_python_module(tmp_path: Path) -> None:
    spec = ProjectSpec(
        project_root=str(tmp_path),
        working_dir=str(tmp_path),
        launcher_template="python -m bad/module {extra_args}",
    )

    section = _check_launcher(tmp_path, spec)

    assert any(
        finding.status == "fail" and finding.message == "invalid python module entrypoint: bad/module"
        for finding in section.findings
    )


def test_environment_detects_uv_without_running_sync(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_which(name: str) -> str | None:
        return "/usr/bin/uv" if name == "uv" else None

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="usage: uv sync", stderr="")

    monkeypatch.setattr("trainee.doctor.shutil.which", fake_which)
    monkeypatch.setattr("trainee.doctor.subprocess.run", fake_run)

    section = _check_environment(tmp_path, None)

    assert [finding.message for finding in section.findings] == [
        "uv project detected",
        "uv found: /usr/bin/uv",
        "uv sync available",
    ]
    assert calls == [["/usr/bin/uv", "sync", "--help"]]


def test_environment_detects_missing_venv_python(tmp_path: Path) -> None:
    (tmp_path / ".venv").mkdir()

    section = _check_environment(tmp_path, None)

    assert any(finding.status == "fail" and finding.message == ".venv/bin/python not found" for finding in section.findings)


def test_environment_detects_missing_conda_env(tmp_path: Path, monkeypatch) -> None:
    spec = ProjectSpec(
        project_root=str(tmp_path),
        working_dir=str(tmp_path),
        launcher_template="conda run -n mousekin python train.py {extra_args}",
    )

    def fake_which(name: str) -> str | None:
        return "/opt/conda/bin/conda" if name == "conda" else None

    def fake_run(argv, **kwargs):
        payload = {"envs": ["/opt/conda/envs/other"]}
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("trainee.doctor.shutil.which", fake_which)
    monkeypatch.setattr("trainee.doctor.subprocess.run", fake_run)

    section = _check_environment(tmp_path, spec)

    assert any(finding.status == "fail" and finding.message == "conda env mousekin not found" for finding in section.findings)


def test_doctor_cli_returns_nonzero_only_for_failures(tmp_path: Path, capsys, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()

    assert main(["doctor", str(project), "--skip-provider-test"]) == 1
    assert "not ready" in capsys.readouterr().out

    (project / "data").mkdir()
    (project / "train.py").write_text("print('train')\n", encoding="utf-8")
    init_project(project, skip_provider_test=True)
    project_yaml_path = project / ".trainee" / "project.yaml"
    payload = yaml.safe_load(project_yaml_path.read_text(encoding="utf-8"))
    payload["launch"]["args"].append(
        {"flag": "--output_dir", "value": ".trainee/runs/latest"}
    )
    project_yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    def fake_which(name: str) -> str | None:
        paths = {
            "bwrap": "/usr/bin/bwrap",
            "python3": "/usr/bin/python3",
            "python": "/usr/bin/python",
        }
        return paths.get(name)

    monkeypatch.setattr("trainee.doctor.shutil.which", fake_which)

    assert main(["doctor", str(project), "--skip-provider-test"]) == 0
    output = capsys.readouterr().out
    assert "Trainee doctor" in output
    assert "Result\n  ready" in output
    assert "Baseline command" in output
