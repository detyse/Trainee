from __future__ import annotations

import os
import re

import pytest
import yaml

import trainee.cli as cli
from trainee.cli import build_tool_manifest, call_tool, init_project, load_tool_input, main, prepare_project
from trainee.provider_probe import ProviderProbeResult


def test_tool_manifest_exposes_tool_call_safe_names():
    manifest = build_tool_manifest()
    tools = manifest["tools"]
    names = [item["function"]["name"] for item in tools]

    assert "project_register" in names
    assert "loop_start" in names
    assert "runs_get" in names
    assert "runtime_debug_update" in names
    assert "runtime_provider_test" in names
    assert "runtime_system_prompt_get" in names
    assert "runtime_system_prompt_update" in names
    assert "project_get_program" not in names
    assert "project_update_program" not in names
    assert all(re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names)

    register_tool = next(item for item in tools if item["function"]["name"] == "project_register")
    assert register_tool["http"] == {"method": "POST", "path": "/api/project/register"}
    assert "project_root" in register_tool["function"]["parameters"]["properties"]


def test_single_tool_manifest_filters_by_name():
    manifest = build_tool_manifest(tool_name="loop_get", base_url="http://example.test")

    assert manifest["base_url"] == "http://example.test"
    assert [item["function"]["name"] for item in manifest["tools"]] == ["loop_get"]


def test_load_tool_input_requires_json_object():
    assert load_tool_input(None) == {}
    assert load_tool_input('{"run_id": 7}') == {"run_id": 7}

    with pytest.raises(ValueError, match="JSON object"):
        load_tool_input("[1, 2, 3]")


def test_call_tool_requires_path_params_before_http():
    with pytest.raises(ValueError, match="run_id"):
        call_tool("runs_get", {}, base_url="http://127.0.0.1:1", timeout=0.01)


def test_init_project_initializes_project_files(tmp_path):
    project = tmp_path / "toymodel"
    project.mkdir()
    (project / "README.md").write_text("# Toy Model\n\nTiny trainer.\n", encoding="utf-8")
    (project / "train.py").write_text("print('train')\n", encoding="utf-8")
    (project / "config.yaml").write_text("epochs: 1\n", encoding="utf-8")

    result = init_project(project, skip_provider_test=True)

    trainee_dir = project / ".trainee"
    project_yaml = yaml.safe_load((trainee_dir / "project.yaml").read_text(encoding="utf-8"))
    tuning_yaml = yaml.safe_load((trainee_dir / "tuning.yaml").read_text(encoding="utf-8"))
    context_md = (trainee_dir / "context.md").read_text(encoding="utf-8")
    assert project_yaml["launch"]["command"] == ["python", "train.py"]
    assert "env_name" in project_yaml["launch"]
    assert project_yaml["launch"]["env_name"] is None
    assert project_yaml["launch"]["baseline_config"] is None
    assert project_yaml["launch"]["args"] == []
    assert project_yaml["output"] == {"config_path": None}
    assert "tuning" not in project_yaml
    assert tuning_yaml["params"] == []
    assert project_yaml["advanced"]["security_mode"] == "guarded"
    assert project_yaml["advanced"]["log_paths"] == [".trainee/logs/**/*.log", ".trainee/runs/**/*.log"]
    assert "Toy Model" in context_md
    assert not (trainee_dir / "program.md").exists()
    assert trainee_dir / "project.yaml" in result["files_written"]
    assert project / "README.md" in result["files_read"]
    assert result["already_initialized"] is False


def test_init_project_detects_initialized_project_and_keeps_files(tmp_path):
    project = tmp_path / "toymodel"
    trainee_dir = project / ".trainee"
    trainee_dir.mkdir(parents=True)
    project_yaml = _minimal_project_yaml()
    (trainee_dir / "project.yaml").write_text(project_yaml, encoding="utf-8")
    (trainee_dir / "tuning.yaml").write_text("version: 1\nparams: []\n", encoding="utf-8")
    (trainee_dir / "context.md").write_text("custom context\n", encoding="utf-8")
    (trainee_dir / "program.md").write_text("custom rules\n", encoding="utf-8")
    (trainee_dir / "README.md").write_text("custom readme\n", encoding="utf-8")

    result = init_project(project, skip_provider_test=True)

    assert result["already_initialized"] is True
    assert result["files_written"] == []
    assert set(result["files_skipped"]) == {
        trainee_dir / "project.yaml",
        trainee_dir / "tuning.yaml",
        trainee_dir / "context.md",
        trainee_dir / "README.md",
    }
    assert (trainee_dir / "project.yaml").read_text(encoding="utf-8") == project_yaml
    assert (trainee_dir / "program.md").read_text(encoding="utf-8") == "custom rules\n"


def test_init_project_does_not_create_program_for_existing_project(tmp_path):
    project = tmp_path / "toymodel"
    trainee_dir = project / ".trainee"
    trainee_dir.mkdir(parents=True)
    project_yaml = _minimal_project_yaml()
    (trainee_dir / "project.yaml").write_text(project_yaml, encoding="utf-8")
    (trainee_dir / "tuning.yaml").write_text("version: 1\nparams: []\n", encoding="utf-8")
    (trainee_dir / "context.md").write_text("custom context\n", encoding="utf-8")
    (trainee_dir / "README.md").write_text("custom readme\n", encoding="utf-8")

    result = init_project(project, skip_provider_test=True)

    assert result["files_written"] == []
    assert (trainee_dir / "project.yaml").read_text(encoding="utf-8") == project_yaml
    assert not (trainee_dir / "program.md").exists()


def test_init_command_prints_agent_style_activity(tmp_path, capsys):
    project = tmp_path / "toymodel"
    project.mkdir()
    (project / "data").mkdir()
    (project / "train.py").write_text("print('train')\n", encoding="utf-8")
    (project / "config.yaml").write_text("epochs: 1\n", encoding="utf-8")

    exit_code = main(["init", str(project), "--skip-provider-test"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Trainee init" in output
    assert "- Status: initialized new project files" in output
    assert "- Read: train.py" in output
    assert "- Wrote: .trainee/project.yaml" in output
    assert "- Wrote: .trainee/program.md" not in output
    assert "Discovery" in output
    assert "- Environment: system" in output
    assert "- Entrypoints: train.py" in output
    assert "- Data candidates: data" in output
    assert "- Config candidates: config.yaml" in output
    assert "- Training limit candidates: none" in output
    assert "Effective configuration" in output
    assert "- Security: guarded" in output
    assert "- Budget: max_rounds=3, timeout=60 minutes" in output
    assert "- Data inputs: data" in output
    assert "- Baseline config: not set" in output
    assert "- Launch arguments: none" in output
    assert "- Fixed arguments: none" in output
    assert "- Tunable parameters: none" in output
    assert "- Metrics: none (built-in loss/total_loss parsing only)" in output
    assert "- Runtime: round_timeout=60 minutes, wandb=disabled" in output
    assert (
        "- Activity monitor: every 5s, "
        "sources=stdout; log_file_mtime(.trainee/logs/**/*.log, .trainee/runs/**/*.log)"
    ) in output
    assert "- Log paths: .trainee/logs/**/*.log, .trainee/runs/**/*.log" in output
    assert "- Launcher: python train.py {extra_args}" in output
    assert "- Review: .trainee/project.yaml, .trainee/tuning.yaml, and .trainee/context.md" in output
    assert "- Prepare: set launch.baseline_config, then run `trainee prepare`" in output
    assert "- Validate: trainee doctor or trainee run --dry-run" in output
    assert "- Next: run `trainee prepare`, review generated config, then run `trainee doctor`" in output
    assert "uvicorn" not in output.lower()


def test_init_command_fails_when_provider_test_fails(tmp_path, capsys, monkeypatch):
    project = tmp_path / "toymodel"
    project.mkdir()
    (project / "train.py").write_text("print('train')\n", encoding="utf-8")

    async def fake_probe_provider(settings):
        return ProviderProbeResult(
            provider="moonshot",
            model="kimi-test",
            ok=False,
            status="request_failed",
            http_status=401,
            error_message="Provider returned HTTP 401.",
        )

    monkeypatch.setattr(cli, "probe_provider", fake_probe_provider)

    exit_code = main(["init", str(project)])

    assert exit_code == 1
    assert "LLM provider test failed" in capsys.readouterr().err


def test_init_command_sets_explicit_baseline_config(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TRAINEE_LLM_PROVIDER", "none")
    project = tmp_path / "toymodel"
    (project / "configs").mkdir(parents=True)
    (project / "train.py").write_text("print('train')\n", encoding="utf-8")
    (project / "configs" / "base.yaml").write_text("lr: 0.001\n", encoding="utf-8")

    exit_code = main(
        [
            "init",
            str(project),
            "--baseline-config",
            "configs/base.yaml",
            "--skip-provider-test",
        ]
    )

    assert exit_code == 0
    payload = yaml.safe_load((project / ".trainee" / "project.yaml").read_text(encoding="utf-8"))
    tuning_payload = yaml.safe_load((project / ".trainee" / "tuning.yaml").read_text(encoding="utf-8"))
    assert payload["launch"]["baseline_config"] == "configs/base.yaml"
    assert "tuning" not in payload
    assert tuning_payload["params"] == []
    output = capsys.readouterr().out
    assert "- Baseline config: configs/base.yaml" in output
    assert "- Launcher: python train.py --config {config_path} {extra_args}" in output


def test_prepare_supplements_empty_tuning_after_project_yaml_baseline_is_set(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TRAINEE_LLM_PROVIDER", "none")
    project = tmp_path / "toymodel"
    (project / "configs").mkdir(parents=True)
    (project / "train.py").write_text("print('train')\n", encoding="utf-8")
    (project / "configs" / "base.yaml").write_text("lr: 0.001\n", encoding="utf-8")

    assert main(["init", str(project), "--skip-provider-test"]) == 0
    project_yaml_path = project / ".trainee" / "project.yaml"
    project_yaml = yaml.safe_load(project_yaml_path.read_text(encoding="utf-8"))
    project_yaml["launch"]["baseline_config"] = "configs/base.yaml"
    project_yaml_path.write_text(yaml.safe_dump(project_yaml, sort_keys=False), encoding="utf-8")
    edited_project_yaml = project_yaml_path.read_text(encoding="utf-8")
    capsys.readouterr()

    assert main(["prepare", str(project), "--skip-provider-test"]) == 0

    output = capsys.readouterr().out
    tuning_payload = yaml.safe_load((project / ".trainee" / "tuning.yaml").read_text(encoding="utf-8"))
    assert project_yaml_path.read_text(encoding="utf-8") == edited_project_yaml
    assert tuning_payload["params"][0]["config_path"] == "lr"
    assert "Trainee prepare" in output
    assert "- Wrote: .trainee/tuning.yaml" in output
    assert "- Generated: 1 parameter(s) in tuning.yaml params" in output


def test_prepare_uses_configured_output_root(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TRAINEE_LLM_PROVIDER", "none")
    project = tmp_path / "toymodel"
    (project / "configs").mkdir(parents=True)
    (project / "train.py").write_text("print('train')\n", encoding="utf-8")
    (project / "configs" / "base.yaml").write_text(
        """
lr: 0.001
output:
  root: outputs
  run_name: baseline
""".lstrip(),
        encoding="utf-8",
    )

    assert main(["init", str(project), "--baseline-config", "configs/base.yaml", "--skip-provider-test"]) == 0
    project_yaml_path = project / ".trainee" / "project.yaml"
    project_yaml = yaml.safe_load(project_yaml_path.read_text(encoding="utf-8"))
    project_yaml["output"]["config_path"] = "output.root"
    project_yaml_path.write_text(yaml.safe_dump(project_yaml, sort_keys=False), encoding="utf-8")
    capsys.readouterr()

    assert main(["prepare", str(project), "--skip-provider-test"]) == 0

    output = capsys.readouterr().out
    project_yaml = yaml.safe_load((project / ".trainee" / "project.yaml").read_text(encoding="utf-8"))
    tuning_yaml = yaml.safe_load((project / ".trainee" / "tuning.yaml").read_text(encoding="utf-8"))
    assert project_yaml["output"] == {"config_path": "output.root"}
    assert tuning_yaml["params"][0]["config_path"] == "lr"
    assert "- Status: configured (output.root -> {round_dir}/outputs)" in output


def test_tunables_discover_applies_to_tuning_yaml(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TRAINEE_LLM_PROVIDER", "none")
    project = tmp_path / "toymodel"
    (project / "configs").mkdir(parents=True)
    (project / "train.py").write_text("print('train')\n", encoding="utf-8")
    (project / "configs" / "base.yaml").write_text("lr: 0.001\n", encoding="utf-8")
    assert main(["init", str(project), "--baseline-config", "configs/base.yaml", "--skip-provider-test"]) == 0
    (project / ".trainee" / "tuning.yaml").write_text("version: 1\nparams: []\n", encoding="utf-8")
    capsys.readouterr()

    exit_code = main(["tunables", "discover", str(project), "--apply"])

    output = capsys.readouterr().out
    tuning_payload = yaml.safe_load((project / ".trainee" / "tuning.yaml").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "- Applied: 1 to .trainee/tuning.yaml" in output
    assert tuning_payload["params"][0]["config_path"] == "lr"


def test_prepare_excludes_fixed_args_from_generated_tunables(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TRAINEE_LLM_PROVIDER", "none")
    project = tmp_path / "toymodel"
    (project / "configs").mkdir(parents=True)
    (project / "train.py").write_text(
        'parser.add_argument("--max-iter", type=int, default=1000)\n',
        encoding="utf-8",
    )
    (project / "configs" / "base.yaml").write_text(
        """
fit:
  stages:
    global:
      max_iters: 1000
  term_weights:
    theta: 9.0
""".lstrip(),
        encoding="utf-8",
    )

    init_project(project, baseline_config="configs/base.yaml", skip_provider_test=True)
    result = prepare_project(project, skip_provider_test=True)

    paths = {item.config_path for item in result["config"].tuning.params}
    assert "fit.term_weights.theta" in paths
    assert "fit.stages.global.max_iters" not in paths


def test_init_command_reports_already_initialized_project(tmp_path, capsys):
    project = tmp_path / "toymodel"
    trainee_dir = project / ".trainee"
    trainee_dir.mkdir(parents=True)
    (trainee_dir / "project.yaml").write_text(_minimal_project_yaml(), encoding="utf-8")
    (trainee_dir / "tuning.yaml").write_text("version: 1\nparams: []\n", encoding="utf-8")
    (trainee_dir / "context.md").write_text("custom context\n", encoding="utf-8")
    (trainee_dir / "program.md").write_text("custom rules\n", encoding="utf-8")
    (trainee_dir / "README.md").write_text("custom readme\n", encoding="utf-8")

    exit_code = main(["init", str(project), "--skip-provider-test"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "- Status: already initialized; kept existing project files" in output
    assert "- Kept: .trainee/project.yaml" in output


def test_launch_alias_is_removed(tmp_path):
    project = tmp_path / "toymodel"
    project.mkdir()
    (project / "train.py").write_text("print('train')\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(["launch", str(project)])

    assert exc.value.code == 2
    assert not (project / ".trainee" / "project.yaml").exists()


def test_suggest_tunables_command_is_removed(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["suggest-tunables", str(tmp_path)])

    assert exc.value.code == 2


def test_webui_command_opens_browser_and_starts_service(monkeypatch):
    captured: dict[str, object] = {}

    def fake_open(url):
        captured["url"] = url
        return True

    def fake_run(app, host, port, reload):
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port
        captured["reload"] = reload

    monkeypatch.setattr(cli.webbrowser, "open", fake_open)
    monkeypatch.setattr(cli.uvicorn, "run", fake_run)

    exit_code = main(["webui", "--host", "0.0.0.0", "--port", "8765", "--reload"])

    assert exit_code == 0
    assert captured["url"] == "http://127.0.0.1:8765/"
    assert captured["app"] == "trainee.app:app"
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 8765
    assert captured["reload"] is True


def test_webui_command_binds_project_root_and_restores_env(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    previous_project = tmp_path / "previous"
    previous_project.mkdir()
    captured: dict[str, object] = {}

    def fake_run(app, host, port, reload):
        captured["app"] = app
        captured["project_root"] = os.environ.get("TRAINEE_PROJECT_ROOT")

    monkeypatch.setenv("TRAINEE_PROJECT_ROOT", str(previous_project))
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: True)
    monkeypatch.setattr(cli.uvicorn, "run", fake_run)

    exit_code = main(["webui", str(project), "--no-open"])

    assert exit_code == 0
    assert captured["app"] == "trainee.app:app"
    assert captured["project_root"] == str(project.resolve())
    assert os.environ["TRAINEE_PROJECT_ROOT"] == str(previous_project)


def test_serve_command_binds_project_root(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    captured: dict[str, object] = {}

    def fake_run(app, host, port, reload):
        captured["app"] = app
        captured["project_root"] = os.environ.get("TRAINEE_PROJECT_ROOT")

    monkeypatch.delenv("TRAINEE_PROJECT_ROOT", raising=False)
    monkeypatch.setattr(cli.uvicorn, "run", fake_run)

    exit_code = main(["serve", str(project)])

    assert exit_code == 0
    assert captured["app"] == "trainee.app:app"
    assert captured["project_root"] == str(project.resolve())
    assert "TRAINEE_PROJECT_ROOT" not in os.environ


def test_serve_command_rejects_invalid_project_root(tmp_path, capsys, monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(app, host, port, reload):
        captured["called"] = True

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)

    exit_code = main(["serve", str(tmp_path / "missing")])

    assert exit_code == 1
    assert "project root does not exist or is not a directory" in capsys.readouterr().err
    assert "called" not in captured


def test_version_command_prints_version_and_last_update(capsys):
    exit_code = main(["version"])

    assert exit_code == 0
    assert capsys.readouterr().out == "Trainee 0.1.7\nLast updated: 2026-06-30 13:18:01 +08:00\n"


def test_run_command_executes_project_config_unsafe(tmp_path, capsys, monkeypatch):
    project = tmp_path / "toymodel"
    project.mkdir()
    (project / "data").mkdir()
    (project / "train.py").write_text("print('total_loss=0.5')\n", encoding="utf-8")

    async def fake_probe_provider(settings):
        return ProviderProbeResult(provider="openai", model="test-model", ok=True, status="success")

    monkeypatch.setattr(cli, "probe_provider", fake_probe_provider)

    assert main(["init", str(project), "--skip-provider-test"]) == 0
    project_yaml_path = project / ".trainee" / "project.yaml"
    project_yaml = yaml.safe_load(project_yaml_path.read_text(encoding="utf-8"))
    project_yaml["run"]["max_rounds"] = 1
    project_yaml_path.write_text(yaml.safe_dump(project_yaml, sort_keys=False), encoding="utf-8")

    exit_code = main(["run", str(project), "--unsafe"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Round 1/1 started" in output
    assert "Round 1/1 completed" in output
    assert "Trainee run" in output
    assert "- Security: unsafe" in output
    assert "- Status: stopped" in output
    assert (project / ".trainee" / "runtime.sqlite3").exists()


def _minimal_project_yaml() -> str:
    return yaml.safe_dump(
        {
            "version": 1,
            "launch": {
                "environment": "system",
                "command": ["python", "train.py"],
                "args": [],
            },
        },
        sort_keys=False,
    )
