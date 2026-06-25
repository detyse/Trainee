from __future__ import annotations

import re

import pytest
import yaml

import trainee.cli as cli
from trainee.cli import build_tool_manifest, call_tool, init_project, load_tool_input, main


def test_tool_manifest_exposes_tool_call_safe_names():
    manifest = build_tool_manifest()
    tools = manifest["tools"]
    names = [item["function"]["name"] for item in tools]

    assert "project_register" in names
    assert "loop_start" in names
    assert "runs_get" in names
    assert "runtime_debug_update" in names
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

    result = init_project(project)

    trainee_dir = project / ".trainee"
    project_yaml = yaml.safe_load((trainee_dir / "project.yaml").read_text(encoding="utf-8"))
    context_md = (trainee_dir / "context.md").read_text(encoding="utf-8")
    program_md = (trainee_dir / "program.md").read_text(encoding="utf-8")
    assert project_yaml["launch"]["command"] == ["python", "train.py"]
    assert project_yaml["launch"]["args"] == [{"flag": "--config", "value": "config.yaml"}]
    assert project_yaml["advanced"]["security_mode"] == "guarded"
    assert project_yaml["advanced"]["log_paths"] == [".trainee/logs/**/*.log", ".trainee/runs/**/*.log"]
    assert "Toy Model" in context_md
    assert "# Trainee Agent Rules" in program_md
    assert "Change only parameters listed in `tunable_params`." in program_md
    assert trainee_dir / "project.yaml" in result["files_written"]
    assert trainee_dir / "program.md" in result["files_written"]
    assert project / "README.md" in result["files_read"]
    assert result["already_initialized"] is False


def test_init_project_detects_initialized_project_and_keeps_files(tmp_path):
    project = tmp_path / "toymodel"
    trainee_dir = project / ".trainee"
    trainee_dir.mkdir(parents=True)
    project_yaml = _minimal_project_yaml()
    (trainee_dir / "project.yaml").write_text(project_yaml, encoding="utf-8")
    (trainee_dir / "context.md").write_text("custom context\n", encoding="utf-8")
    (trainee_dir / "program.md").write_text("custom rules\n", encoding="utf-8")
    (trainee_dir / "README.md").write_text("custom readme\n", encoding="utf-8")

    result = init_project(project)

    assert result["already_initialized"] is True
    assert result["files_written"] == []
    assert set(result["files_skipped"]) == {
        trainee_dir / "project.yaml",
        trainee_dir / "context.md",
        trainee_dir / "program.md",
        trainee_dir / "README.md",
    }
    assert (trainee_dir / "project.yaml").read_text(encoding="utf-8") == project_yaml
    assert (trainee_dir / "program.md").read_text(encoding="utf-8") == "custom rules\n"


def test_init_project_adds_program_to_existing_project_without_overwriting_other_files(tmp_path):
    project = tmp_path / "toymodel"
    trainee_dir = project / ".trainee"
    trainee_dir.mkdir(parents=True)
    project_yaml = _minimal_project_yaml()
    (trainee_dir / "project.yaml").write_text(project_yaml, encoding="utf-8")
    (trainee_dir / "context.md").write_text("custom context\n", encoding="utf-8")
    (trainee_dir / "README.md").write_text("custom readme\n", encoding="utf-8")

    result = init_project(project)

    assert result["files_written"] == [trainee_dir / "program.md"]
    assert (trainee_dir / "project.yaml").read_text(encoding="utf-8") == project_yaml
    assert "# Trainee Agent Rules" in (trainee_dir / "program.md").read_text(encoding="utf-8")


def test_init_command_prints_agent_style_activity(tmp_path, capsys):
    project = tmp_path / "toymodel"
    project.mkdir()
    (project / "train.py").write_text("print('train')\n", encoding="utf-8")

    exit_code = main(["init", str(project)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Trainee init" in output
    assert "- Read: train.py" in output
    assert "- Wrote: .trainee/project.yaml" in output
    assert "- Wrote: .trainee/program.md" in output
    assert "- Next: edit .trainee/project.yaml, run `trainee doctor`, then run `trainee run`" in output
    assert "uvicorn" not in output.lower()


def test_init_command_reports_already_initialized_project(tmp_path, capsys):
    project = tmp_path / "toymodel"
    trainee_dir = project / ".trainee"
    trainee_dir.mkdir(parents=True)
    (trainee_dir / "project.yaml").write_text(_minimal_project_yaml(), encoding="utf-8")
    (trainee_dir / "context.md").write_text("custom context\n", encoding="utf-8")
    (trainee_dir / "program.md").write_text("custom rules\n", encoding="utf-8")
    (trainee_dir / "README.md").write_text("custom readme\n", encoding="utf-8")

    exit_code = main(["init", str(project)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "- Status: already initialized; kept existing project files" in output
    assert "- Kept: .trainee/project.yaml" in output


def test_launch_alias_still_initializes_project(tmp_path, capsys):
    project = tmp_path / "toymodel"
    project.mkdir()
    (project / "train.py").write_text("print('train')\n", encoding="utf-8")

    exit_code = main(["launch", str(project)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Trainee init" in output
    assert (project / ".trainee" / "project.yaml").exists()


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


def test_run_command_executes_project_config_unsafe(tmp_path, capsys):
    project = tmp_path / "toymodel"
    project.mkdir()
    (project / "data").mkdir()
    (project / "train.py").write_text("print('total_loss=0.5')\n", encoding="utf-8")

    assert main(["init", str(project)]) == 0
    project_yaml_path = project / ".trainee" / "project.yaml"
    project_yaml = yaml.safe_load(project_yaml_path.read_text(encoding="utf-8"))
    project_yaml["run"]["max_rounds"] = 1
    project_yaml_path.write_text(yaml.safe_dump(project_yaml, sort_keys=False), encoding="utf-8")

    exit_code = main(["run", str(project), "--unsafe"])

    output = capsys.readouterr().out
    assert exit_code == 0
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
