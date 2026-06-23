from __future__ import annotations

import json
import re

import pytest

from trainee.cli import build_tool_manifest, call_tool, init_project, load_tool_input, main


def test_tool_manifest_exposes_tool_call_safe_names():
    manifest = build_tool_manifest()
    tools = manifest["tools"]
    names = [item["function"]["name"] for item in tools]

    assert "project_register" in names
    assert "loop_start" in names
    assert "runs_get" in names
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
    project_json = json.loads((trainee_dir / "project.json").read_text(encoding="utf-8"))
    context_md = (trainee_dir / "context.md").read_text(encoding="utf-8")
    assert project_json["project_root"] == str(project)
    assert project_json["working_dir"] == str(project)
    assert project_json["launcher_template"] == "python {project_root}/train.py {extra_args}"
    assert project_json["security_mode"] == "guarded"
    assert project_json["log_paths"] == [".trainee/logs/**/*.log", ".trainee/runs/**/*.log"]
    assert "Toy Model" in context_md
    assert trainee_dir / "project.json" in result["files_written"]
    assert project / "README.md" in result["files_read"]


def test_init_command_prints_agent_style_activity(tmp_path, capsys):
    project = tmp_path / "toymodel"
    project.mkdir()
    (project / "train.py").write_text("print('train')\n", encoding="utf-8")

    exit_code = main(["init", str(project)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Trainee init" in output
    assert "- Read: train.py" in output
    assert "- Wrote: .trainee/project.json" in output
    assert "uvicorn" not in output.lower()


def test_launch_alias_still_initializes_project(tmp_path, capsys):
    project = tmp_path / "toymodel"
    project.mkdir()
    (project / "train.py").write_text("print('train')\n", encoding="utf-8")

    exit_code = main(["launch", str(project)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Trainee init" in output
    assert (project / ".trainee" / "project.json").exists()


def test_run_command_executes_project_config_unsafe(tmp_path, capsys):
    project = tmp_path / "toymodel"
    project.mkdir()
    (project / "train.py").write_text("print('total_loss=0.5')\n", encoding="utf-8")

    assert main(["init", str(project)]) == 0
    project_json_path = project / ".trainee" / "project.json"
    project_json = json.loads(project_json_path.read_text(encoding="utf-8"))
    project_json["max_rounds"] = 1
    project_json_path.write_text(json.dumps(project_json), encoding="utf-8")

    exit_code = main(["run", str(project), "--unsafe"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Trainee run" in output
    assert "- Security: unsafe" in output
    assert "- Status: stopped" in output
    assert (project / ".trainee" / "runtime.sqlite3").exists()
