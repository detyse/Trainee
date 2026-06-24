from __future__ import annotations

import shutil
import subprocess

import pytest

from trainee.executor import TrainingExecutor
from trainee.models import ProjectSpec

REGISTER_PAYLOAD_TEMPLATE = {
    "security_mode": "unsafe",
    "heartbeat_interval_sec": 0.1,
    "stall_timeout_sec": 1.5,
    "max_rounds": 2,
    "wandb_enabled": True,
    "tunable_params": [
        {"name": "lr", "flag": "--lr", "type": "float", "default": 0.2, "min_value": 0.05, "max_value": 0.4},
        {"name": "epochs", "flag": "--epochs", "type": "int", "default": 2, "min_value": 1, "max_value": 5},
    ],
    "metric_specs": [
        {
            "name": "total_loss",
            "source": "log_regex",
            "key_or_pattern": r"total_loss=(?P<value>-?\d+(?:\.\d+)?)",
            "goal": "min",
            "required": True,
        }
    ],
    "metric_prompt": "Use total_loss.",
    "tuning_prompt": "Lower lr if loss gets worse.",
}


def test_rendered_command_shell_quotes_values(runtime_env):
    external_project = runtime_env["external_project"]
    spec = REGISTER_PAYLOAD_TEMPLATE | {
        "project_root": str(external_project),
        "working_dir": str(external_project),
        "launcher_template": "python {project_root}/train.py --log-file {project_root}/logs/run.log {extra_args}",
        "data_paths": [str(external_project / "data")],
        "log_paths": [str(external_project / "logs" / "*.log")],
    }

    rendered = TrainingExecutor().render_command(
        ProjectSpec.model_validate(spec),
        {"lr": 0.2, "epochs": 2},
    )

    assert "--lr 0.2" in rendered
    assert "--epochs 2" in rendered
    assert "{extra_args}" not in rendered


def test_rendered_command_exposes_round_workspace(runtime_env):
    external_project = runtime_env["external_project"]
    spec = REGISTER_PAYLOAD_TEMPLATE | {
        "project_root": str(external_project),
        "working_dir": str(external_project),
        "launcher_template": "python {project_root}/scripts/wrapper.py --session-id {session_id} --round-index {round_index} --config-out {config_path} --work-dir {round_dir} {extra_args}",
        "data_paths": [str(external_project / "data")],
        "log_paths": [str(external_project / ".trainee" / "runs" / "**" / "*.log")],
    }

    rendered = TrainingExecutor().render_command(
        ProjectSpec.model_validate(spec),
        {"lr": 0.2, "epochs": 2},
        session_id=3,
        round_index=2,
    )

    assert "--session-id 3" in rendered
    assert "--round-index 2" in rendered
    assert ".trainee/runs/session-0003/round-0002/config.yaml" in rendered
    assert ".trainee/runs/session-0003/round-0002" in rendered
    assert "--lr 0.2" in rendered


def test_health_endpoint_reports_db_and_loop_state(runtime_env):
    client = runtime_env["client"]

    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["db_ok"] is True
    assert payload["loop_state"] in {"idle", "ready"}
    assert payload["llm_provider"] == "none"


def test_loop_runs_two_rounds_and_collects_metrics(runtime_env, wait_for):
    client = runtime_env["client"]
    external_project = runtime_env["external_project"]
    python = runtime_env["python"]

    register_payload = REGISTER_PAYLOAD_TEMPLATE | {
        "project_root": str(external_project),
        "working_dir": str(external_project),
        "launcher_template": f"{python} {{project_root}}/train.py --log-file {{project_root}}/logs/external.log {{extra_args}}",
        "data_paths": [str(external_project / "data")],
        "log_paths": [str(external_project / "logs" / "*.log")],
    }

    response = client.post("/api/project/register", json=register_payload)
    assert response.status_code == 200
    assert response.json()["context"]["project_summary"]

    start = client.post("/api/loop/start")
    assert start.status_code == 200

    wait_for(lambda: client.get("/api/loop").json()["status"] == "stopped")
    runs_payload = client.get("/api/runs").json()
    rounds = runs_payload["rounds"]
    session_id = runs_payload["sessions"][0]["id"]
    assert len(rounds) == 2
    assert rounds[0]["metrics"]["total_loss"] > 0
    assert rounds[1]["agent_decision"]["action"] in {"continue", "stop"}
    assert any("wandb.ai" in (item.get("wandb_run_url") or "") for item in rounds)

    detail_html = client.get(f"/fragments/run-detail?run_id={rounds[0]['id']}")
    assert detail_html.status_code == 200
    assert "Recent log tail" in detail_html.text

    index = client.get("/")
    assert index.status_code == 200
    assert "Agent Runtime / Agent Loop" in index.text

    report = client.get(f"/api/sessions/{session_id}/report")
    assert report.status_code == 200
    assert "Trainee Session Report" in report.text
    assert (runtime_env["data_dir"] / "artifacts" / f"session-{session_id:04d}" / "report.md").exists()


def test_loop_writes_wrapper_config_under_round_workspace(runtime_env, wait_for):
    client = runtime_env["client"]
    external_project = runtime_env["external_project"]
    python = runtime_env["python"]

    wrapper_path = external_project / "scripts" / "trainee_launch.py"
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path.write_text(
        """
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--config-out", required=True)
parser.add_argument("--round-dir", required=True)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=1)
args = parser.parse_args()

config_path = Path(args.config_out)
config_path.parent.mkdir(parents=True, exist_ok=True)
config_path.write_text(f"lr: {args.lr}\\nepochs: {args.epochs}\\n", encoding="utf-8")

log_path = Path(args.round_dir) / "logs" / "train.log"
log_path.parent.mkdir(parents=True, exist_ok=True)
line = "step=0 total_loss=0.25 loss=0.25\\n"
print(line, end="", flush=True)
log_path.write_text(line, encoding="utf-8")
""".strip()
        + "\n",
        encoding="utf-8",
    )

    register_payload = REGISTER_PAYLOAD_TEMPLATE | {
        "security_mode": "unsafe",
        "project_root": str(external_project),
        "working_dir": str(external_project),
        "launcher_template": f"{python} {{project_root}}/scripts/trainee_launch.py --config-out {{config_path}} --round-dir {{round_dir}} {{extra_args}}",
        "data_paths": [str(external_project / "data")],
        "log_paths": [str(external_project / ".trainee" / "runs" / "**" / "*.log")],
        "max_rounds": 1,
    }

    assert client.post("/api/project/register", json=register_payload).status_code == 200
    assert client.post("/api/loop/start").status_code == 200

    wait_for(lambda: client.get("/api/loop").json()["status"] == "stopped")
    runs_payload = client.get("/api/runs").json()
    session_id = runs_payload["sessions"][0]["id"]
    config_path = external_project / ".trainee" / "runs" / f"session-{session_id:04d}" / "round-0001" / "config.yaml"

    assert runs_payload["rounds"][0]["status"] == "completed"
    assert config_path.read_text(encoding="utf-8") == "lr: 0.2\nepochs: 2\n"
    assert str(config_path) in runs_payload["rounds"][0]["resolved_command"]


def test_loop_collects_metrics_from_log_file_created_during_round(runtime_env, wait_for):
    client = runtime_env["client"]
    external_project = runtime_env["external_project"]
    python = runtime_env["python"]

    script_path = external_project / "scripts" / "file_only_metrics.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(
        """
import argparse
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--log-file", required=True)
args, _ = parser.parse_known_args()

log_path = Path(args.log_file)
log_path.parent.mkdir(parents=True, exist_ok=True)
for step, loss in enumerate([0.9, 0.4]):
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"step={step} total_loss={loss}\\n")
    time.sleep(0.12)
""".strip()
        + "\n",
        encoding="utf-8",
    )

    log_glob = str(external_project / "logs" / "late-*.log")
    register_payload = REGISTER_PAYLOAD_TEMPLATE | {
        "project_root": str(external_project),
        "working_dir": str(external_project),
        "launcher_template": f"{python} {{project_root}}/scripts/file_only_metrics.py --log-file {{project_root}}/logs/late-created.log",
        "data_paths": [str(external_project / "data")],
        "log_paths": [],
        "signal_sources": [{"type": "log_file_mtime", "paths": [log_glob]}],
        "metric_specs": [
            {
                "name": "total_loss",
                "source": "log_file_regex",
                "path": log_glob,
                "key_or_pattern": r"total_loss=(?P<value>-?\d+(?:\.\d+)?)",
                "goal": "min",
                "required": True,
            }
        ],
        "heartbeat_interval_sec": 0.05,
        "stall_timeout_sec": 0.2,
        "max_rounds": 1,
    }

    assert client.post("/api/project/register", json=register_payload).status_code == 200
    assert client.post("/api/loop/start").status_code == 200

    wait_for(lambda: client.get("/api/loop").json()["status"] == "stopped")
    run = client.get("/api/runs").json()["rounds"][0]
    assert run["status"] == "completed"
    assert run["metrics"]["total_loss"] == 0.4
    assert any("late-created.log" in path for path in run["log_paths"])


def test_guarded_loop_writes_logs_under_trainee(runtime_env, wait_for):
    _skip_without_working_bwrap()
    client = runtime_env["client"]
    external_project = runtime_env["external_project"]
    python = runtime_env["python"]

    log_file = external_project / ".trainee" / "logs" / "guarded.log"
    register_payload = REGISTER_PAYLOAD_TEMPLATE | {
        "security_mode": "guarded",
        "project_root": str(external_project),
        "working_dir": str(external_project),
        "launcher_template": f"{python} {{project_root}}/train.py --log-file {log_file} {{extra_args}}",
        "data_paths": [str(external_project / "data")],
        "log_paths": [str(external_project / ".trainee" / "logs" / "*.log")],
        "max_rounds": 1,
    }

    response = client.post("/api/project/register", json=register_payload)
    assert response.status_code == 200, response.text
    assert client.post("/api/loop/start").status_code == 200

    wait_for(lambda: client.get("/api/loop").json()["status"] == "stopped")
    runs_payload = client.get("/api/runs").json()
    assert runs_payload["rounds"][0]["status"] == "completed"
    assert log_file.exists()


def _skip_without_working_bwrap() -> None:
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        pytest.skip("bubblewrap is not installed")
    result = subprocess.run(
        [bwrap, "--ro-bind", "/", "/", "--proc", "/proc", "/bin/true"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        pytest.skip(f"bubblewrap unavailable in this environment: {result.stderr.strip()}")


def test_stalled_round_marks_failed_session(runtime_env, wait_for):
    client = runtime_env["client"]
    external_project = runtime_env["external_project"]
    python = runtime_env["python"]

    register_payload = REGISTER_PAYLOAD_TEMPLATE | {
        "project_root": str(external_project),
        "working_dir": str(external_project),
        "launcher_template": f"{python} {{project_root}}/train.py --log-file {{project_root}}/logs/stall.log --stall {{extra_args}}",
        "data_paths": [str(external_project / "data")],
        "log_paths": [str(external_project / "logs" / "*.log")],
        "heartbeat_interval_sec": 0.1,
        "stall_timeout_sec": 0.3,
        "max_rounds": 1,
    }

    assert client.post("/api/project/register", json=register_payload).status_code == 200
    assert client.post("/api/loop/start").status_code == 200

    wait_for(lambda: client.get("/api/loop").json()["status"] == "failed")
    runs_payload = client.get("/api/runs").json()
    assert runs_payload["rounds"][0]["status"] == "stalled"


def test_round_timeout_terminates_process_and_marks_failed_session(runtime_env, wait_for):
    client = runtime_env["client"]
    external_project = runtime_env["external_project"]
    python = runtime_env["python"]

    register_payload = REGISTER_PAYLOAD_TEMPLATE | {
        "project_root": str(external_project),
        "working_dir": str(external_project),
        "launcher_template": f"{python} {{project_root}}/train.py --log-file {{project_root}}/logs/timeout.log --sleep 2 {{extra_args}}",
        "data_paths": [str(external_project / "data")],
        "log_paths": [str(external_project / "logs" / "*.log")],
        "heartbeat_interval_sec": 0.1,
        "stall_timeout_sec": 5.0,
        "round_timeout_sec": 0.3,
        "max_rounds": 1,
    }

    assert client.post("/api/project/register", json=register_payload).status_code == 200
    assert client.post("/api/loop/start").status_code == 200

    wait_for(lambda: client.get("/api/loop").json()["status"] == "failed")
    runs_payload = client.get("/api/runs").json()
    assert runs_payload["rounds"][0]["status"] == "timeout"
    assert "round_timeout_sec" in runs_payload["rounds"][0]["error"]


def test_force_stop_terminates_active_round_and_marks_session_stopped(runtime_env, wait_for):
    client = runtime_env["client"]
    external_project = runtime_env["external_project"]
    python = runtime_env["python"]

    register_payload = REGISTER_PAYLOAD_TEMPLATE | {
        "project_root": str(external_project),
        "working_dir": str(external_project),
        "launcher_template": f"{python} {{project_root}}/train.py --log-file {{project_root}}/logs/force-stop.log --sleep 2 {{extra_args}}",
        "data_paths": [str(external_project / "data")],
        "log_paths": [str(external_project / "logs" / "*.log")],
        "heartbeat_interval_sec": 0.1,
        "stall_timeout_sec": 5.0,
        "max_rounds": 1,
    }

    assert client.post("/api/project/register", json=register_payload).status_code == 200
    assert client.post("/api/loop/start").status_code == 200
    wait_for(lambda: client.get("/api/loop").json()["status"] == "running_round")

    stop = client.post("/api/loop/stop", json={"force": True})
    assert stop.status_code == 200

    wait_for(lambda: client.get("/api/loop").json()["status"] == "stopped")
    runs_payload = client.get("/api/runs").json()
    assert runs_payload["rounds"][0]["status"] == "stopped"


def test_resume_uses_source_session_snapshot_not_current_project_spec(runtime_env, wait_for):
    client = runtime_env["client"]
    external_project = runtime_env["external_project"]
    python = runtime_env["python"]

    original_payload = REGISTER_PAYLOAD_TEMPLATE | {
        "project_root": str(external_project),
        "working_dir": str(external_project),
        "launcher_template": f"{python} {{project_root}}/train.py --log-file {{project_root}}/logs/original.log {{extra_args}}",
        "data_paths": [str(external_project / "data")],
        "log_paths": [str(external_project / "logs" / "*.log")],
        "max_rounds": 2,
    }
    changed_payload = original_payload | {
        "launcher_template": f"{python} {{project_root}}/train.py --log-file {{project_root}}/logs/changed.log {{extra_args}}",
        "tunable_params": [
            {"name": "lr", "flag": "--lr", "type": "float", "default": 0.4, "min_value": 0.05, "max_value": 0.4},
            {"name": "epochs", "flag": "--epochs", "type": "int", "default": 2, "min_value": 1, "max_value": 5},
        ],
    }

    assert client.post("/api/project/register", json=original_payload).status_code == 200
    assert client.post("/api/loop/start").status_code == 200
    wait_for(lambda: client.get("/api/loop").json()["status"] == "running_round")
    assert client.post("/api/loop/stop").status_code == 200
    wait_for(lambda: client.get("/api/loop").json()["status"] == "stopped")
    source_session_id = client.get("/api/runs").json()["sessions"][0]["id"]

    assert client.post("/api/project/register", json=changed_payload).status_code == 200
    resume = client.post("/api/loop/start", json={"resume_session_id": source_session_id})
    assert resume.status_code == 200
    wait_for(lambda: client.get("/api/loop").json()["status"] == "stopped")

    runs_payload = client.get("/api/runs").json()
    resumed_session = runs_payload["sessions"][0]
    resumed_rounds = [item for item in runs_payload["rounds"] if item["session_id"] == resumed_session["id"]]
    assert resumed_session["resumed_from"] == source_session_id
    assert resumed_rounds[0]["round_index"] == 2
    assert resumed_rounds[0]["param_values"]["lr"] == 0.2
    assert "original.log" in resumed_rounds[0]["resolved_command"]
    assert "changed.log" not in resumed_rounds[0]["resolved_command"]
