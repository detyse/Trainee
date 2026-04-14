from __future__ import annotations

from trainee.executor import TrainingExecutor
from trainee.models import ProjectSpec

REGISTER_PAYLOAD_TEMPLATE = {
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
