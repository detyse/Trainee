from __future__ import annotations

import json

import yaml

from trainee.models import AgentTrace, RoundRecord, RunSession
from trainee.provider_probe import ProviderProbeResult, ProviderProbeAttempt
from trainee.settings import load_settings

REGISTER_PAYLOAD_TEMPLATE = {
    "security_mode": "unsafe",
    "heartbeat_interval_sec": 0.1,
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


def _project_form(payload: dict[str, object]) -> dict[str, object]:
    shell_command = str(payload["launcher_template"]).replace("{extra_args}", "").strip()
    advanced = {
        "heartbeat_interval_sec": payload.get("heartbeat_interval_sec", 5.0),
        "signal_sources": payload.get("signal_sources", []),
        "log_paths": payload.get("log_paths", []),
        "shell_command": shell_command,
    }
    return {
        "project_root": payload["project_root"],
        "working_dir": payload["working_dir"],
        "launch_environment": "system",
        "launch_env_name": "",
        "launch_command": "python train.py",
        "launch_args_lines": "",
        "fixed_args_lines": "",
        "data_lines": "\n".join(str(path) for path in payload.get("data_paths", [])),
        "advanced_yaml": yaml.safe_dump(advanced, sort_keys=False),
        "security_mode": payload.get("security_mode", "guarded"),
        "wandb_enabled": "on" if payload.get("wandb_enabled") else "",
        "timeout_minutes": "",
        "max_rounds": str(payload.get("max_rounds", 3)),
        "tunable_params_yaml": yaml.safe_dump(payload.get("tunable_params", []), sort_keys=False),
        "metric_specs_yaml": yaml.safe_dump(payload.get("metric_specs", []), sort_keys=False),
        "metric_prompt": payload.get("metric_prompt", ""),
        "tuning_prompt": payload.get("tuning_prompt", ""),
    }


def _registration(payload: dict[str, object]) -> dict[str, object]:
    shell_command = str(payload["launcher_template"]).replace("{extra_args}", "").strip()
    return {
        "project_root": payload["project_root"],
        "version": 1,
        "data": [{"path": path} for path in payload.get("data_paths", [])],
        "launch": {
            "environment": "system",
            "command": ["python", "train.py"],
            "args": [],
        },
        "run": {
            "max_rounds": payload.get("max_rounds", 3),
            "timeout_minutes": None,
            "fixed_args": [],
        },
        "tuning": {"params": payload.get("tunable_params", [])},
        "metrics": {
            "specs": payload.get("metric_specs", []),
            "prompt": payload.get("metric_prompt", ""),
        },
        "advanced": {
            "security_mode": payload.get("security_mode", "guarded"),
            "working_dir": payload.get("working_dir"),
            "heartbeat_interval_sec": payload.get("heartbeat_interval_sec", 5.0),
            "signal_sources": payload.get("signal_sources", []),
            "log_paths": payload.get("log_paths", []),
            "wandb_enabled": payload.get("wandb_enabled", False),
            "tuning_prompt": payload.get("tuning_prompt", ""),
            "shell_command": shell_command,
        },
    }


def test_prompt_preview_api_shows_model_request(runtime_env):
    client = runtime_env["client"]
    external_project = runtime_env["external_project"]
    python = runtime_env["python"]
    register_payload = REGISTER_PAYLOAD_TEMPLATE | {
        "project_root": str(external_project),
        "working_dir": str(external_project),
        "launcher_template": f"{python} {{project_root}}/train.py --log-file {{project_root}}/logs/preview.log {{extra_args}}",
        "data_paths": [str(external_project / "data")],
        "log_paths": [str(external_project / "logs" / "*.log")],
    }

    assert client.post("/api/project/register", json=_registration(register_payload)).status_code == 200

    response = client.get("/api/prompt-preview")

    assert response.status_code == 200
    payload = response.json()
    preview = payload["prompt_preview"]
    assert payload["label"] == "Next decision preview based on saved project state"
    assert preview["status"] == "preview"
    assert preview["provider"] == "none"
    assert preview["system_prompt"].startswith("# Trainee System Prompt")
    assert "JSON only" in preview["system_prompt"]
    assert preview["user_prompt"].startswith("<STATIC_CONTEXT>\n")
    assert "\n</STATIC_CONTEXT>\n\n<DYNAMIC_ROUND_STATE>\n" in preview["user_prompt"]
    assert '"tuning_prompt":"Lower lr if loss gets worse."' in preview["user_prompt"]
    assert preview["payload"]["user"] == preview["user_prompt"]
    assert preview["static_context_json"]["tuning_prompt"] == "Lower lr if loss gets worse."
    assert preview["dynamic_state_json"]["current_params"] == {"epochs": 2, "lr": 0.2}


def test_system_prompt_ui_updates_global_config_and_prompt_preview(runtime_env):
    client = runtime_env["client"]
    external_project = runtime_env["external_project"]
    python = runtime_env["python"]
    register_payload = REGISTER_PAYLOAD_TEMPLATE | {
        "project_root": str(external_project),
        "working_dir": str(external_project),
        "launcher_template": f"{python} {{project_root}}/train.py --log-file {{project_root}}/logs/program.log {{extra_args}}",
    }
    assert client.post("/api/project/register", json=_registration(register_payload)).status_code == 200

    custom_prompt = "# Custom System Prompt\n\nReturn exactly the configured response."
    response = client.post(
        "/ui/runtime/system-prompt",
        data={"system_prompt": custom_prompt},
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 200
    config_path = runtime_env["config_path"]
    assert json.loads(config_path.read_text(encoding="utf-8"))["system_prompt"] == custom_prompt
    assert not (external_project / ".trainee" / "program.md").exists()
    payload = client.get("/api/runtime/system-prompt").json()
    assert payload == {
        "system_prompt": custom_prompt,
        "config_path": str(config_path),
    }
    reloaded = load_settings(repo_root=external_project, global_config_path=config_path)
    assert reloaded.system_prompt == custom_prompt
    assert client.get("/api/project/program").status_code == 404
    assert client.post("/api/project/program", json={"content": "legacy"}).status_code == 404
    preview = client.get("/api/prompt-preview").json()["prompt_preview"]
    assert preview["system_prompt"] == custom_prompt
    assert "Runtime response contract" not in preview["system_prompt"]


def test_system_prompt_rejects_blank_and_running_loop_updates(runtime_env, monkeypatch):
    client = runtime_env["client"]
    runtime = client.app.state.runtime

    response = client.post("/api/runtime/system-prompt", json={"system_prompt": "   "})
    assert response.status_code == 400
    assert "cannot be blank" in response.json()["detail"]

    monkeypatch.setattr(runtime, "loop_is_running", lambda: True)
    response = client.post("/api/runtime/system-prompt", json={"system_prompt": "new prompt"})
    assert response.status_code == 409
    assert "stop the loop" in response.json()["detail"]


def test_runtime_provider_settings_save_to_config_json(runtime_env):
    client = runtime_env["client"]
    config_path = runtime_env["config_path"]

    response = client.post(
        "/ui/runtime/provider",
        data={
            "llm_provider": "openai",
            "llm_timeout_sec": "12",
            "llm_temperature": "0.7",
            "openai_api_key": "test-openai-key",
            "openai_base_url": "https://openai.example/v1",
            "openai_model": "gpt-ui",
            "moonshot_api_key": "test-moonshot-key",
            "moonshot_base_url": "https://moonshot.example/v1",
            "moonshot_model": "kimi-ui",
            "anthropic_base_url": "https://anthropic.example",
            "anthropic_model": "claude-ui",
            "anthropic_version": "2023-06-01",
            "anthropic_max_tokens": "777",
        },
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 200
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["llm_provider"] == "openai"
    assert config["llm_timeout_sec"] == 12.0
    assert config["llm_temperature"] == 0.7
    assert config["openai"]["api_key"] == "test-openai-key"
    assert config["openai"]["base_url"] == "https://openai.example/v1"
    assert config["openai"]["model"] == "gpt-ui"
    assert config["moonshot"]["api_key"] == "test-moonshot-key"
    assert config["moonshot"]["base_url"] == "https://moonshot.example/v1"
    assert config["moonshot"]["model"] == "kimi-ui"
    assert client.get("/api/health").json()["llm_provider"] == "openai"


def test_runtime_provider_settings_api_manages_config_without_exposing_keys(runtime_env):
    client = runtime_env["client"]
    config_path = runtime_env["config_path"]

    response = client.post(
        "/api/runtime/provider",
        json={
            "llm_provider": "moonshot",
            "llm_timeout_sec": 9,
            "llm_temperature": 0.8,
            "moonshot": {
                "api_key": "moonshot-secret",
                "base_url": "https://moonshot.example/v1",
                "model": "kimi-api",
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["llm_provider"] == "moonshot"
    assert payload["llm_temperature"] == 0.8
    assert payload["active_model"] == "kimi-api"
    assert payload["moonshot_key_configured"] is True
    assert "api_key" not in json.dumps(payload)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["llm_provider"] == "moonshot"
    assert config["llm_temperature"] == 0.8
    assert config["moonshot"]["api_key"] == "moonshot-secret"
    assert client.get("/api/health").json()["llm_provider"] == "moonshot"


def test_runtime_provider_settings_reject_negative_temperature(runtime_env):
    client = runtime_env["client"]

    api_response = client.post(
        "/api/runtime/provider",
        json={"llm_provider": "none", "llm_temperature": -0.1},
    )
    ui_response = client.post(
        "/ui/runtime/provider",
        data={"llm_provider": "none", "llm_temperature": "-0.1"},
        headers={"HX-Request": "true"},
    )

    assert api_response.status_code == 400
    assert "temperature" in api_response.json()["detail"]
    assert ui_response.status_code == 400
    assert "temperature" in ui_response.json()["detail"]


def test_runtime_provider_test_api_returns_probe_result(runtime_env, monkeypatch):
    client = runtime_env["client"]

    async def fake_probe(settings):
        return ProviderProbeResult(
            provider="moonshot",
            model="kimi-test",
            ok=False,
            status="request_failed",
            http_status=401,
            error_message="Provider returned HTTP 401.",
            error_body='{"error":"Invalid Authentication"}',
            attempts=[
                ProviderProbeAttempt(
                    provider="moonshot",
                    model="kimi-test",
                    ok=False,
                    status="request_failed",
                    http_status=401,
                    error_message="Provider returned HTTP 401.",
                )
            ],
        )

    monkeypatch.setattr("trainee.app.probe_provider", fake_probe)

    response = client.post("/api/runtime/provider/test")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert payload["provider"] == "moonshot"
    assert payload["http_status"] == 401
    assert "api_key" not in json.dumps(payload)


def test_agent_debug_setting_defaults_off_and_persists(runtime_env):
    client = runtime_env["client"]
    config_path = runtime_env["config_path"]

    assert client.get("/api/runtime/debug").json() == {"agent_debug_enabled": False}

    response = client.post("/api/runtime/debug", json={"agent_debug_enabled": True})

    assert response.status_code == 200
    assert response.json() == {"agent_debug_enabled": True}
    assert json.loads(config_path.read_text(encoding="utf-8"))["agent_debug_enabled"] is True
    assert client.get("/api/runtime/provider").json()["agent_debug_enabled"] is True

    ui_response = client.post(
        "/ui/runtime/debug",
        data={},
        headers={"HX-Request": "true"},
    )
    assert ui_response.status_code == 200
    assert client.get("/api/runtime/debug").json() == {"agent_debug_enabled": False}


def test_agent_debug_setting_cannot_change_while_loop_is_running(runtime_env, monkeypatch):
    client = runtime_env["client"]
    runtime = client.app.state.runtime
    monkeypatch.setattr(runtime, "loop_is_running", lambda: True)

    response = client.post("/api/runtime/debug", json={"agent_debug_enabled": True})

    assert response.status_code == 409
    assert "stop the loop" in response.json()["detail"]


def test_run_detail_renders_saved_agent_trace(runtime_env):
    client = runtime_env["client"]
    runtime = client.app.state.runtime
    session = runtime.storage.create_session(RunSession(status="stopped"))
    record = runtime.storage.create_round(
        RoundRecord(
            session_id=session.id,
            round_index=1,
            resolved_command="python train.py",
            status="completed",
            agent_trace=AgentTrace(
                provider="openai",
                model="gpt-test",
                status="parse_failed",
                raw_output="<unsafe>&raw",
                parse_error="no JSON object found",
                fallback_reason="no JSON object found",
            ),
        )
    )

    api_payload = client.get(f"/api/runs/{record.id}").json()
    detail = client.get(f"/fragments/run-detail?run_id={record.id}")

    assert api_payload["agent_trace"]["status"] == "parse_failed"
    assert detail.status_code == 200
    assert "Agent Debug" in detail.text
    assert "parse_failed" in detail.text
    assert "&lt;unsafe&gt;&amp;raw" in detail.text


def test_llm_image_probe_is_limited_per_session(runtime_env):
    client = runtime_env["client"]
    runtime = client.app.state.runtime
    session = runtime.storage.create_session(RunSession(status="running"))
    assert session.id is not None

    async def fake_probe(prompt: str, image=None):
        return {
            "provider": "fake",
            "model": "fake-vision",
            "has_image": image is not None,
            "content": prompt,
        }

    runtime.decision_engine.probe = fake_probe

    for expected_count in (1, 2, 3):
        response = client.post(
            "/api/llm/test",
            data={"prompt": "describe", "session_id": str(session.id)},
            files={"image": ("probe.png", b"fake-image", "image/png")},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["has_image"] is True
        assert payload["image_analysis"] == {
            "session_id": session.id,
            "used": expected_count,
            "limit": 3,
        }

    limited = client.post(
        "/api/llm/test",
        data={"prompt": "describe", "session_id": str(session.id)},
        files={"image": ("probe.png", b"fake-image", "image/png")},
    )
    assert limited.status_code == 429
    assert "image analysis limit reached" in limited.json()["detail"]

    text_only = client.post("/api/llm/test", data={"prompt": "ping", "session_id": str(session.id)})
    assert text_only.status_code == 200
    assert text_only.json()["has_image"] is False

    saved = runtime.storage.get_session(session.id)
    assert saved is not None
    assert saved.image_analysis_count == 3

    session.status = "stopped"
    runtime.storage.update_session(session)
    saved = runtime.storage.get_session(session.id)
    assert saved is not None
    assert saved.image_analysis_count == 3


def test_prompt_presets_are_project_scoped_and_apply_to_project(runtime_env):
    client = runtime_env["client"]
    external_project = runtime_env["external_project"]
    python = runtime_env["python"]
    register_payload = REGISTER_PAYLOAD_TEMPLATE | {
        "project_root": str(external_project),
        "working_dir": str(external_project),
        "launcher_template": f"{python} {{project_root}}/train.py --log-file {{project_root}}/logs/preset.log {{extra_args}}",
        "data_paths": [str(external_project / "data")],
        "log_paths": [str(external_project / "logs" / "*.log")],
    }
    assert client.post("/api/project/register", json=_registration(register_payload)).status_code == 200

    saved = client.post(
        "/api/prompt-presets",
        json={
            "project_root": str(external_project),
            "name": "conservative sweep",
            "metric_prompt": "Preset metric prompt.",
            "tuning_prompt": "Preset tuning prompt.",
        },
    )
    assert saved.status_code == 200
    preset = saved.json()

    listed = client.get("/api/prompt-presets", params={"project_root": str(external_project)})
    assert listed.status_code == 200
    assert [item["name"] for item in listed.json()["presets"]] == ["conservative sweep"]

    other_project = client.get("/api/prompt-presets", params={"project_root": str(external_project) + "-other"})
    assert other_project.status_code == 200
    assert other_project.json()["presets"] == []

    form = _project_form(register_payload | {"metric_prompt": "Unsaved metric.", "tuning_prompt": "Unsaved tuning."})
    form["prompt_preset_id"] = preset["id"]
    applied = client.post("/ui/prompt-presets/apply", data=form, headers={"HX-Request": "true"})

    assert applied.status_code == 200
    spec = client.get("/api/project").json()["spec"]
    assert spec["metric_prompt"] == "Preset metric prompt."
    assert spec["tuning_prompt"] == "Preset tuning prompt."
