from __future__ import annotations

import json

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


def _project_form(payload: dict[str, object]) -> dict[str, object]:
    return {
        "project_root": payload["project_root"],
        "working_dir": payload["working_dir"],
        "launcher_template": payload["launcher_template"],
        "security_mode": payload.get("security_mode", "guarded"),
        "data_paths_json": json.dumps(payload.get("data_paths", [])),
        "log_paths_json": json.dumps(payload.get("log_paths", [])),
        "wandb_enabled": "on" if payload.get("wandb_enabled") else "",
        "heartbeat_interval_sec": str(payload.get("heartbeat_interval_sec", 5.0)),
        "stall_timeout_sec": str(payload.get("stall_timeout_sec", 120.0)),
        "max_rounds": str(payload.get("max_rounds", 3)),
        "tunable_params_json": json.dumps(payload.get("tunable_params", [])),
        "metric_specs_json": json.dumps(payload.get("metric_specs", [])),
        "metric_prompt": payload.get("metric_prompt", ""),
        "tuning_prompt": payload.get("tuning_prompt", ""),
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

    assert client.post("/api/project/register", json=register_payload).status_code == 200

    response = client.get("/api/prompt-preview")

    assert response.status_code == 200
    payload = response.json()
    preview = payload["prompt_preview"]
    assert payload["label"] == "Next decision preview based on saved project state"
    assert preview["status"] == "preview"
    assert preview["provider"] == "none"
    assert "JSON only" in preview["system_prompt"]
    assert '"tuning_prompt": "Lower lr if loss gets worse."' in preview["user_prompt"]
    assert preview["payload"]["user"] == preview["user_prompt"]


def test_runtime_provider_settings_save_to_config_json(runtime_env):
    client = runtime_env["client"]
    data_dir = runtime_env["data_dir"]

    response = client.post(
        "/ui/runtime/provider",
        data={
            "llm_provider": "openai",
            "llm_timeout_sec": "12",
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
    config = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
    assert config["llm_provider"] == "openai"
    assert config["llm_timeout_sec"] == 12.0
    assert config["openai"]["api_key"] == "test-openai-key"
    assert config["openai"]["base_url"] == "https://openai.example/v1"
    assert config["openai"]["model"] == "gpt-ui"
    assert config["moonshot"]["api_key"] == "test-moonshot-key"
    assert config["moonshot"]["base_url"] == "https://moonshot.example/v1"
    assert config["moonshot"]["model"] == "kimi-ui"
    assert client.get("/api/health").json()["llm_provider"] == "openai"


def test_runtime_provider_settings_api_manages_config_without_exposing_keys(runtime_env):
    client = runtime_env["client"]
    data_dir = runtime_env["data_dir"]

    response = client.post(
        "/api/runtime/provider",
        json={
            "llm_provider": "moonshot",
            "llm_timeout_sec": 9,
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
    assert payload["active_model"] == "kimi-api"
    assert payload["moonshot_key_configured"] is True
    assert "api_key" not in json.dumps(payload)

    config = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
    assert config["llm_provider"] == "moonshot"
    assert config["moonshot"]["api_key"] == "moonshot-secret"
    assert client.get("/api/health").json()["llm_provider"] == "moonshot"


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
    assert client.post("/api/project/register", json=register_payload).status_code == 200

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
