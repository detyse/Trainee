from __future__ import annotations

import asyncio
import json as jsonlib
from pathlib import Path

import httpx

from trainee.decision import DecisionEngine
from trainee.models import MetricSpec, ProjectContext, ProjectSpec, RoundRecord, TunableParam
from trainee.prompt_assembler import PromptAssembler
from trainee.providers import provider_settings_payload
from trainee.research_state import ResearchStateBuilder
from trainee.settings import DEFAULT_LLM_TEMPERATURE, Settings, load_default_system_prompt, load_settings


def test_load_settings_defaults_runtime_data_to_home_and_config_to_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("TRAINEE_DATA_DIR", raising=False)
    monkeypatch.delenv("TRAINEE_PROJECT_ROOT", raising=False)

    settings = load_settings(repo_root=tmp_path / "repo")

    assert settings.project_root is None
    assert settings.project_data_dir == tmp_path / ".trainee" / "runtime"
    assert settings.data_dir == settings.project_data_dir
    assert settings.database_path == tmp_path / ".trainee" / "runtime" / "runtime.sqlite3"
    assert settings.artifacts_dir == tmp_path / ".trainee" / "runtime" / "artifacts"
    assert settings.global_config_path == tmp_path / ".trainee" / "config.json"
    assert settings.config_path == settings.global_config_path
    assert settings.agent_debug_enabled is False
    assert settings.llm_temperature == DEFAULT_LLM_TEMPERATURE == 1.0
    assert settings.system_prompt == load_default_system_prompt()
    saved = jsonlib.loads(settings.global_config_path.read_text(encoding="utf-8"))
    assert saved["system_prompt"] == settings.system_prompt


def test_load_settings_adds_default_system_prompt_without_overwriting_config(tmp_path):
    config_path = tmp_path / "home" / ".trainee" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        jsonlib.dumps({"llm_provider": "none", "custom_key": {"keep": True}}),
        encoding="utf-8",
    )

    settings = load_settings(repo_root=tmp_path / "repo", global_config_path=config_path)

    saved = jsonlib.loads(config_path.read_text(encoding="utf-8"))
    assert settings.system_prompt == load_default_system_prompt()
    assert saved["system_prompt"] == settings.system_prompt
    assert saved["llm_provider"] == "none"
    assert saved["custom_key"] == {"keep": True}


def test_load_settings_rejects_invalid_system_prompt(tmp_path):
    config_path = tmp_path / "home" / ".trainee" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(jsonlib.dumps({"system_prompt": "   "}), encoding="utf-8")

    try:
        load_settings(repo_root=tmp_path / "repo", global_config_path=config_path)
    except ValueError as exc:
        assert "system_prompt cannot be blank" in str(exc)
    else:
        raise AssertionError("blank system_prompt should be rejected")


def test_load_settings_uses_project_root_for_runtime_data(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("TRAINEE_DATA_DIR", raising=False)
    monkeypatch.delenv("TRAINEE_PROJECT_ROOT", raising=False)
    project_root = tmp_path / "project"

    settings = load_settings(repo_root=tmp_path / "repo", project_root=project_root)

    assert settings.project_root == project_root
    assert settings.project_data_dir == project_root / ".trainee"
    assert settings.database_path == project_root / ".trainee" / "runtime.sqlite3"
    assert settings.artifacts_dir == project_root / ".trainee" / "artifacts"
    assert settings.global_config_path == tmp_path / "home" / ".trainee" / "config.json"


def test_load_settings_uses_env_project_root_for_runtime_data(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("TRAINEE_DATA_DIR", raising=False)
    project_root = tmp_path / "project"
    monkeypatch.setenv("TRAINEE_PROJECT_ROOT", str(project_root))

    settings = load_settings(repo_root=tmp_path / "repo")

    assert settings.project_root == project_root
    assert settings.project_data_dir == project_root / ".trainee"
    assert settings.database_path == project_root / ".trainee" / "runtime.sqlite3"
    assert settings.artifacts_dir == project_root / ".trainee" / "artifacts"
    assert settings.global_config_path == tmp_path / "home" / ".trainee" / "config.json"


def test_load_settings_is_not_project_bound_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("TRAINEE_DATA_DIR", raising=False)
    monkeypatch.delenv("TRAINEE_PROJECT_ROOT", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "project").mkdir()
    monkeypatch.chdir(tmp_path / "project")

    settings = load_settings()

    assert settings.project_root is None
    assert settings.repo_root != tmp_path / "project"
    assert settings.global_config_path == tmp_path / "home" / ".trainee" / "config.json"
    assert settings.data_dir == tmp_path / "home" / ".trainee" / "runtime"


def test_load_settings_ignores_dotenv_file(tmp_path, monkeypatch):
    for key in (
        "TRAINEE_DATA_DIR",
        "TRAINEE_LLM_PROVIDER",
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "MOONSHOT_API_KEY",
        "MOONSHOT_BASE_URL",
        "MOONSHOT_MODEL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_VERSION",
        "ANTHROPIC_MAX_TOKENS",
        "TRAINEE_LLM_TIMEOUT_SEC",
        "TRAINEE_LLM_TEMPERATURE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "TRAINEE_LLM_PROVIDER=anthropic",
                "ANTHROPIC_API_KEY=test-anthropic-key",
                "ANTHROPIC_BASE_URL=https://anthropic.example",
                "ANTHROPIC_MODEL=claude-custom",
                "ANTHROPIC_VERSION=2023-06-01",
                "ANTHROPIC_MAX_TOKENS=2048",
                "TRAINEE_LLM_TIMEOUT_SEC=45",
                "TRAINEE_LLM_TEMPERATURE=0.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    settings = load_settings(repo_root=tmp_path)

    assert settings.config_path == tmp_path / ".trainee" / "config.json"
    assert settings.global_config_path == tmp_path / ".trainee" / "config.json"
    assert settings.llm_provider == "none"
    assert settings.llm_provider_selection == "auto"
    assert settings.anthropic_api_key is None
    assert settings.anthropic_base_url == "https://api.anthropic.com"
    assert settings.anthropic_model == "claude-3-5-haiku-latest"
    assert settings.anthropic_max_tokens == 1024
    assert settings.llm_timeout_sec == 600.0
    assert settings.llm_temperature == DEFAULT_LLM_TEMPERATURE


def test_load_settings_reads_home_config_for_provider(tmp_path, monkeypatch):
    for key in (
        "TRAINEE_DATA_DIR",
        "TRAINEE_LLM_PROVIDER",
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "MOONSHOT_API_KEY",
        "MOONSHOT_BASE_URL",
        "MOONSHOT_MODEL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_VERSION",
        "ANTHROPIC_MAX_TOKENS",
        "TRAINEE_LLM_TIMEOUT_SEC",
        "TRAINEE_LLM_TEMPERATURE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config_path = tmp_path / "home" / ".trainee" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        jsonlib.dumps(
            {
                "llm_provider": "openai",
                "llm_timeout_sec": 22,
                "llm_temperature": 0.6,
                "openai": {
                    "api_key": "config-openai-key",
                    "base_url": "https://openai.example/v1",
                    "model": "gpt-custom",
                },
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(repo_root=tmp_path / "project")

    assert settings.repo_root == tmp_path / "project"
    assert settings.project_root is None
    assert settings.data_dir == tmp_path / "home" / ".trainee" / "runtime"
    assert settings.config_path == config_path
    assert settings.llm_provider_selection == "openai"
    assert settings.llm_provider == "openai"
    assert settings.openai_api_key == "config-openai-key"
    assert settings.openai_base_url == "https://openai.example/v1"
    assert settings.openai_model == "gpt-custom"
    assert settings.llm_timeout_sec == 22.0
    assert settings.llm_temperature == 0.6
    assert settings.agent_debug_enabled is False


def test_trainee_data_dir_overrides_runtime_only(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TRAINEE_DATA_DIR", "custom-runtime")

    settings = load_settings(repo_root=tmp_path / "repo", project_root=tmp_path / "project")

    assert settings.project_data_dir == tmp_path / "repo" / "custom-runtime"
    assert settings.database_path == tmp_path / "repo" / "custom-runtime" / "runtime.sqlite3"
    assert settings.artifacts_dir == tmp_path / "repo" / "custom-runtime" / "artifacts"
    assert settings.global_config_path == tmp_path / "home" / ".trainee" / "config.json"


def test_load_settings_reads_moonshot_provider(tmp_path, monkeypatch):
    for key in (
        "TRAINEE_DATA_DIR",
        "TRAINEE_LLM_PROVIDER",
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "MOONSHOT_API_KEY",
        "MOONSHOT_BASE_URL",
        "MOONSHOT_MODEL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_VERSION",
        "ANTHROPIC_MAX_TOKENS",
        "TRAINEE_LLM_TIMEOUT_SEC",
        "TRAINEE_LLM_TEMPERATURE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config_path = tmp_path / "home" / ".trainee" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        jsonlib.dumps(
            {
                "moonshot": {
                    "api_key": "test-moonshot-key",
                    "base_url": "https://moonshot.example/v1",
                    "model": "kimi-custom",
                },
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(repo_root=tmp_path)

    assert settings.llm_provider_selection == "auto"
    assert settings.llm_provider == "moonshot"
    assert settings.moonshot_api_key == "test-moonshot-key"
    assert settings.moonshot_base_url == "https://moonshot.example/v1"
    assert settings.moonshot_model == "kimi-custom"


def test_load_settings_treats_legacy_none_with_keys_as_auto(tmp_path, monkeypatch):
    for key in (
        "TRAINEE_DATA_DIR",
        "TRAINEE_LLM_PROVIDER",
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "MOONSHOT_API_KEY",
        "MOONSHOT_BASE_URL",
        "MOONSHOT_MODEL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_VERSION",
        "ANTHROPIC_MAX_TOKENS",
        "TRAINEE_LLM_TIMEOUT_SEC",
        "TRAINEE_LLM_TEMPERATURE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config_path = tmp_path / "home" / ".trainee" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        jsonlib.dumps(
            {
                "llm_provider": "none",
                "openai": {"api_key": "test-openai-key"},
                "moonshot": {"api_key": "test-moonshot-key"},
                "anthropic": {"api_key": "test-anthropic-key"},
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(repo_root=tmp_path)

    assert settings.llm_provider_selection == "auto"
    assert settings.llm_provider == "moonshot"
    assert settings.moonshot_api_key == "test-moonshot-key"


def test_load_settings_keeps_schema_v2_none_disabled_with_keys(tmp_path, monkeypatch):
    for key in (
        "TRAINEE_DATA_DIR",
        "TRAINEE_LLM_PROVIDER",
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "MOONSHOT_API_KEY",
        "MOONSHOT_BASE_URL",
        "MOONSHOT_MODEL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_VERSION",
        "ANTHROPIC_MAX_TOKENS",
        "TRAINEE_LLM_TIMEOUT_SEC",
        "TRAINEE_LLM_TEMPERATURE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config_path = tmp_path / "home" / ".trainee" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        jsonlib.dumps(
            {
                "llm_provider_schema": 2,
                "llm_provider": "none",
                "moonshot": {"api_key": "test-moonshot-key"},
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(repo_root=tmp_path)

    assert settings.llm_provider_selection == "none"
    assert settings.llm_provider == "none"
    assert settings.moonshot_api_key == "test-moonshot-key"


def test_environment_overrides_home_config(tmp_path, monkeypatch):
    for key in (
        "TRAINEE_DATA_DIR",
        "TRAINEE_LLM_PROVIDER",
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "MOONSHOT_API_KEY",
        "MOONSHOT_BASE_URL",
        "MOONSHOT_MODEL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_VERSION",
        "ANTHROPIC_MAX_TOKENS",
        "TRAINEE_LLM_TIMEOUT_SEC",
        "TRAINEE_LLM_TEMPERATURE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    monkeypatch.setenv("TRAINEE_LLM_TEMPERATURE", "0.4")
    config_path = tmp_path / "home" / ".trainee" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        jsonlib.dumps(
            {
                "llm_provider": "openai",
                "llm_temperature": 0.8,
                "openai": {"api_key": "config-key", "model": "config-model"},
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(repo_root=tmp_path / "project")

    assert settings.openai_model == "env-model"
    assert settings.openai_api_key == "config-key"
    assert settings.llm_temperature == 0.4


def test_provider_payload_reports_environment_overrides(tmp_path, monkeypatch):
    for key in (
        "TRAINEE_DATA_DIR",
        "TRAINEE_LLM_PROVIDER",
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "MOONSHOT_API_KEY",
        "MOONSHOT_BASE_URL",
        "MOONSHOT_MODEL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_VERSION",
        "ANTHROPIC_MAX_TOKENS",
        "TRAINEE_LLM_TIMEOUT_SEC",
        "TRAINEE_LLM_TEMPERATURE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TRAINEE_LLM_PROVIDER", "none")
    config_path = tmp_path / "home" / ".trainee" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        jsonlib.dumps(
            {
                "llm_provider": "openai",
                "openai": {"api_key": "config-openai-key", "model": "gpt-config"},
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(repo_root=tmp_path / "project")
    payload = provider_settings_payload(settings)

    assert settings.llm_provider_selection == "none"
    assert settings.llm_provider == "none"
    assert settings.environment_overrides == ("TRAINEE_LLM_PROVIDER",)
    assert payload["llm_provider_selection"] == "none"
    assert payload["llm_provider"] == "none"
    assert payload["llm_provider_source"] == "environment"
    assert payload["global_config_path"] == str(config_path)
    assert payload["environment_overrides"] == ["TRAINEE_LLM_PROVIDER"]
    assert "TRAINEE_LLM_PROVIDER" in payload["environment_override_warning"]


def test_llm_provider_env_overrides_config_provider(tmp_path, monkeypatch):
    for key in (
        "TRAINEE_DATA_DIR",
        "TRAINEE_LLM_PROVIDER",
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "MOONSHOT_API_KEY",
        "MOONSHOT_BASE_URL",
        "MOONSHOT_MODEL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_VERSION",
        "ANTHROPIC_MAX_TOKENS",
        "TRAINEE_LLM_TIMEOUT_SEC",
        "TRAINEE_LLM_TEMPERATURE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    config_path = tmp_path / "home" / ".trainee" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        jsonlib.dumps(
            {
                "llm_provider": "openai",
                "openai": {"api_key": "config-openai-key"},
                "anthropic": {"api_key": "config-anthropic-key"},
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(repo_root=tmp_path / "project")

    assert settings.llm_provider == "anthropic"
    assert settings.anthropic_api_key == "config-anthropic-key"
    assert settings.environment_overrides == ("LLM_PROVIDER",)


def test_decision_engine_uses_moonshot_chat_completions(monkeypatch):
    captured: dict[str, object] = {}
    fake_root = Path("/tmp/trainee-tests")

    class FakeAsyncClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json, headers):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            request = httpx.Request("POST", url)
            return httpx.Response(
                200,
                request=request,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": jsonlib.dumps(
                                    {
                                        "action": "continue",
                                        "next_params": {"lr": 0.12},
                                        "reason": "Lower lr after reviewing total_loss.",
                                        "focus_metrics": ["total_loss"],
                                    }
                                )
                            }
                        }
                    ]
                },
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    settings = Settings(
        repo_root=fake_root,
        project_root=None,
        project_data_dir=fake_root / ".trainee",
        database_path=fake_root / ".trainee" / "runtime.sqlite3",
        artifacts_dir=fake_root / ".trainee" / "artifacts",
        template_dir=fake_root,
        static_dir=fake_root,
        global_config_path=fake_root / "home" / ".trainee" / "config.json",
        llm_provider="moonshot",
        llm_timeout_sec=12.0,
        openai_api_key=None,
        openai_base_url="https://api.openai.com/v1",
        openai_model="gpt-4o-mini",
        anthropic_api_key=None,
        anthropic_base_url="https://api.anthropic.com",
        anthropic_model="claude-3-5-sonnet-latest",
        anthropic_version="2023-06-01",
        anthropic_max_tokens=512,
        system_prompt="exact configured system prompt",
        moonshot_api_key="moonshot-key",
        moonshot_base_url="https://api.moonshot.cn/v1",
        moonshot_model="kimi-k2.6",
    )
    engine = DecisionEngine(settings)

    spec = ProjectSpec(
        project_root="/tmp/project",
        working_dir="/tmp/project",
        launcher_template="python train.py {extra_args}",
        tunable_params=[
            TunableParam(name="lr", flag="--lr", type="float", default=0.2, min_value=0.05, max_value=0.4),
        ],
        metric_specs=[
            MetricSpec(
                name="total_loss",
                source="log_regex",
                key_or_pattern=r"total_loss=(?P<value>-?\d+(?:\.\d+)?)",
                goal="min",
                required=True,
            )
        ],
        tuning_prompt="Try smaller lr when loss gets worse.",
    )
    context = ProjectContext(project_summary="Fake project")
    history = [
        RoundRecord(
            session_id=1,
            round_index=1,
            resolved_command="python train.py --lr 0.2",
            param_values={"lr": 0.2},
            status="completed",
            metrics={"total_loss": 0.55},
            exit_code=0,
        )
    ]

    research_state = ResearchStateBuilder().build(spec, history)
    result = asyncio.run(
        engine.decide_with_prompt(
            spec=spec,
            context=context,
            research_state=research_state,
            current_params={"lr": 0.2},
            prompt_documents=[],
        )
    )
    decision = result.decision

    assert decision.action == "continue"
    assert decision.next_params["lr"] == 0.12
    assert result.prompt_preview is not None
    assert result.prompt_preview.status == "sent"
    assert result.prompt_preview.provider == "moonshot"
    assert result.prompt_preview.payload == captured["json"]
    assert captured["url"] == "https://api.moonshot.cn/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer moonshot-key"
    assert captured["json"]["model"] == "kimi-k2.6"
    assert captured["json"]["messages"][0]["role"] == "system"
    assert captured["json"]["messages"][0]["content"] == "exact configured system prompt"
    user_prompt = captured["json"]["messages"][1]["content"]
    assert user_prompt.startswith("<STATIC_CONTEXT>\n")
    assert "\n</STATIC_CONTEXT>\n\n<DYNAMIC_ROUND_STATE>\n" in user_prompt
    assert '"tuning_prompt":"Try smaller lr when loss gets worse."' in user_prompt

    assembler = PromptAssembler()
    prompt_a = assembler.assemble(
        spec,
        context,
        research_state,
        {"lr": 0.2},
        [],
        "exact configured system prompt",
    ).user_prompt
    prompt_b = assembler.assemble(
        spec,
        context,
        research_state,
        {"lr": 0.12},
        [],
        "exact configured system prompt",
    ).user_prompt
    static_a = prompt_a.split("\n</STATIC_CONTEXT>", 1)[0]
    static_b = prompt_b.split("\n</STATIC_CONTEXT>", 1)[0]
    assert static_a == static_b
    assert '"current_params":{"lr":0.2}' in prompt_a
    assert '"current_params":{"lr":0.12}' in prompt_b


def test_decision_engine_uses_anthropic_messages_api(monkeypatch):
    captured: dict[str, object] = {}
    fake_root = Path("/tmp/trainee-tests")

    class FakeAsyncClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json, headers):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            request = httpx.Request("POST", url)
            return httpx.Response(
                200,
                request=request,
                json={
                    "content": [
                        {
                            "type": "text",
                            "text": jsonlib.dumps(
                                {
                                    "action": "continue",
                                    "next_params": {"lr": 0.15},
                                    "reason": "Lower lr after reviewing total_loss.",
                                    "focus_metrics": ["total_loss"],
                                }
                            ),
                        }
                    ]
                },
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    settings = Settings(
        repo_root=fake_root,
        project_root=None,
        project_data_dir=fake_root / ".trainee",
        database_path=fake_root / ".trainee" / "runtime.sqlite3",
        artifacts_dir=fake_root / ".trainee" / "artifacts",
        template_dir=fake_root,
        static_dir=fake_root,
        global_config_path=fake_root / "home" / ".trainee" / "config.json",
        llm_provider="anthropic",
        llm_timeout_sec=12.0,
        openai_api_key=None,
        openai_base_url="https://api.openai.com/v1",
        openai_model="gpt-4o-mini",
        anthropic_api_key="anthropic-key",
        anthropic_base_url="https://api.anthropic.com",
        anthropic_model="claude-3-5-sonnet-latest",
        anthropic_version="2023-06-01",
        anthropic_max_tokens=512,
    )
    engine = DecisionEngine(settings)

    spec = ProjectSpec(
        project_root="/tmp/project",
        working_dir="/tmp/project",
        launcher_template="python train.py {extra_args}",
        tunable_params=[
            TunableParam(name="lr", flag="--lr", type="float", default=0.2, min_value=0.05, max_value=0.4),
        ],
        metric_specs=[
            MetricSpec(
                name="total_loss",
                source="log_regex",
                key_or_pattern=r"total_loss=(?P<value>-?\d+(?:\.\d+)?)",
                goal="min",
                required=True,
            )
        ],
        tuning_prompt="Try smaller lr when loss gets worse.",
    )
    context = ProjectContext(project_summary="Fake project")
    history = [
        RoundRecord(
            session_id=1,
            round_index=1,
            resolved_command="python train.py --lr 0.2",
            param_values={"lr": 0.2},
            status="completed",
            metrics={"total_loss": 0.55},
            exit_code=0,
        )
    ]

    result = asyncio.run(
        engine.decide_with_prompt(
            spec=spec,
            context=context,
            research_state=ResearchStateBuilder().build(spec, history),
            current_params={"lr": 0.2},
            prompt_documents=[],
        )
    )
    decision = result.decision

    assert decision.action == "continue"
    assert decision.next_params["lr"] == 0.15
    assert result.prompt_preview is not None
    assert result.prompt_preview.status == "sent"
    assert result.prompt_preview.payload == captured["json"]
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "anthropic-key"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["json"]["model"] == "claude-3-5-sonnet-latest"
    assert captured["json"]["max_tokens"] == 512
    assert captured["json"]["messages"][0]["role"] == "user"
    assert "JSON only" in captured["json"]["system"]
