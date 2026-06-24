from __future__ import annotations

import asyncio
import json as jsonlib
from pathlib import Path

import httpx

from trainee.decision import DecisionEngine
from trainee.models import MetricSpec, ProjectContext, ProjectSpec, RoundRecord, TunableParam
from trainee.settings import Settings, load_settings


def test_load_settings_defaults_data_dir_to_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("TRAINEE_DATA_DIR", raising=False)

    settings = load_settings(repo_root=tmp_path / "repo")

    assert settings.data_dir == tmp_path / ".trainee"
    assert settings.database_path == tmp_path / ".trainee" / "runtime.sqlite3"
    assert settings.artifacts_dir == tmp_path / ".trainee" / "artifacts"
    assert settings.config_path == tmp_path / ".trainee" / "config.json"


def test_load_settings_is_not_project_bound_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("TRAINEE_DATA_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "project").mkdir()
    monkeypatch.chdir(tmp_path / "project")

    settings = load_settings()

    assert settings.project_root is None
    assert settings.repo_root != tmp_path / "project"
    assert settings.data_dir == tmp_path / "home" / ".trainee"


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
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    settings = load_settings(repo_root=tmp_path)

    assert settings.config_path == tmp_path / ".trainee" / "config.json"
    assert settings.llm_provider == "none"
    assert settings.anthropic_api_key is None
    assert settings.anthropic_base_url == "https://api.anthropic.com"
    assert settings.anthropic_model == "claude-3-5-haiku-latest"
    assert settings.anthropic_max_tokens == 1024
    assert settings.llm_timeout_sec == 30.0


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
    assert settings.data_dir == tmp_path / "home" / ".trainee"
    assert settings.config_path == config_path
    assert settings.llm_provider == "openai"
    assert settings.openai_api_key == "config-openai-key"
    assert settings.openai_base_url == "https://openai.example/v1"
    assert settings.openai_model == "gpt-custom"
    assert settings.llm_timeout_sec == 22.0


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

    assert settings.llm_provider == "moonshot"
    assert settings.moonshot_api_key == "test-moonshot-key"
    assert settings.moonshot_base_url == "https://moonshot.example/v1"
    assert settings.moonshot_model == "kimi-custom"


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
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    config_path = tmp_path / "home" / ".trainee" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        jsonlib.dumps({"llm_provider": "openai", "openai": {"api_key": "config-key", "model": "config-model"}}),
        encoding="utf-8",
    )

    settings = load_settings(repo_root=tmp_path / "project")

    assert settings.openai_model == "env-model"
    assert settings.openai_api_key == "config-key"


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
        data_dir=fake_root / ".trainee",
        database_path=fake_root / ".trainee" / "runtime.sqlite3",
        artifacts_dir=fake_root / ".trainee" / "artifacts",
        template_dir=fake_root,
        static_dir=fake_root,
        config_path=fake_root / ".trainee" / "config.json",
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

    result = asyncio.run(engine.decide_with_prompt(spec, context, history, {"lr": 0.2}))
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
    assert "JSON only" in captured["json"]["messages"][0]["content"]


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
        data_dir=fake_root / ".trainee",
        database_path=fake_root / ".trainee" / "runtime.sqlite3",
        artifacts_dir=fake_root / ".trainee" / "artifacts",
        template_dir=fake_root,
        static_dir=fake_root,
        config_path=fake_root / ".trainee" / "config.json",
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

    result = asyncio.run(engine.decide_with_prompt(spec, context, history, {"lr": 0.2}))
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
