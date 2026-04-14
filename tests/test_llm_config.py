from __future__ import annotations

import asyncio
import json as jsonlib
from pathlib import Path

import httpx

from trainee.decision import DecisionEngine
from trainee.models import MetricSpec, ProjectContext, ProjectSpec, RoundRecord, TunableParam
from trainee.settings import Settings, load_settings


def test_load_settings_reads_dotenv_for_anthropic(tmp_path, monkeypatch):
    for key in (
        "TRAINEE_LLM_PROVIDER",
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_VERSION",
        "ANTHROPIC_MAX_TOKENS",
        "TRAINEE_LLM_TIMEOUT_SEC",
    ):
        monkeypatch.delenv(key, raising=False)

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

    assert settings.dotenv_path == tmp_path / ".env"
    assert settings.llm_provider == "anthropic"
    assert settings.anthropic_api_key == "test-anthropic-key"
    assert settings.anthropic_base_url == "https://anthropic.example"
    assert settings.anthropic_model == "claude-custom"
    assert settings.anthropic_max_tokens == 2048
    assert settings.llm_timeout_sec == 45.0


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
        data_dir=fake_root / ".trainee",
        database_path=fake_root / ".trainee" / "runtime.sqlite3",
        artifacts_dir=fake_root / ".trainee" / "artifacts",
        template_dir=fake_root,
        static_dir=fake_root,
        dotenv_path=fake_root / ".env",
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

    decision = asyncio.run(engine.decide(spec, context, history, {"lr": 0.2}))

    assert decision.action == "continue"
    assert decision.next_params["lr"] == 0.15
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "anthropic-key"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["json"]["model"] == "claude-3-5-sonnet-latest"
    assert captured["json"]["max_tokens"] == 512
    assert captured["json"]["messages"][0]["role"] == "user"
    assert "JSON only" in captured["json"]["system"]
