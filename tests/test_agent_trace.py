from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from trainee.decision import DecisionEngine
from trainee.llm import LLMClient
from trainee.models import AgentTrace, MetricSpec, ProjectContext, ProjectSpec, RoundRecord, RunSession, TunableParam
from trainee.research_state import ResearchStateBuilder
from trainee.settings import DEFAULT_LLM_TEMPERATURE, Settings
from trainee.storage import Storage


def _settings(tmp_path: Path, provider: str, *, debug: bool = True) -> Settings:
    return Settings(
        repo_root=tmp_path,
        project_root=None,
        project_data_dir=tmp_path / ".trainee",
        database_path=tmp_path / ".trainee" / "runtime.sqlite3",
        artifacts_dir=tmp_path / ".trainee" / "artifacts",
        template_dir=tmp_path,
        static_dir=tmp_path,
        global_config_path=tmp_path / "home" / ".trainee" / "config.json",
        llm_provider=provider,  # type: ignore[arg-type]
        llm_timeout_sec=5.0,
        openai_api_key="openai-key",
        openai_base_url="https://openai.example/v1",
        openai_model="gpt-test",
        moonshot_api_key="moonshot-key",
        moonshot_base_url="https://moonshot.example/v1",
        moonshot_model="kimi-test",
        anthropic_api_key="anthropic-key",
        anthropic_base_url="https://anthropic.example",
        anthropic_model="claude-test",
        anthropic_version="2023-06-01",
        anthropic_max_tokens=512,
        agent_debug_enabled=debug,
    )


def _decision_inputs(tmp_path: Path) -> tuple[ProjectSpec, ProjectContext, list[RoundRecord]]:
    spec = ProjectSpec(
        project_root=str(tmp_path / "project"),
        working_dir=str(tmp_path / "project"),
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
    )
    context = ProjectContext(project_summary="Trace test")
    history = [
        RoundRecord(
            session_id=1,
            round_index=1,
            resolved_command="python train.py --lr 0.2",
            param_values={"lr": 0.2},
            status="completed",
            metrics={"total_loss": 1.0},
            exit_code=0,
        )
    ]
    return spec, context, history


def _run_decision(engine: DecisionEngine, spec: ProjectSpec, context: ProjectContext, history: list[RoundRecord]):
    return asyncio.run(
        engine.decide_with_prompt(
            spec=spec,
            context=context,
            research_state=ResearchStateBuilder().build(spec, history),
            current_params={"lr": 0.2},
            prompt_documents=[],
        )
    )


@pytest.mark.parametrize("provider", ["openai", "moonshot", "anthropic"])
def test_provider_payload_uses_configured_temperature(tmp_path: Path, provider: str) -> None:
    settings = replace(_settings(tmp_path, provider), llm_temperature=0.7)
    payload = LLMClient(settings).build_payload(provider, "system", "user")

    assert payload["temperature"] == settings.llm_temperature == 0.7
    assert _settings(tmp_path, provider).llm_temperature == DEFAULT_LLM_TEMPERATURE == 1.0


@pytest.mark.parametrize(
    ("provider", "response_body", "expected_model", "expected_usage", "expected_finish"),
    [
        (
            "openai",
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "action": "continue",
                                    "next_params": {"lr": 0.15},
                                    "reason": "test",
                                }
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
            "gpt-test",
            {"prompt_tokens": 10, "completion_tokens": 5},
            "stop",
        ),
        (
            "moonshot",
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "action": "continue",
                                    "next_params": {"lr": 0.15},
                                    "reason": "test",
                                }
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 15},
            },
            "kimi-test",
            {"total_tokens": 15},
            "stop",
        ),
        (
            "anthropic",
            {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "action": "continue",
                                "next_params": {"lr": 0.15},
                                "reason": "test",
                            }
                        ),
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "stop_reason": "end_turn",
            },
            "claude-test",
            {"input_tokens": 10, "output_tokens": 5},
            "end_turn",
        ),
    ],
)
def test_agent_trace_captures_successful_provider_response(
    tmp_path: Path,
    monkeypatch,
    provider: str,
    response_body: dict[str, Any],
    expected_model: str,
    expected_usage: dict[str, Any],
    expected_finish: str,
) -> None:
    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json, headers):
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                headers={"x-request-id": "req-trace-123"},
                json=response_body,
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    spec, context, history = _decision_inputs(tmp_path)

    result = _run_decision(DecisionEngine(_settings(tmp_path, provider)), spec, context, history)

    assert result.decision.next_params["lr"] == 0.15
    assert result.agent_trace is not None
    assert result.agent_trace.provider == provider
    assert result.agent_trace.model == expected_model
    assert result.agent_trace.status == "success"
    assert result.agent_trace.http_status == 200
    assert result.agent_trace.request_id == "req-trace-123"
    assert result.agent_trace.raw_response_body
    assert result.agent_trace.raw_output
    assert result.agent_trace.extracted_json["next_params"] == {"lr": 0.15}
    assert result.agent_trace.usage == expected_usage
    assert result.agent_trace.finish_reason == expected_finish


def test_agent_trace_is_not_created_when_debug_is_disabled(tmp_path: Path, monkeypatch) -> None:
    response_content = {
        "value": '{"action":"continue","next_params":{"lr":0.15},"reason":"test"}'
    }

    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json, headers):
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={"choices": [{"message": {"content": response_content["value"]}}]},
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    spec, context, history = _decision_inputs(tmp_path)
    settings = replace(_settings(tmp_path, "openai"), agent_debug_enabled=False)

    result = _run_decision(DecisionEngine(settings), spec, context, history)

    assert result.agent_trace is None

    response_content["value"] = "not json"
    fallback_result = _run_decision(DecisionEngine(settings), spec, context, history)
    assert fallback_result.agent_trace is not None
    assert fallback_result.agent_trace.status == "parse_failed"
    assert fallback_result.prompt_preview is not None
    assert fallback_result.prompt_preview.status == "parse_failed"
    assert fallback_result.decision.action == "stop"


@pytest.mark.parametrize(
    ("content", "expected_status", "error_field"),
    [
        ("not json", "parse_failed", "parse_error"),
        ('{"action":"invalid","next_params":{},"reason":"bad"}', "validation_failed", "validation_error"),
        (
            '{"action":"continue","next_params":{"unknown":"bad"},"reason":"bad"}',
            "validation_failed",
            "validation_error",
        ),
    ],
)
def test_agent_trace_records_parse_and_validation_fallbacks(
    tmp_path: Path,
    monkeypatch,
    content: str,
    expected_status: str,
    error_field: str,
) -> None:
    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json, headers):
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={"choices": [{"message": {"content": content}}]},
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    spec, context, history = _decision_inputs(tmp_path)

    result = _run_decision(DecisionEngine(_settings(tmp_path, "openai")), spec, context, history)

    assert result.agent_trace is not None
    assert result.agent_trace.status == expected_status
    assert getattr(result.agent_trace, error_field)
    assert result.agent_trace.raw_output == content
    assert result.agent_trace.fallback_reason
    assert "unknown" not in result.decision.next_params


def test_agent_trace_records_http_error_body(tmp_path: Path, monkeypatch) -> None:
    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json, headers):
            return httpx.Response(
                429,
                request=httpx.Request("POST", url),
                headers={"x-request-id": "req-rate-limit"},
                json={"error": {"message": "rate limited"}},
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    spec, context, history = _decision_inputs(tmp_path)

    result = _run_decision(DecisionEngine(_settings(tmp_path, "openai")), spec, context, history)

    assert result.agent_trace is not None
    assert result.agent_trace.status == "request_failed"
    assert result.agent_trace.http_status == 429
    assert result.agent_trace.request_id == "req-rate-limit"
    assert "rate limited" in (result.agent_trace.error_body or "")
    assert result.agent_trace.provider_error is not None
    assert "Provider returned HTTP 429" in result.agent_trace.provider_error
    assert "https://anthropic.example/v1/messages" in result.agent_trace.provider_error
    assert "request_id=req-rate-limit" in result.agent_trace.provider_error
    assert "rate limited" in result.agent_trace.provider_error
    assert "https://openai.example/v1/chat/completions" in result.agent_trace.attempts[0]["error_message"]


def test_decision_uses_next_provider_when_primary_request_fails(tmp_path: Path, monkeypatch) -> None:
    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json, headers):
            if "moonshot" in url:
                return httpx.Response(401, request=httpx.Request("POST", url), json={"error": "bad key"})
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={
                    "choices": [
                        {
                            "message": {
                                "content": '{"action":"continue","next_params":{"lr":0.12},"reason":"openai ok"}'
                            },
                            "finish_reason": "stop",
                        }
                    ]
                },
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    spec, context, history = _decision_inputs(tmp_path)

    result = _run_decision(DecisionEngine(_settings(tmp_path, "moonshot")), spec, context, history)

    assert result.decision.next_params["lr"] == 0.12
    assert result.agent_trace is not None
    assert result.agent_trace.provider == "openai"
    assert [attempt["provider"] for attempt in result.agent_trace.attempts] == ["moonshot", "openai"]
    assert result.agent_trace.attempts[0]["status"] == "request_failed"


def test_agent_trace_records_transport_and_not_called_reasons(tmp_path: Path, monkeypatch) -> None:
    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json, headers):
            request = httpx.Request("POST", url)
            raise httpx.ReadTimeout("provider timed out", request=request)

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    spec, context, history = _decision_inputs(tmp_path)

    timeout_result = _run_decision(DecisionEngine(_settings(tmp_path, "openai")), spec, context, history)
    disabled_result = _run_decision(DecisionEngine(_settings(tmp_path, "none")), spec, context, history)
    missing_key_settings = replace(
        _settings(tmp_path, "openai"),
        openai_api_key=None,
        moonshot_api_key=None,
        anthropic_api_key=None,
    )
    missing_key_result = _run_decision(DecisionEngine(missing_key_settings), spec, context, history)

    assert timeout_result.agent_trace is not None
    assert timeout_result.agent_trace.status == "request_failed"
    assert timeout_result.agent_trace.provider_error is not None
    assert "ReadTimeout after 5s" in timeout_result.agent_trace.provider_error
    assert "https://anthropic.example/v1/messages" in timeout_result.agent_trace.provider_error
    assert "https://openai.example/v1/chat/completions" in timeout_result.agent_trace.attempts[0]["error_message"]
    assert disabled_result.agent_trace is not None
    assert disabled_result.agent_trace.status == "not_called"
    assert "No configured LLM provider" in (disabled_result.agent_trace.fallback_reason or "")
    assert missing_key_result.agent_trace is not None
    assert missing_key_result.agent_trace.status == "not_called"
    assert "No configured LLM provider" in (missing_key_result.agent_trace.fallback_reason or "")


def test_agent_trace_round_trip_through_storage(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "runtime.sqlite3")
    try:
        session = storage.create_session(RunSession(status="stopped"))
        created = storage.create_round(
            RoundRecord(
                session_id=session.id,
                round_index=1,
                resolved_command="python train.py",
                status="completed",
                agent_trace=AgentTrace(
                    provider="openai",
                    model="gpt-test",
                    status="parse_failed",
                    raw_output="bad output",
                    parse_error="no JSON object found",
                    fallback_reason="no JSON object found",
                ),
            )
        )

        loaded = storage.get_round(created.id or 0)

        assert loaded is not None
        assert loaded.agent_trace is not None
        assert loaded.agent_trace.raw_output == "bad output"
        assert loaded.agent_trace.status == "parse_failed"
    finally:
        storage.close()
