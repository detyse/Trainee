from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from trainee.llm import LLMClient, ProviderCallError
from trainee.settings import Settings


def test_llm_client_builds_existing_provider_payload_shapes(tmp_path: Path) -> None:
    settings = _settings(tmp_path, "moonshot")
    client = LLMClient(settings)
    image = {"media_type": "image/png", "data": "abc123"}

    openai_payload = client.build_payload("openai", "system", "user", image=image)
    moonshot_payload = client.build_payload("moonshot", "system", "user", image=image)
    anthropic_payload = client.build_payload("anthropic", "system", "user", image=image)

    assert openai_payload == {
        "model": "gpt-test",
        "messages": [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "user"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
                ],
            },
        ],
        "temperature": 1.0,
    }
    assert moonshot_payload["model"] == "kimi-test"
    assert moonshot_payload["messages"] == openai_payload["messages"]
    assert anthropic_payload == {
        "model": "claude-test",
        "max_tokens": 512,
        "system": "system",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "user"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": "abc123"},
                    },
                ],
            }
        ],
        "temperature": 1.0,
    }


def test_llm_client_sends_and_parses_openai_compatible_response(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

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
            return httpx.Response(
                200,
                headers={"x-request-id": "req-1"},
                request=httpx.Request("POST", url),
                json={
                    "choices": [{"message": {"content": "pong"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1},
                },
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    completion = asyncio.run(LLMClient(_settings(tmp_path, "moonshot")).complete_active("system", "user"))

    assert captured["url"] == "https://moonshot.example/v1/chat/completions"
    assert captured["headers"] == {"Authorization": "Bearer moonshot-key"}
    assert captured["json"] == {
        "model": "kimi-test",
        "messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
        "temperature": 1.0,
    }
    assert captured["timeout"] == 5.0
    assert completion.content == "pong"
    assert completion.request_id == "req-1"
    assert completion.usage == {"prompt_tokens": 1}
    assert completion.finish_reason == "stop"


def test_llm_client_parses_anthropic_response(tmp_path: Path, monkeypatch) -> None:
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
                json={
                    "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}],
                    "usage": {"input_tokens": 3},
                    "stop_reason": "end_turn",
                },
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    completion = asyncio.run(LLMClient(_settings(tmp_path, "anthropic")).complete_active("system", "user"))

    assert completion.content == "hello\nworld"
    assert completion.usage == {"input_tokens": 3}
    assert completion.finish_reason == "end_turn"


@pytest.mark.parametrize(
    ("response", "expected_status", "expected_message"),
    [
        (httpx.Response(429, request=httpx.Request("POST", "https://openai.example/v1/chat/completions"), text="rate limited"), "request_failed", "Provider returned HTTP 429"),
        (httpx.Response(200, request=httpx.Request("POST", "https://openai.example/v1/chat/completions"), text="not json"), "response_failed", "not valid JSON"),
        (httpx.Response(200, request=httpx.Request("POST", "https://openai.example/v1/chat/completions"), json={"choices": []}), "response_failed", "choices[0].message.content"),
    ],
)
def test_llm_client_maps_provider_failures(tmp_path: Path, monkeypatch, response, expected_status, expected_message) -> None:
    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json, headers):
            return response

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    with pytest.raises(ProviderCallError) as error:
        asyncio.run(LLMClient(_settings(tmp_path, "openai")).complete_active("system", "user"))

    assert error.value.status == expected_status
    assert expected_message in str(error.value)


def test_llm_client_records_timeout_and_fallback_attempts(tmp_path: Path, monkeypatch) -> None:
    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json, headers):
            if "moonshot" in url:
                raise httpx.ReadTimeout("provider timed out", request=httpx.Request("POST", url))
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={"choices": [{"message": {"content": "fallback"}}]},
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    result = asyncio.run(LLMClient(_settings(tmp_path, "moonshot")).complete_with_fallback("system", "user"))

    assert result.provider == "openai"
    assert result.completion.content == "fallback"
    assert [attempt["provider"] for attempt in result.attempts] == ["moonshot", "openai"]
    assert result.attempts[0]["status"] == "request_failed"
    assert "ReadTimeout after 5s" in result.attempts[0]["error_message"]


def _settings(tmp_path: Path, provider: str) -> Settings:
    return Settings(
        repo_root=tmp_path,
        project_root=tmp_path,
        project_data_dir=tmp_path / ".trainee",
        database_path=tmp_path / ".trainee" / "runtime.sqlite3",
        artifacts_dir=tmp_path / ".trainee" / "artifacts",
        template_dir=tmp_path / "templates",
        static_dir=tmp_path / "static",
        global_config_path=tmp_path / "home" / ".trainee" / "config.json",
        llm_provider=provider,  # type: ignore[arg-type]
        llm_timeout_sec=5.0,
        openai_api_key="openai-key",
        openai_base_url="https://openai.example/v1",
        openai_model="gpt-test",
        anthropic_api_key="anthropic-key",
        anthropic_base_url="https://anthropic.example",
        anthropic_model="claude-test",
        anthropic_version="2023-06-01",
        anthropic_max_tokens=512,
        moonshot_api_key="moonshot-key",
        moonshot_base_url="https://moonshot.example/v1",
        moonshot_model="kimi-test",
        system_prompt="test",
    )
