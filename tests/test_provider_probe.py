from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from trainee.provider_probe import probe_provider
from trainee.settings import Settings


def test_provider_probe_success(tmp_path: Path, monkeypatch) -> None:
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
                headers={"x-request-id": "req-ok"},
                json={"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}]},
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    result = asyncio.run(probe_provider(_settings(tmp_path, "moonshot", moonshot_key="key")))

    assert result.ok is True
    assert result.provider == "moonshot"
    assert result.http_status == 200
    assert result.request_id == "req-ok"


def test_provider_probe_records_http_error(tmp_path: Path, monkeypatch) -> None:
    class FakeAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json, headers):
            return httpx.Response(
                401,
                request=httpx.Request("POST", url),
                json={"error": {"message": "Invalid Authentication"}},
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    result = asyncio.run(probe_provider(_settings(tmp_path, "moonshot", moonshot_key="bad-key")))

    assert result.ok is False
    assert result.status == "request_failed"
    assert result.http_status == 401
    assert "Invalid Authentication" in result.error_body


def test_provider_probe_records_timeout(tmp_path: Path, monkeypatch) -> None:
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

    result = asyncio.run(probe_provider(_settings(tmp_path, "moonshot", moonshot_key="key")))

    assert result.ok is False
    assert result.status == "request_failed"
    assert "ReadTimeout" in result.error_message


def test_provider_probe_reports_unconfigured(tmp_path: Path) -> None:
    result = asyncio.run(probe_provider(_settings(tmp_path, "none")))

    assert result.ok is False
    assert result.status == "not_configured"
    assert "No configured LLM provider" in result.error_message


def test_provider_probe_falls_back_to_next_configured_provider(tmp_path: Path, monkeypatch) -> None:
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
                json={"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}]},
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    settings = _settings(tmp_path, "moonshot", moonshot_key="bad", openai_key="good")

    result = asyncio.run(probe_provider(settings))

    assert result.ok is True
    assert result.provider == "openai"
    assert [attempt.provider for attempt in result.attempts] == ["moonshot", "openai"]
    assert result.attempts[0].ok is False


def _settings(
    tmp_path: Path,
    provider: str,
    *,
    moonshot_key: str | None = None,
    openai_key: str | None = None,
) -> Settings:
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
        openai_api_key=openai_key,
        openai_base_url="https://openai.example/v1",
        openai_model="gpt-test",
        moonshot_api_key=moonshot_key,
        moonshot_base_url="https://moonshot.example/v1",
        moonshot_model="kimi-test",
        anthropic_api_key=None,
        anthropic_base_url="https://anthropic.example",
        anthropic_model="claude-test",
        anthropic_version="2023-06-01",
        anthropic_max_tokens=512,
    )
