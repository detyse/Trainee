from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from trainee.llm import LLMClient, ProviderCallError
from trainee.providers import active_model, configured_provider_order, provider_model
from trainee.settings import Settings


PROVIDER_TEST_SYSTEM_PROMPT = "You are a concise API health probe. Return JSON only."
PROVIDER_TEST_USER_PROMPT = 'Return exactly {"ok": true, "message": "pong"} and nothing else.'
MAX_ERROR_BODY_CHARS = 1200


class ProviderProbeAttempt(BaseModel):
    provider: str
    model: str = "none"
    ok: bool = False
    status: str
    http_status: Optional[int] = None
    request_id: Optional[str] = None
    error_message: str = ""
    error_body: str = ""


class ProviderProbeResult(BaseModel):
    provider: str = "none"
    model: str = "none"
    ok: bool = False
    status: str = "not_run"
    http_status: Optional[int] = None
    request_id: Optional[str] = None
    error_message: str = ""
    error_body: str = ""
    attempts: list[ProviderProbeAttempt] = Field(default_factory=list)

    def failure_summary(self) -> str:
        if self.ok:
            return ""
        if self.error_message:
            return self.error_message
        if not self.attempts:
            return self.status
        return "; ".join(
            f"{attempt.provider}:{attempt.status}"
            + (f" HTTP {attempt.http_status}" if attempt.http_status is not None else "")
            + (f" {attempt.error_message}" if attempt.error_message else "")
            for attempt in self.attempts
        )


async def probe_provider(settings: Settings) -> ProviderProbeResult:
    providers = configured_provider_order(settings)
    if not providers:
        return ProviderProbeResult(
            provider=settings.llm_provider,
            model=active_model(settings),
            ok=False,
            status="not_configured",
            error_message="No configured LLM provider is available.",
        )

    attempts: list[ProviderProbeAttempt] = []
    client = LLMClient(settings)
    for provider in providers:
        model = provider_model(settings, provider)
        try:
            completion = await client.complete(
                provider,
                PROVIDER_TEST_SYSTEM_PROMPT,
                PROVIDER_TEST_USER_PROMPT,
            )
        except ProviderCallError as exc:
            attempts.append(_attempt_from_provider_error(provider, model, exc))
            continue
        except Exception as exc:
            attempts.append(
                ProviderProbeAttempt(
                    provider=provider,
                    model=model,
                    ok=False,
                    status="response_failed",
                    error_message=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        attempt = ProviderProbeAttempt(
            provider=provider,
            model=model,
            ok=True,
            status="success",
            http_status=completion.http_status,
            request_id=completion.request_id,
        )
        return ProviderProbeResult(
            provider=provider,
            model=model,
            ok=True,
            status="success",
            http_status=completion.http_status,
            request_id=completion.request_id,
            attempts=[*attempts, attempt],
        )

    last = attempts[-1]
    return ProviderProbeResult(
        provider=last.provider,
        model=last.model,
        ok=False,
        status=last.status,
        http_status=last.http_status,
        request_id=last.request_id,
        error_message=last.error_message,
        error_body=last.error_body,
        attempts=attempts,
    )


def _attempt_from_provider_error(provider: str, model: str, exc: ProviderCallError) -> ProviderProbeAttempt:
    return ProviderProbeAttempt(
        provider=provider,
        model=model,
        ok=False,
        status=exc.status,
        http_status=exc.http_status,
        request_id=exc.request_id,
        error_message=str(exc),
        error_body=_clip_body(exc.error_body or exc.raw_response_body or ""),
    )


def _clip_body(value: str) -> str:
    if len(value) <= MAX_ERROR_BODY_CHARS:
        return value
    return value[:MAX_ERROR_BODY_CHARS] + f"\n... clipped {len(value) - MAX_ERROR_BODY_CHARS} chars ..."
