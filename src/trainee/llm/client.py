from __future__ import annotations

import json
from typing import Any, Optional

import httpx

from trainee.llm.adapters import adapter_for
from trainee.llm.types import ProviderAttempt, ProviderCallError, ProviderCompletion, ProviderDispatchError, ProviderDispatchResult
from trainee.providers import configured_provider_order, provider_model
from trainee.settings import Settings


PROVIDER_ERROR_BODY_PREVIEW_CHARS = 500


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def configured_providers(self) -> list[str]:
        return list(configured_provider_order(self.settings))

    def url_for(self, provider: str) -> str:
        return adapter_for(self.settings, provider).url()

    def build_payload(
        self,
        provider: str,
        system_prompt: str,
        user_prompt: str,
        image: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        return adapter_for(self.settings, provider).build_payload(system_prompt, user_prompt, image=image)

    async def complete_active(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Optional[dict[str, str]] = None,
    ) -> ProviderCompletion:
        return await self.complete(self.settings.llm_provider, system_prompt, user_prompt, image=image)

    async def complete(
        self,
        provider: str,
        system_prompt: str,
        user_prompt: str,
        image: Optional[dict[str, str]] = None,
    ) -> ProviderCompletion:
        payload = self.build_payload(provider, system_prompt, user_prompt, image=image)
        return await self.send_payload(provider, payload)

    async def complete_with_fallback(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Optional[dict[str, str]] = None,
    ) -> ProviderDispatchResult:
        attempts: list[dict[str, Any]] = []
        last_error: ProviderCallError | None = None
        for provider in self.configured_providers():
            model = provider_model(self.settings, provider)
            try:
                completion = await self.complete(provider, system_prompt, user_prompt, image=image)
            except ProviderCallError as exc:
                last_error = exc
                attempts.append(
                    ProviderAttempt(
                        provider=provider,
                        model=model,
                        ok=False,
                        status=exc.status,
                        http_status=exc.http_status,
                        request_id=exc.request_id,
                        error_message=str(exc),
                    ).as_trace_dict()
                )
                continue
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                last_error = ProviderCallError(message, status="response_failed")
                attempts.append(
                    ProviderAttempt(
                        provider=provider,
                        model=model,
                        ok=False,
                        status="response_failed",
                        error_message=message,
                    ).as_trace_dict()
                )
                continue

            attempts.append(
                ProviderAttempt(
                    provider=provider,
                    model=model,
                    ok=True,
                    status="success",
                    http_status=completion.http_status,
                    request_id=completion.request_id,
                ).as_trace_dict()
            )
            return ProviderDispatchResult(
                provider=provider,
                model=model,
                completion=completion,
                attempts=attempts,
            )

        raise ProviderDispatchError(
            "All configured LLM providers failed.",
            attempts=attempts,
            last_error=last_error,
        )

    async def send_payload(self, provider: str, payload: dict[str, Any]) -> ProviderCompletion:
        adapter = adapter_for(self.settings, provider)
        body, raw_body, http_status, request_id = await self._post_provider_json(
            adapter.url(),
            payload,
            adapter.headers(),
        )
        return adapter.parse_completion(body, raw_body, http_status, request_id)

    async def _post_provider_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> tuple[dict[str, Any], str, int, Optional[str]]:
        try:
            async with httpx.AsyncClient(timeout=self.settings.llm_timeout_sec) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderCallError(
                self._provider_timeout_message(url, exc),
                status="request_failed",
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderCallError(
                self._provider_request_error_message(url, exc),
                status="request_failed",
            ) from exc

        raw_body = response.text
        request_id = self._request_id(response)
        if response.status_code < 200 or response.status_code >= 300:
            raise ProviderCallError(
                self._provider_http_error_message(url, response.status_code, request_id, raw_body),
                status="request_failed",
                http_status=response.status_code,
                request_id=request_id,
                raw_response_body=raw_body,
                error_body=raw_body,
            )
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProviderCallError(
                f"Provider response body from {url} was not valid JSON: {exc}",
                status="response_failed",
                http_status=response.status_code,
                request_id=request_id,
                raw_response_body=raw_body,
            ) from exc
        if not isinstance(body, dict):
            raise ProviderCallError(
                f"Provider response body from {url} must be a JSON object.",
                status="response_failed",
                http_status=response.status_code,
                request_id=request_id,
                raw_response_body=raw_body,
            )
        return body, raw_body, response.status_code, request_id

    def _provider_timeout_message(self, url: str, exc: httpx.TimeoutException) -> str:
        message = f"{type(exc).__name__} after {self.settings.llm_timeout_sec:g}s while POST {url}."
        detail = str(exc).strip()
        if detail:
            message += f" {detail}"
        return message

    def _provider_request_error_message(self, url: str, exc: httpx.RequestError) -> str:
        message = f"{type(exc).__name__} while POST {url}."
        detail = str(exc).strip()
        if detail:
            message += f" {detail}"
        return message

    def _provider_http_error_message(
        self,
        url: str,
        http_status: int,
        request_id: Optional[str],
        raw_body: str,
    ) -> str:
        message = f"Provider returned HTTP {http_status} from {url}."
        if request_id:
            message += f" request_id={request_id}."
        body = raw_body.strip()
        if body:
            message += f" body={self._clip_provider_error_body(body)}"
        return message

    def _clip_provider_error_body(self, value: str) -> str:
        if len(value) <= PROVIDER_ERROR_BODY_PREVIEW_CHARS:
            return value
        return value[:PROVIDER_ERROR_BODY_PREVIEW_CHARS].rstrip() + (
            f"... clipped {len(value) - PROVIDER_ERROR_BODY_PREVIEW_CHARS} chars"
        )

    def _request_id(self, response: httpx.Response) -> Optional[str]:
        for header in ("x-request-id", "request-id", "openai-request-id"):
            value = response.headers.get(header)
            if value:
                return value
        return None
