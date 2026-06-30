from __future__ import annotations

from typing import Any, Optional, Protocol

from trainee.llm.types import ProviderCallError, ProviderCompletion
from trainee.settings import Settings


class ProviderAdapter(Protocol):
    provider: str

    def url(self) -> str: ...

    def headers(self) -> dict[str, str]: ...

    def build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]: ...

    def parse_completion(
        self,
        body: dict[str, Any],
        raw_body: str,
        http_status: int,
        request_id: Optional[str],
    ) -> ProviderCompletion: ...


class OpenAIAdapter:
    provider = "openai"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def url(self) -> str:
        return self.settings.openai_base_url.rstrip("/") + "/chat/completions"

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.openai_api_key}"}

    def build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        return {
            "model": self.settings.openai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _openai_user_content(user_prompt, image)},
            ],
            "temperature": self.settings.llm_temperature,
        }

    def parse_completion(
        self,
        body: dict[str, Any],
        raw_body: str,
        http_status: int,
        request_id: Optional[str],
    ) -> ProviderCompletion:
        try:
            choice = body["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("message content must be a string")
        except (KeyError, IndexError, TypeError) as exc:
            raise _response_shape_error(
                "OpenAI response did not contain choices[0].message.content",
                exc,
                raw_body,
                http_status,
                request_id,
            ) from exc
        return ProviderCompletion(
            content=content,
            raw_response_body=raw_body,
            http_status=http_status,
            request_id=request_id,
            usage=body.get("usage") if isinstance(body.get("usage"), dict) else None,
            finish_reason=str(choice["finish_reason"]) if choice.get("finish_reason") is not None else None,
        )


class MoonshotAdapter:
    provider = "moonshot"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def url(self) -> str:
        return self.settings.moonshot_base_url.rstrip("/") + "/chat/completions"

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.moonshot_api_key}"}

    def build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        return {
            "model": self.settings.moonshot_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _openai_user_content(user_prompt, image)},
            ],
            "temperature": self.settings.llm_temperature,
        }

    def parse_completion(
        self,
        body: dict[str, Any],
        raw_body: str,
        http_status: int,
        request_id: Optional[str],
    ) -> ProviderCompletion:
        try:
            choice = body["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("message content must be a string")
        except (KeyError, IndexError, TypeError) as exc:
            raise _response_shape_error(
                "Moonshot response did not contain choices[0].message.content",
                exc,
                raw_body,
                http_status,
                request_id,
            ) from exc
        return ProviderCompletion(
            content=content,
            raw_response_body=raw_body,
            http_status=http_status,
            request_id=request_id,
            usage=body.get("usage") if isinstance(body.get("usage"), dict) else None,
            finish_reason=str(choice["finish_reason"]) if choice.get("finish_reason") is not None else None,
        )


class AnthropicAdapter:
    provider = "anthropic"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def url(self) -> str:
        return self.settings.anthropic_base_url.rstrip("/") + "/v1/messages"

    def headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.settings.anthropic_api_key or "",
            "anthropic-version": self.settings.anthropic_version,
            "content-type": "application/json",
        }

    def build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        return {
            "model": self.settings.anthropic_model,
            "max_tokens": self.settings.anthropic_max_tokens,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": _anthropic_user_content(user_prompt, image),
                }
            ],
            "temperature": self.settings.llm_temperature,
        }

    def parse_completion(
        self,
        body: dict[str, Any],
        raw_body: str,
        http_status: int,
        request_id: Optional[str],
    ) -> ProviderCompletion:
        try:
            text_parts = [
                block.get("text", "")
                for block in body["content"]
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = "\n".join(part for part in text_parts if part)
            if not content:
                raise ValueError("no text content blocks")
        except (KeyError, TypeError, ValueError) as exc:
            raise _response_shape_error(
                "Anthropic response did not contain text content",
                exc,
                raw_body,
                http_status,
                request_id,
            ) from exc
        return ProviderCompletion(
            content=content,
            raw_response_body=raw_body,
            http_status=http_status,
            request_id=request_id,
            usage=body.get("usage") if isinstance(body.get("usage"), dict) else None,
            finish_reason=str(body["stop_reason"]) if body.get("stop_reason") is not None else None,
        )


def adapter_for(settings: Settings, provider: str) -> ProviderAdapter:
    if provider == "openai":
        return OpenAIAdapter(settings)
    if provider == "moonshot":
        return MoonshotAdapter(settings)
    if provider == "anthropic":
        return AnthropicAdapter(settings)
    raise ProviderCallError(f"unsupported provider: {provider}", status="not_called")


def _openai_user_content(prompt: str, image: Optional[dict[str, str]]) -> Any:
    if image is None:
        return prompt
    return [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{image['media_type']};base64,{image['data']}"},
        },
    ]


def _anthropic_user_content(prompt: str, image: Optional[dict[str, str]]) -> Any:
    if image is None:
        return prompt
    return [
        {"type": "text", "text": prompt},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image["media_type"],
                "data": image["data"],
            },
        },
    ]


def _response_shape_error(
    message: str,
    exc: Exception,
    raw_body: str,
    http_status: int,
    request_id: Optional[str],
) -> ProviderCallError:
    return ProviderCallError(
        f"{message}: {exc}",
        status="response_failed",
        http_status=http_status,
        request_id=request_id,
        raw_response_body=raw_body,
    )
