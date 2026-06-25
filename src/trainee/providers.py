from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from trainee.settings import DEFAULT_MOONSHOT_BASE_URL, DEFAULT_MOONSHOT_MODEL, LLMProvider, Settings

ProviderName = Literal["none", "moonshot", "openai", "anthropic"]
PROVIDER_NAMES: tuple[ProviderName, ...] = ("none", "moonshot", "openai", "anthropic")

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-haiku-latest"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_ANTHROPIC_MAX_TOKENS = 1024
DEFAULT_LLM_TIMEOUT_SEC = 30.0


class OpenAIProviderUpdate(BaseModel):
    api_key: Optional[str] = None
    clear_api_key: bool = False
    base_url: str = DEFAULT_OPENAI_BASE_URL
    model: str = DEFAULT_OPENAI_MODEL


class MoonshotProviderUpdate(BaseModel):
    api_key: Optional[str] = None
    clear_api_key: bool = False
    base_url: str = DEFAULT_MOONSHOT_BASE_URL
    model: str = DEFAULT_MOONSHOT_MODEL


class AnthropicProviderUpdate(BaseModel):
    api_key: Optional[str] = None
    clear_api_key: bool = False
    base_url: str = DEFAULT_ANTHROPIC_BASE_URL
    model: str = DEFAULT_ANTHROPIC_MODEL
    version: str = DEFAULT_ANTHROPIC_VERSION
    max_tokens: int = DEFAULT_ANTHROPIC_MAX_TOKENS


class ProviderSettingsUpdate(BaseModel):
    llm_provider: str = "none"
    llm_timeout_sec: float = DEFAULT_LLM_TIMEOUT_SEC
    openai: OpenAIProviderUpdate = Field(default_factory=OpenAIProviderUpdate)
    moonshot: MoonshotProviderUpdate = Field(default_factory=MoonshotProviderUpdate)
    anthropic: AnthropicProviderUpdate = Field(default_factory=AnthropicProviderUpdate)


class AgentDebugSettingsUpdate(BaseModel):
    agent_debug_enabled: bool = False


class SystemPromptUpdate(BaseModel):
    system_prompt: str


def normalize_provider(provider: str) -> LLMProvider:
    normalized = provider.strip().lower()
    if normalized not in PROVIDER_NAMES:
        raise ValueError("provider must be one of: none, moonshot, openai, anthropic")
    return normalized  # type: ignore[return-value]


def active_model(settings: Settings) -> str:
    if settings.llm_provider == "openai":
        return settings.openai_model
    if settings.llm_provider == "moonshot":
        return settings.moonshot_model
    if settings.llm_provider == "anthropic":
        return settings.anthropic_model
    return "none"


def provider_is_configured(settings: Settings) -> bool:
    if settings.llm_provider == "openai":
        return bool(settings.openai_api_key)
    if settings.llm_provider == "moonshot":
        return bool(settings.moonshot_api_key)
    if settings.llm_provider == "anthropic":
        return bool(settings.anthropic_api_key)
    return False


def provider_settings_payload(settings: Settings) -> Dict[str, Any]:
    return {
        "llm_provider": settings.llm_provider,
        "llm_timeout_sec": settings.llm_timeout_sec,
        "agent_debug_enabled": settings.agent_debug_enabled,
        "active_model": active_model(settings),
        "openai_base_url": settings.openai_base_url,
        "openai_model": settings.openai_model,
        "openai_key_configured": bool(settings.openai_api_key),
        "moonshot_base_url": settings.moonshot_base_url,
        "moonshot_model": settings.moonshot_model,
        "moonshot_key_configured": bool(settings.moonshot_api_key),
        "anthropic_base_url": settings.anthropic_base_url,
        "anthropic_model": settings.anthropic_model,
        "anthropic_version": settings.anthropic_version,
        "anthropic_max_tokens": settings.anthropic_max_tokens,
        "anthropic_key_configured": bool(settings.anthropic_api_key),
    }


def build_provider_config_payload(update: ProviderSettingsUpdate) -> Dict[str, Any]:
    provider = normalize_provider(update.llm_provider)
    if update.llm_timeout_sec <= 0:
        raise ValueError("timeout must be positive")
    if update.anthropic.max_tokens <= 0:
        raise ValueError("anthropic max tokens must be positive")

    payload: Dict[str, Any] = {
        "llm_provider": provider,
        "llm_timeout_sec": update.llm_timeout_sec,
        "openai": {
            "base_url": _clean(update.openai.base_url, DEFAULT_OPENAI_BASE_URL),
            "model": _clean(update.openai.model, DEFAULT_OPENAI_MODEL),
        },
        "moonshot": {
            "base_url": _clean(update.moonshot.base_url, DEFAULT_MOONSHOT_BASE_URL),
            "model": _clean(update.moonshot.model, DEFAULT_MOONSHOT_MODEL),
        },
        "anthropic": {
            "base_url": _clean(update.anthropic.base_url, DEFAULT_ANTHROPIC_BASE_URL),
            "model": _clean(update.anthropic.model, DEFAULT_ANTHROPIC_MODEL),
            "version": _clean(update.anthropic.version, DEFAULT_ANTHROPIC_VERSION),
            "max_tokens": update.anthropic.max_tokens,
        },
    }
    _apply_secret_update(payload["openai"], update.openai.api_key, update.openai.clear_api_key)
    _apply_secret_update(payload["moonshot"], update.moonshot.api_key, update.moonshot.clear_api_key)
    _apply_secret_update(payload["anthropic"], update.anthropic.api_key, update.anthropic.clear_api_key)
    return payload


def provider_update_from_form(
    *,
    llm_provider: str,
    llm_timeout_sec: float,
    openai_api_key: str,
    clear_openai_api_key: bool,
    openai_base_url: str,
    openai_model: str,
    moonshot_api_key: str,
    clear_moonshot_api_key: bool,
    moonshot_base_url: str,
    moonshot_model: str,
    anthropic_api_key: str,
    clear_anthropic_api_key: bool,
    anthropic_base_url: str,
    anthropic_model: str,
    anthropic_version: str,
    anthropic_max_tokens: int,
) -> ProviderSettingsUpdate:
    return ProviderSettingsUpdate(
        llm_provider=llm_provider,
        llm_timeout_sec=llm_timeout_sec,
        openai=OpenAIProviderUpdate(
            api_key=openai_api_key,
            clear_api_key=clear_openai_api_key,
            base_url=openai_base_url,
            model=openai_model,
        ),
        moonshot=MoonshotProviderUpdate(
            api_key=moonshot_api_key,
            clear_api_key=clear_moonshot_api_key,
            base_url=moonshot_base_url,
            model=moonshot_model,
        ),
        anthropic=AnthropicProviderUpdate(
            api_key=anthropic_api_key,
            clear_api_key=clear_anthropic_api_key,
            base_url=anthropic_base_url,
            model=anthropic_model,
            version=anthropic_version,
            max_tokens=anthropic_max_tokens,
        ),
    )


def _apply_secret_update(provider_payload: Dict[str, Any], api_key: Optional[str], clear_api_key: bool) -> None:
    if clear_api_key:
        provider_payload["api_key"] = None
        return
    if api_key and api_key.strip():
        provider_payload["api_key"] = api_key.strip()


def _clean(value: str, default: str) -> str:
    normalized = value.strip()
    return normalized or default
