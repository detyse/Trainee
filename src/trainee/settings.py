from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal

from dotenv import dotenv_values

LLMProvider = Literal["none", "openai", "anthropic"]


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    project_root: Path | None
    data_dir: Path
    database_path: Path
    artifacts_dir: Path
    template_dir: Path
    static_dir: Path
    config_path: Path
    dotenv_path: Path
    llm_provider: LLMProvider
    llm_timeout_sec: float
    openai_api_key: str | None
    openai_base_url: str
    openai_model: str
    anthropic_api_key: str | None
    anthropic_base_url: str
    anthropic_model: str
    anthropic_version: str
    anthropic_max_tokens: int


def load_settings(
    repo_root: Path | None = None,
    data_dir: Path | None = None,
    project_root: Path | None = None,
) -> Settings:
    env_project_root = os.getenv("TRAINEE_PROJECT_ROOT")
    if project_root is None and env_project_root:
        project_root = Path(env_project_root)
    project_root = project_root.expanduser().resolve() if project_root is not None else None
    repo_root = (repo_root or project_root or Path(__file__).resolve().parents[2]).resolve()
    dotenv_path = repo_root / ".env"
    dotenv_payload = _read_dotenv(dotenv_path)

    data_dir = (
        data_dir.expanduser().resolve()
        if data_dir is not None
        else _resolve_path(_env_value("TRAINEE_DATA_DIR", dotenv_payload, str(Path.home() / ".trainee")), repo_root)
    )
    config_path = data_dir / "config.json"
    config_payload = _read_config(config_path)
    database_path = data_dir / "runtime.sqlite3"
    artifacts_dir = data_dir / "artifacts"
    return Settings(
        repo_root=repo_root,
        project_root=project_root,
        data_dir=data_dir,
        database_path=database_path,
        artifacts_dir=artifacts_dir,
        template_dir=Path(__file__).resolve().parent / "templates",
        static_dir=Path(__file__).resolve().parent / "static",
        config_path=config_path,
        dotenv_path=dotenv_path,
        llm_provider=_resolve_llm_provider(dotenv_payload, config_payload),
        llm_timeout_sec=float(_settings_value("TRAINEE_LLM_TIMEOUT_SEC", dotenv_payload, config_payload, "30")),
        openai_api_key=_settings_value("OPENAI_API_KEY", dotenv_payload, config_payload),
        openai_base_url=_settings_value("OPENAI_BASE_URL", dotenv_payload, config_payload, "https://api.openai.com/v1"),
        openai_model=_settings_value("OPENAI_MODEL", dotenv_payload, config_payload, "gpt-4o-mini"),
        anthropic_api_key=_settings_value("ANTHROPIC_API_KEY", dotenv_payload, config_payload),
        anthropic_base_url=_settings_value("ANTHROPIC_BASE_URL", dotenv_payload, config_payload, "https://api.anthropic.com"),
        anthropic_model=_settings_value("ANTHROPIC_MODEL", dotenv_payload, config_payload, "claude-3-5-haiku-latest"),
        anthropic_version=_settings_value("ANTHROPIC_VERSION", dotenv_payload, config_payload, "2023-06-01"),
        anthropic_max_tokens=int(_settings_value("ANTHROPIC_MAX_TOKENS", dotenv_payload, config_payload, "1024")),
    )


def save_provider_config(config_path: Path, payload: Dict[str, Any]) -> None:
    config = _read_config(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    for key in ("llm_provider", "llm_timeout_sec"):
        if key in payload:
            config[key] = payload[key]

    for provider in ("openai", "anthropic"):
        provider_payload = payload.get(provider)
        if not isinstance(provider_payload, dict):
            continue
        existing = config.get(provider)
        if not isinstance(existing, dict):
            existing = {}
        for key, value in provider_payload.items():
            if value is None:
                existing.pop(key, None)
            else:
                existing[key] = value
        config[provider] = existing

    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_llm_provider(dotenv_payload: Dict[str, str], config_payload: Dict[str, Any]) -> LLMProvider:
    requested = (
        _settings_value("TRAINEE_LLM_PROVIDER", dotenv_payload, config_payload)
        or _settings_value("LLM_PROVIDER", dotenv_payload, config_payload)
        or ""
    ).strip().lower()
    if requested:
        if requested not in {"none", "openai", "anthropic"}:
            raise ValueError("TRAINEE_LLM_PROVIDER must be one of: none, openai, anthropic")
        return requested  # type: ignore[return-value]

    if _settings_value("ANTHROPIC_API_KEY", dotenv_payload, config_payload) and not _settings_value(
        "OPENAI_API_KEY", dotenv_payload, config_payload
    ):
        return "anthropic"
    if _settings_value("OPENAI_API_KEY", dotenv_payload, config_payload):
        return "openai"
    if _settings_value("ANTHROPIC_API_KEY", dotenv_payload, config_payload):
        return "anthropic"
    return "none"


def _read_dotenv(dotenv_path: Path) -> Dict[str, str]:
    if not dotenv_path.exists():
        return {}
    payload = dotenv_values(dotenv_path)
    return {key: value for key, value in payload.items() if value is not None}


def _read_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{config_path} must contain valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    return payload


def _env_value(name: str, dotenv_payload: Dict[str, str], default: str | None = None) -> str | None:
    return os.getenv(name, dotenv_payload.get(name, default))


def _settings_value(
    name: str,
    dotenv_payload: Dict[str, str],
    config_payload: Dict[str, Any],
    default: str | None = None,
) -> str | None:
    env_value = os.getenv(name)
    if env_value is not None:
        return env_value
    config_value = _config_value(name, config_payload)
    if config_value is not None:
        return str(config_value)
    return dotenv_payload.get(name, default)


def _config_value(name: str, config_payload: Dict[str, Any]) -> Any:
    if name in {"TRAINEE_LLM_PROVIDER", "LLM_PROVIDER"}:
        return config_payload.get("llm_provider")
    if name == "TRAINEE_LLM_TIMEOUT_SEC":
        return config_payload.get("llm_timeout_sec")
    provider_keys = {
        "OPENAI_API_KEY": ("openai", "api_key"),
        "OPENAI_BASE_URL": ("openai", "base_url"),
        "OPENAI_MODEL": ("openai", "model"),
        "ANTHROPIC_API_KEY": ("anthropic", "api_key"),
        "ANTHROPIC_BASE_URL": ("anthropic", "base_url"),
        "ANTHROPIC_MODEL": ("anthropic", "model"),
        "ANTHROPIC_VERSION": ("anthropic", "version"),
        "ANTHROPIC_MAX_TOKENS": ("anthropic", "max_tokens"),
    }
    if name in provider_keys:
        provider, key = provider_keys[name]
        return _provider_config_value(config_payload, provider, key)
    return None


def _provider_config_value(config_payload: Dict[str, Any], provider: str, key: str) -> Any:
    provider_payload = config_payload.get(provider)
    if not isinstance(provider_payload, dict):
        return None
    return provider_payload.get(key)


def _resolve_path(raw: str, repo_root: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()
