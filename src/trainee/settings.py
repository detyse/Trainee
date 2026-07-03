from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal

LLMProvider = Literal["none", "moonshot", "openai", "anthropic"]

DEFAULT_MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_MOONSHOT_MODEL = "kimi-k2.6"
DEFAULT_LLM_TIMEOUT_SEC = 600.0
DEFAULT_LLM_TEMPERATURE = 1.0
DEFAULT_MAX_IMAGE_ANALYSES_PER_SESSION = 3
DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "defaults" / "system_prompt.txt"
PROVIDER_ENV_NAMES = (
    "TRAINEE_LLM_PROVIDER",
    "LLM_PROVIDER",
    "TRAINEE_LLM_TIMEOUT_SEC",
    "TRAINEE_LLM_TEMPERATURE",
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
)
PROVIDER_CONFIG_KEYS = {
    "OPENAI_API_KEY": ("openai", "api_key"),
    "OPENAI_BASE_URL": ("openai", "base_url"),
    "OPENAI_MODEL": ("openai", "model"),
    "MOONSHOT_API_KEY": ("moonshot", "api_key"),
    "MOONSHOT_BASE_URL": ("moonshot", "base_url"),
    "MOONSHOT_MODEL": ("moonshot", "model"),
    "ANTHROPIC_API_KEY": ("anthropic", "api_key"),
    "ANTHROPIC_BASE_URL": ("anthropic", "base_url"),
    "ANTHROPIC_MODEL": ("anthropic", "model"),
    "ANTHROPIC_VERSION": ("anthropic", "version"),
    "ANTHROPIC_MAX_TOKENS": ("anthropic", "max_tokens"),
}


def load_default_system_prompt() -> str:
    try:
        content = DEFAULT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"default system prompt is unavailable: {DEFAULT_SYSTEM_PROMPT_PATH}") from exc
    if not content.strip():
        raise ValueError(f"default system prompt is empty: {DEFAULT_SYSTEM_PROMPT_PATH}")
    return content


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    project_root: Path | None
    project_data_dir: Path
    database_path: Path
    artifacts_dir: Path
    template_dir: Path
    static_dir: Path
    global_config_path: Path
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
    system_prompt: str = field(default_factory=load_default_system_prompt)
    moonshot_api_key: str | None = None
    moonshot_base_url: str = DEFAULT_MOONSHOT_BASE_URL
    moonshot_model: str = DEFAULT_MOONSHOT_MODEL
    max_image_analyses_per_session: int = DEFAULT_MAX_IMAGE_ANALYSES_PER_SESSION
    agent_debug_enabled: bool = False
    llm_temperature: float = DEFAULT_LLM_TEMPERATURE
    environment_overrides: tuple[str, ...] = field(default_factory=tuple)

    @property
    def data_dir(self) -> Path:
        return self.project_data_dir

    @property
    def config_path(self) -> Path:
        return self.global_config_path


def load_settings(
    repo_root: Path | None = None,
    data_dir: Path | None = None,
    project_root: Path | None = None,
    global_config_path: Path | None = None,
) -> Settings:
    env_project_root = os.getenv("TRAINEE_PROJECT_ROOT")
    if project_root is None and env_project_root:
        project_root = Path(env_project_root)
    project_root = project_root.expanduser().resolve() if project_root is not None else None
    repo_root = (repo_root or project_root or Path(__file__).resolve().parents[2]).resolve()

    global_config_path = (
        global_config_path.expanduser().resolve()
        if global_config_path is not None
        else (Path.home() / ".trainee" / "config.json").expanduser().resolve()
    )
    project_data_dir = _resolve_project_data_dir(repo_root, project_root, data_dir)
    config_payload = _read_config(global_config_path)
    system_prompt = _ensure_system_prompt(global_config_path, config_payload)
    database_path = project_data_dir / "runtime.sqlite3"
    artifacts_dir = project_data_dir / "artifacts"
    return Settings(
        repo_root=repo_root,
        project_root=project_root,
        project_data_dir=project_data_dir,
        database_path=database_path,
        artifacts_dir=artifacts_dir,
        template_dir=Path(__file__).resolve().parent / "templates",
        static_dir=Path(__file__).resolve().parent / "static",
        global_config_path=global_config_path,
        llm_provider=_resolve_llm_provider(config_payload),
        llm_timeout_sec=float(
            _settings_value("TRAINEE_LLM_TIMEOUT_SEC", config_payload, str(DEFAULT_LLM_TIMEOUT_SEC))
        ),
        llm_temperature=float(_settings_value("TRAINEE_LLM_TEMPERATURE", config_payload, str(DEFAULT_LLM_TEMPERATURE))),
        openai_api_key=_settings_value("OPENAI_API_KEY", config_payload),
        openai_base_url=_settings_value("OPENAI_BASE_URL", config_payload, "https://api.openai.com/v1"),
        openai_model=_settings_value("OPENAI_MODEL", config_payload, "gpt-4o-mini"),
        moonshot_api_key=_settings_value("MOONSHOT_API_KEY", config_payload),
        moonshot_base_url=_settings_value("MOONSHOT_BASE_URL", config_payload, DEFAULT_MOONSHOT_BASE_URL),
        moonshot_model=_settings_value("MOONSHOT_MODEL", config_payload, DEFAULT_MOONSHOT_MODEL),
        anthropic_api_key=_settings_value("ANTHROPIC_API_KEY", config_payload),
        anthropic_base_url=_settings_value("ANTHROPIC_BASE_URL", config_payload, "https://api.anthropic.com"),
        anthropic_model=_settings_value("ANTHROPIC_MODEL", config_payload, "claude-3-5-haiku-latest"),
        anthropic_version=_settings_value("ANTHROPIC_VERSION", config_payload, "2023-06-01"),
        anthropic_max_tokens=int(_settings_value("ANTHROPIC_MAX_TOKENS", config_payload, "1024")),
        system_prompt=system_prompt,
        max_image_analyses_per_session=int(
            _settings_value(
                "TRAINEE_MAX_IMAGE_ANALYSES_PER_SESSION",
                config_payload,
                str(DEFAULT_MAX_IMAGE_ANALYSES_PER_SESSION),
            )
        ),
        agent_debug_enabled=_config_bool(config_payload, "agent_debug_enabled", False),
        environment_overrides=_active_provider_env_overrides(),
    )


def save_global_config(config_path: Path, payload: Dict[str, Any]) -> None:
    config = _read_config(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    for key in ("llm_provider", "llm_timeout_sec", "llm_temperature", "agent_debug_enabled", "system_prompt"):
        if key in payload:
            config[key] = payload[key]

    for provider in ("openai", "moonshot", "anthropic"):
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

    _write_config(config_path, config)


def _ensure_system_prompt(config_path: Path, config: Dict[str, Any]) -> str:
    if "system_prompt" not in config:
        system_prompt = load_default_system_prompt()
        config["system_prompt"] = system_prompt
        _write_config(config_path, config)
        return system_prompt

    system_prompt = config["system_prompt"]
    if not isinstance(system_prompt, str):
        raise ValueError(f"{config_path} system_prompt must be a string")
    if not system_prompt.strip():
        raise ValueError(f"{config_path} system_prompt cannot be blank")
    return system_prompt


def _write_config(config_path: Path, config: Dict[str, Any]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_llm_provider(config_payload: Dict[str, Any]) -> LLMProvider:
    requested = (os.getenv("TRAINEE_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "").strip().lower()
    if not requested:
        requested = str(config_payload.get("llm_provider") or "").strip().lower()
    if requested:
        if requested not in {"none", "moonshot", "openai", "anthropic"}:
            raise ValueError("TRAINEE_LLM_PROVIDER must be one of: none, moonshot, openai, anthropic")
        return requested  # type: ignore[return-value]

    if _settings_value("MOONSHOT_API_KEY", config_payload):
        return "moonshot"
    if _settings_value("ANTHROPIC_API_KEY", config_payload) and not _settings_value("OPENAI_API_KEY", config_payload):
        return "anthropic"
    if _settings_value("OPENAI_API_KEY", config_payload):
        return "openai"
    if _settings_value("ANTHROPIC_API_KEY", config_payload):
        return "anthropic"
    return "none"


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


def _settings_value(
    name: str,
    config_payload: Dict[str, Any],
    default: str | None = None,
) -> str | None:
    env_value = os.getenv(name)
    if env_value is not None:
        return env_value
    config_value = _config_value(name, config_payload)
    if config_value is not None:
        return str(config_value)
    return default


def _config_value(name: str, config_payload: Dict[str, Any]) -> Any:
    if name in {"TRAINEE_LLM_PROVIDER", "LLM_PROVIDER"}:
        return config_payload.get("llm_provider")
    if name == "TRAINEE_LLM_TIMEOUT_SEC":
        return config_payload.get("llm_timeout_sec")
    if name == "TRAINEE_LLM_TEMPERATURE":
        return config_payload.get("llm_temperature")
    if name == "TRAINEE_MAX_IMAGE_ANALYSES_PER_SESSION":
        return config_payload.get("max_image_analyses_per_session")
    if name in PROVIDER_CONFIG_KEYS:
        provider, key = PROVIDER_CONFIG_KEYS[name]
        return _provider_config_value(config_payload, provider, key)
    return None


def _active_provider_env_overrides() -> tuple[str, ...]:
    return tuple(name for name in PROVIDER_ENV_NAMES if os.getenv(name) is not None)


def _provider_config_value(config_payload: Dict[str, Any], provider: str, key: str) -> Any:
    provider_payload = config_payload.get(provider)
    if not isinstance(provider_payload, dict):
        return None
    return provider_payload.get(key)


def _config_bool(config_payload: Dict[str, Any], key: str, default: bool) -> bool:
    value = config_payload.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    return bool(value)


def _resolve_path(raw: str, repo_root: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _resolve_project_data_dir(repo_root: Path, project_root: Path | None, data_dir: Path | None) -> Path:
    if data_dir is not None:
        return data_dir.expanduser().resolve()
    env_data_dir = os.getenv("TRAINEE_DATA_DIR")
    if env_data_dir:
        return _resolve_path(env_data_dir, repo_root)
    if project_root is not None:
        return (project_root / ".trainee").resolve()
    return (Path.home() / ".trainee" / "runtime").expanduser().resolve()
