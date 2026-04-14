from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal

from dotenv import dotenv_values

LLMProvider = Literal["none", "openai", "anthropic"]


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    data_dir: Path
    database_path: Path
    artifacts_dir: Path
    template_dir: Path
    static_dir: Path
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


def load_settings(repo_root: Path | None = None) -> Settings:
    repo_root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    dotenv_path = repo_root / ".env"
    dotenv_payload = _read_dotenv(dotenv_path)

    data_dir = _resolve_path(_env_value("TRAINEE_DATA_DIR", dotenv_payload, str(repo_root / ".trainee")), repo_root)
    database_path = data_dir / "runtime.sqlite3"
    artifacts_dir = data_dir / "artifacts"
    return Settings(
        repo_root=repo_root,
        data_dir=data_dir,
        database_path=database_path,
        artifacts_dir=artifacts_dir,
        template_dir=Path(__file__).resolve().parent / "templates",
        static_dir=Path(__file__).resolve().parent / "static",
        dotenv_path=dotenv_path,
        llm_provider=_resolve_llm_provider(dotenv_payload),
        llm_timeout_sec=float(_env_value("TRAINEE_LLM_TIMEOUT_SEC", dotenv_payload, "30")),
        openai_api_key=_env_value("OPENAI_API_KEY", dotenv_payload),
        openai_base_url=_env_value("OPENAI_BASE_URL", dotenv_payload, "https://api.openai.com/v1"),
        openai_model=_env_value("OPENAI_MODEL", dotenv_payload, "gpt-4o-mini"),
        anthropic_api_key=_env_value("ANTHROPIC_API_KEY", dotenv_payload),
        anthropic_base_url=_env_value("ANTHROPIC_BASE_URL", dotenv_payload, "https://api.anthropic.com"),
        anthropic_model=_env_value("ANTHROPIC_MODEL", dotenv_payload, "claude-3-5-haiku-latest"),
        anthropic_version=_env_value("ANTHROPIC_VERSION", dotenv_payload, "2023-06-01"),
        anthropic_max_tokens=int(_env_value("ANTHROPIC_MAX_TOKENS", dotenv_payload, "1024")),
    )


def _resolve_llm_provider(dotenv_payload: Dict[str, str]) -> LLMProvider:
    requested = (_env_value("TRAINEE_LLM_PROVIDER", dotenv_payload) or _env_value("LLM_PROVIDER", dotenv_payload) or "").strip().lower()
    if requested:
        if requested not in {"none", "openai", "anthropic"}:
            raise ValueError("TRAINEE_LLM_PROVIDER must be one of: none, openai, anthropic")
        return requested  # type: ignore[return-value]

    if _env_value("ANTHROPIC_API_KEY", dotenv_payload) and not _env_value("OPENAI_API_KEY", dotenv_payload):
        return "anthropic"
    if _env_value("OPENAI_API_KEY", dotenv_payload):
        return "openai"
    if _env_value("ANTHROPIC_API_KEY", dotenv_payload):
        return "anthropic"
    return "none"


def _read_dotenv(dotenv_path: Path) -> Dict[str, str]:
    if not dotenv_path.exists():
        return {}
    payload = dotenv_values(dotenv_path)
    return {key: value for key, value in payload.items() if value is not None}


def _env_value(name: str, dotenv_payload: Dict[str, str], default: str | None = None) -> str | None:
    return os.getenv(name, dotenv_payload.get(name, default))


def _resolve_path(raw: str, repo_root: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()
