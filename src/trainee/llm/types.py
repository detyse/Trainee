from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ProviderCompletion:
    content: str
    raw_response_body: str
    http_status: int
    request_id: Optional[str] = None
    usage: Optional[dict[str, Any]] = None
    finish_reason: Optional[str] = None


class ProviderCallError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: str,
        http_status: Optional[int] = None,
        request_id: Optional[str] = None,
        raw_response_body: Optional[str] = None,
        error_body: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.http_status = http_status
        self.request_id = request_id
        self.raw_response_body = raw_response_body
        self.error_body = error_body


@dataclass(frozen=True)
class ProviderAttempt:
    provider: str
    model: str
    ok: bool
    status: str
    http_status: Optional[int] = None
    request_id: Optional[str] = None
    error_message: str = ""

    def as_trace_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "ok": self.ok,
            "status": self.status,
            "http_status": self.http_status,
            "request_id": self.request_id,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class ProviderDispatchResult:
    provider: str
    model: str
    completion: ProviderCompletion
    attempts: list[dict[str, Any]]


class ProviderDispatchError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        attempts: list[dict[str, Any]],
        last_error: ProviderCallError | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error
