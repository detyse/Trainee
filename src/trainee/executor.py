from __future__ import annotations

import asyncio
import glob
import shlex
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from trainee.models import ProjectSpec, utc_now
from trainee.parsers import extract_wandb_url


HeartbeatReporter = Callable[[Dict[str, Any]], Awaitable[None]]


@dataclass
class ExecutionResult:
    resolved_command: str
    internal_log_path: str
    log_paths: List[str]
    exit_code: Optional[int]
    started_at: str
    ended_at: str
    stalled: bool
    wandb_run_url: Optional[str]
    last_signal_at: Optional[str]


class TrainingExecutor:
    async def run_round(
        self,
        spec: ProjectSpec,
        session_id: int,
        round_index: int,
        param_values: Dict[str, Any],
        artifacts_dir: Path,
        heartbeat: HeartbeatReporter,
    ) -> ExecutionResult:
        started_at = utc_now()
        session_dir = artifacts_dir / f"session-{session_id:04d}"
        session_dir.mkdir(parents=True, exist_ok=True)
        internal_log_path = session_dir / f"round-{round_index:04d}.log"
        resolved_command = self.render_command(spec, param_values)
        working_dir = Path(spec.working_dir).expanduser().resolve()
        external_log_paths = self.resolve_log_paths(spec)

        state = _ExecutionState()
        process = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-lc",
            resolved_command,
            cwd=str(working_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        await heartbeat(
            {
                "status": "running",
                "session_id": session_id,
                "round_index": round_index,
                "command": resolved_command,
                "started_at": started_at,
            }
        )

        state.last_signal_at = started_at
        reader_task = asyncio.create_task(
            self._stream_output(process, internal_log_path, state),
            name=f"round-{round_index}-reader",
        )
        heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(spec, session_id, round_index, external_log_paths, state, heartbeat),
            name=f"round-{round_index}-heartbeat",
        )

        await reader_task
        exit_code = await process.wait()
        heartbeat_task.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat_task

        ended_at = utc_now()
        log_paths = [str(internal_log_path)] + [str(path) for path in external_log_paths]
        return ExecutionResult(
            resolved_command=resolved_command,
            internal_log_path=str(internal_log_path),
            log_paths=log_paths,
            exit_code=exit_code,
            started_at=started_at,
            ended_at=ended_at,
            stalled=state.stalled,
            wandb_run_url=state.wandb_run_url,
            last_signal_at=state.last_signal_at,
        )

    def render_command(self, spec: ProjectSpec, param_values: Dict[str, Any]) -> str:
        validated = spec.merge_param_values(param_values)
        extra_args = self._render_cli_args(spec, validated)
        template_vars = {
            "project_root": shlex.quote(str(Path(spec.project_root).expanduser().resolve())),
            "working_dir": shlex.quote(str(Path(spec.working_dir).expanduser().resolve())),
            "extra_args": extra_args,
        }
        if "{extra_args}" in spec.launcher_template:
            return spec.launcher_template.format_map(template_vars).strip()
        rendered = spec.launcher_template.format_map({k: v for k, v in template_vars.items() if k != "extra_args"}).strip()
        return f"{rendered} {extra_args}".strip()

    def resolve_log_paths(self, spec: ProjectSpec) -> List[Path]:
        resolved: List[Path] = []
        working_dir = Path(spec.working_dir).expanduser().resolve()
        for raw_path in spec.log_paths:
            base_candidate = Path(raw_path).expanduser()
            if not base_candidate.is_absolute():
                base_candidate = (working_dir / raw_path).resolve()
            matches = glob.glob(str(base_candidate), recursive=True)
            if matches:
                resolved.extend(Path(item).resolve() for item in matches)
            else:
                resolved.append(base_candidate.resolve())
        deduped = []
        seen = set()
        for path in resolved:
            marker = str(path)
            if marker in seen:
                continue
            seen.add(marker)
            deduped.append(path)
        return deduped

    def _render_cli_args(self, spec: ProjectSpec, param_values: Dict[str, Any]) -> str:
        parts: List[str] = []
        for param in spec.tunable_params:
            if param.name not in param_values:
                continue
            value = param.normalize_value(param_values[param.name])
            if param.type == "bool":
                if value:
                    parts.append(param.flag)
                continue
            parts.append(param.flag)
            parts.append(shlex.quote(str(value)))
        return " ".join(parts)

    async def _stream_output(self, process: asyncio.subprocess.Process, log_path: Path, state: "_ExecutionState") -> None:
        assert process.stdout is not None
        with log_path.open("a", encoding="utf-8") as handle:
            while True:
                chunk = await process.stdout.readline()
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="ignore")
                handle.write(text)
                handle.flush()
                state.last_output_line = text.strip()
                state.last_signal_at = utc_now()
                if state.wandb_run_url is None:
                    state.wandb_run_url = extract_wandb_url(text)

    async def _heartbeat_loop(
        self,
        spec: ProjectSpec,
        session_id: int,
        round_index: int,
        external_log_paths: List[Path],
        state: "_ExecutionState",
        heartbeat: HeartbeatReporter,
    ) -> None:
        while True:
            await asyncio.sleep(spec.heartbeat_interval_sec)
            external_signal = self._latest_external_signal(external_log_paths)
            last_signal = self._latest_signal(state.last_signal_at, external_signal)
            is_stalled = False
            if last_signal:
                last_signal_dt = datetime.fromisoformat(last_signal)
                age = (datetime.now(timezone.utc) - last_signal_dt).total_seconds()
                is_stalled = age > spec.stall_timeout_sec
            if is_stalled:
                state.stalled = True
            await heartbeat(
                {
                    "status": "heartbeat",
                    "session_id": session_id,
                    "round_index": round_index,
                    "last_signal_at": last_signal,
                    "last_output_line": state.last_output_line,
                    "stalled": is_stalled,
                    "wandb_run_url": state.wandb_run_url,
                }
            )

    def _latest_external_signal(self, log_paths: List[Path]) -> Optional[str]:
        latest_timestamp = None
        for path in log_paths:
            if not path.exists():
                continue
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
            latest_timestamp = self._latest_signal(latest_timestamp, mtime)
        return latest_timestamp

    def _latest_signal(self, current: Optional[str], candidate: Optional[str]) -> Optional[str]:
        if current is None:
            return candidate
        if candidate is None:
            return current
        return current if datetime.fromisoformat(current) >= datetime.fromisoformat(candidate) else candidate


@dataclass
class _ExecutionState:
    last_signal_at: Optional[str] = None
    last_output_line: str = ""
    wandb_run_url: Optional[str] = None
    stalled: bool = False
