from __future__ import annotations

import asyncio
import glob
import os
import shlex
import signal
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from trainee.models import ProjectSpec, utc_now
from trainee.parsers import extract_wandb_url
from trainee.security import build_secure_command, project_trainee_dir


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
    terminated_reason: Optional[str] = None


class TrainingExecutor:
    async def run_round(
        self,
        spec: ProjectSpec,
        session_id: int,
        round_index: int,
        param_values: Dict[str, Any],
        artifacts_dir: Path,
        heartbeat: HeartbeatReporter,
        stop_event: Optional[asyncio.Event] = None,
    ) -> ExecutionResult:
        started_at = utc_now()
        self.validate_paths(spec, artifacts_dir)
        session_dir = artifacts_dir / f"session-{session_id:04d}"
        session_dir.mkdir(parents=True, exist_ok=True)
        internal_log_path = session_dir / f"round-{round_index:04d}.log"
        resolved_command = self.render_command(spec, param_values)
        working_dir = Path(spec.working_dir).expanduser().resolve()
        external_log_paths = self.resolve_log_paths(spec)
        secure_command = build_secure_command(
            project_root=Path(spec.project_root),
            working_dir=working_dir,
            command=resolved_command,
            security_mode=spec.security_mode,
        )

        state = _ExecutionState()
        process = await asyncio.create_subprocess_exec(
            *secure_command.argv,
            cwd=str(secure_command.cwd) if secure_command.cwd is not None else None,
            env=secure_command.env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
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
        wait_task = asyncio.create_task(process.wait(), name=f"round-{round_index}-wait")
        stop_task = (
            asyncio.create_task(stop_event.wait(), name=f"round-{round_index}-stop")
            if stop_event is not None
            else None
        )

        terminated_reason = await self._wait_for_completion(
            process=process,
            spec=spec,
            state=state,
            wait_task=wait_task,
            stop_task=stop_task,
        )
        exit_code = await wait_task

        if stop_task is not None:
            stop_task.cancel()
            with suppress(asyncio.CancelledError):
                await stop_task
        with suppress(asyncio.CancelledError):
            await reader_task
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
            terminated_reason=terminated_reason,
        )

    async def _wait_for_completion(
        self,
        *,
        process: asyncio.subprocess.Process,
        spec: ProjectSpec,
        state: "_ExecutionState",
        wait_task: asyncio.Task[int],
        stop_task: Optional[asyncio.Task[bool]],
    ) -> Optional[str]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + spec.round_timeout_sec if spec.round_timeout_sec is not None else None

        while not wait_task.done():
            waiters: List[asyncio.Task[Any]] = [wait_task]
            if stop_task is not None:
                waiters.append(stop_task)
            done, _ = await asyncio.wait(waiters, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
            if wait_task in done:
                return None
            if stop_task is not None and stop_task in done:
                await self.kill_process(process)
                return "stop_requested"
            if deadline is not None and loop.time() >= deadline:
                await self.kill_process(process)
                return "timeout"
            if spec.kill_on_stall and state.stalled:
                await self.kill_process(process)
                return "stall"
        return None

    async def kill_process(self, process: asyncio.subprocess.Process, grace_sec: float = 10.0) -> None:
        if process.returncode is not None:
            return

        self._signal_process_group(process, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=grace_sec)
            return
        except asyncio.TimeoutError:
            pass

        self._signal_process_group(process, signal.SIGKILL)
        await process.wait()

    def _signal_process_group(self, process: asyncio.subprocess.Process, sig: signal.Signals) -> None:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, sig)

    def render_command(self, spec: ProjectSpec, param_values: Dict[str, Any]) -> str:
        validated = spec.merge_param_values(param_values)
        extra_args = self._render_cli_args(spec, validated)
        template_vars = {
            "project_root": shlex.quote(str(Path(spec.project_root).expanduser().resolve())),
            "working_dir": shlex.quote(str(Path(spec.working_dir).expanduser().resolve())),
            "trainee_dir": shlex.quote(str(project_trainee_dir(Path(spec.project_root)))),
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

    def validate_paths(self, spec: ProjectSpec, artifacts_dir: Path) -> None:
        project_root = Path(spec.project_root).expanduser().resolve()
        working_dir = Path(spec.working_dir).expanduser().resolve()

        self._ensure_within(working_dir, project_root, "working_dir")
        for field_name, raw_paths in (("data_paths", spec.data_paths), ("log_paths", spec.log_paths)):
            for raw_path in raw_paths:
                candidate = self._configured_path(raw_path, working_dir)
                self._ensure_within(candidate, project_root, field_name)
                if spec.security_mode == "guarded" and field_name == "log_paths":
                    self._ensure_within(candidate, project_trainee_dir(project_root), "log_paths")

    def _configured_path(self, raw_path: str, working_dir: Path) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = working_dir / path
        return path.resolve()

    def _ensure_within(self, path: Path, root: Path, field_name: str) -> None:
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{field_name} must stay within project_root: {path}") from exc

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
