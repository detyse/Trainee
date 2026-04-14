from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

from trainee.context_builder import ContextBuilder
from trainee.decision import DecisionEngine
from trainee.events import EventBus
from trainee.executor import TrainingExecutor
from trainee.models import EventMessage, LoopSnapshot, ProjectBundle, ProjectContext, ProjectSpec, RoundRecord, RunSession, utc_now
from trainee.parsers import discover_wandb_summary, missing_required_metrics, parse_metrics_from_logs
from trainee.settings import Settings
from trainee.storage import Storage


class RuntimeService:
    def __init__(
        self,
        settings: Settings,
        storage: Storage,
        event_bus: EventBus,
        context_builder: Optional[ContextBuilder] = None,
        executor: Optional[TrainingExecutor] = None,
        decision_engine: Optional[DecisionEngine] = None,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.event_bus = event_bus
        self.context_builder = context_builder or ContextBuilder()
        self.executor = executor or TrainingExecutor()
        self.decision_engine = decision_engine or DecisionEngine(settings)
        self._loop_task: Optional[asyncio.Task[None]] = None
        self._task_lock = asyncio.Lock()

    def get_bundle(self) -> ProjectBundle:
        return ProjectBundle(
            spec=self.storage.get_project_spec(),
            context=self.storage.get_project_context(),
            loop=self.storage.get_loop_snapshot(),
        )

    async def register_project(self, spec: ProjectSpec) -> ProjectBundle:
        context = self.context_builder.build(spec)
        self.storage.save_project_spec(spec)
        self.storage.save_project_context(context)
        snapshot = self.storage.get_loop_snapshot()
        if not self.loop_is_running():
            snapshot.status = "ready"
            snapshot.current_session_id = None
            snapshot.active_round_id = None
            snapshot.current_round_index = 0
            snapshot.requested_stop = False
            snapshot.message = "Project registered and context built."
            self.storage.save_loop_snapshot(snapshot)
        await self._publish("project_registered", {"project_root": spec.project_root})
        return self.get_bundle()

    async def update_project_context(self, context: ProjectContext) -> ProjectBundle:
        self.storage.save_project_context(context)
        await self._publish("context_updated", {"warnings": context.warnings})
        return self.get_bundle()

    async def start_loop(self) -> LoopSnapshot:
        async with self._task_lock:
            if self._loop_task and not self._loop_task.done():
                return self.storage.get_loop_snapshot()
            spec = self.storage.get_project_spec()
            if spec is None:
                raise ValueError("register a project before starting the loop")
            session = self.storage.create_session(RunSession(status="running"))
            snapshot = LoopSnapshot(
                status="building_context",
                current_session_id=session.id,
                current_round_index=0,
                requested_stop=False,
                message="Starting agent loop.",
            )
            self.storage.save_loop_snapshot(snapshot)
            self._loop_task = asyncio.create_task(self._run_session(session.id), name=f"agent-loop-{session.id}")
            self._loop_task.add_done_callback(self._clear_task_reference)
            await self._publish("loop_started", {"session_id": session.id})
            return snapshot

    async def stop_loop(self) -> LoopSnapshot:
        snapshot = self.storage.get_loop_snapshot()
        if snapshot.current_session_id is None:
            snapshot.status = "stopped"
            snapshot.message = "No active session."
            self.storage.save_loop_snapshot(snapshot)
            return snapshot
        session = self.storage.get_session(snapshot.current_session_id)
        if session is None:
            return snapshot
        session.requested_stop = True
        session.stop_reason = "User requested stop after the current round."
        self.storage.update_session(session)
        snapshot.requested_stop = True
        snapshot.message = "Stop requested; the runtime will not start another round."
        self.storage.save_loop_snapshot(snapshot)
        await self._publish("loop_stop_requested", {"session_id": session.id})
        return snapshot

    def loop_is_running(self) -> bool:
        return self._loop_task is not None and not self._loop_task.done()

    def dashboard_payload(self, selected_run_id: Optional[int] = None) -> Dict[str, Any]:
        bundle = self.get_bundle()
        sessions = self.storage.list_sessions()
        rounds = self.storage.list_rounds()
        selected_run = None
        if selected_run_id is not None:
            selected_run = self.storage.get_round(selected_run_id)
        elif rounds:
            selected_run = rounds[0]
        selected_log_excerpt = self._read_log_tail(selected_run.log_paths[0]) if selected_run and selected_run.log_paths else ""
        return {
            "bundle": bundle,
            "spec": bundle.spec,
            "context": bundle.context,
            "loop": bundle.loop,
            "sessions": sessions,
            "rounds": rounds,
            "selected_run": selected_run,
            "selected_log_excerpt": selected_log_excerpt,
            "selected_decision_json": self._pretty_json(selected_run.agent_decision.model_dump(mode="json")) if selected_run and selected_run.agent_decision else "",
            "tunable_params_json": self._pretty_json([item.model_dump(mode="json") for item in bundle.spec.tunable_params] if bundle.spec else []),
            "metric_specs_json": self._pretty_json([item.model_dump(mode="json") for item in bundle.spec.metric_specs] if bundle.spec else []),
        }

    async def _run_session(self, session_id: int) -> None:
        try:
            spec = self.storage.get_project_spec()
            if spec is None:
                raise ValueError("project spec disappeared before the loop started")
            context = self.storage.get_project_context()
            if context is None:
                context = self.context_builder.build(spec)
                self.storage.save_project_context(context)

            snapshot = self.storage.get_loop_snapshot()
            snapshot.status = "ready"
            snapshot.message = "Project context is ready."
            self.storage.save_loop_snapshot(snapshot)

            current_params = self._initial_params(spec)
            round_index = 1

            while True:
                session = self.storage.get_session(session_id)
                if session is None:
                    raise ValueError(f"session {session_id} disappeared")
                session.current_round = round_index
                self.storage.update_session(session)

                snapshot = self.storage.get_loop_snapshot()
                snapshot.status = "starting_next_round"
                snapshot.current_session_id = session_id
                snapshot.current_round_index = round_index
                snapshot.message = f"Starting round {round_index}."
                self.storage.save_loop_snapshot(snapshot)

                resolved_command = self.executor.render_command(spec, current_params)
                round_record = self.storage.create_round(
                    RoundRecord(
                        session_id=session_id,
                        round_index=round_index,
                        resolved_command=resolved_command,
                        param_values=current_params,
                        status="running",
                    )
                )
                snapshot.active_round_id = round_record.id
                snapshot.status = "running_round"
                snapshot.message = f"Round {round_index} is running."
                self.storage.save_loop_snapshot(snapshot)
                await self._publish("round_started", {"session_id": session_id, "round_id": round_record.id, "round_index": round_index})

                result = await self.executor.run_round(
                    spec=spec,
                    session_id=session_id,
                    round_index=round_index,
                    param_values=current_params,
                    artifacts_dir=self.settings.artifacts_dir,
                    heartbeat=lambda payload: self._handle_heartbeat(session_id, round_record.id or 0, round_index, payload),
                )

                round_record.resolved_command = result.resolved_command
                round_record.log_paths = result.log_paths
                round_record.exit_code = result.exit_code
                round_record.end_time = result.ended_at
                round_record.wandb_run_url = result.wandb_run_url

                snapshot = self.storage.get_loop_snapshot()
                snapshot.last_signal_at = result.last_signal_at
                snapshot.last_heartbeat_at = utc_now()
                snapshot.status = "evaluating"
                snapshot.message = f"Evaluating round {round_index}."
                self.storage.save_loop_snapshot(snapshot)

                log_text = self._read_log(result.internal_log_path)
                wandb_summary_path, wandb_summary = discover_wandb_summary(spec, round_record.start_time)
                metrics = parse_metrics_from_logs(log_text, spec, wandb_summary)
                if wandb_summary_path and wandb_summary_path not in round_record.log_paths:
                    round_record.log_paths.append(wandb_summary_path)
                round_record.metrics = metrics

                if result.stalled:
                    round_record.status = "stalled"
                    round_record.error = "Heartbeat timed out before training produced a fresh signal."
                elif (result.exit_code or 0) != 0:
                    round_record.status = "failed"
                    round_record.error = f"Training process exited with code {result.exit_code}."
                else:
                    missing = missing_required_metrics(spec, metrics)
                    if missing:
                        round_record.status = "completed_without_metrics"
                        round_record.error = "Missing required metrics: " + ", ".join(missing)
                    else:
                        round_record.status = "completed"
                self.storage.update_round(round_record)
                await self._publish(
                    "round_finished",
                    {
                        "session_id": session_id,
                        "round_id": round_record.id,
                        "round_index": round_index,
                        "status": round_record.status,
                        "metrics": round_record.metrics,
                    },
                )

                if round_record.status != "completed":
                    await self._finish_session(session_id, "failed", round_record.error or round_record.status)
                    return

                session = self.storage.get_session(session_id)
                if session is None:
                    raise ValueError("session disappeared during evaluation")
                if session.requested_stop:
                    await self._finish_session(session_id, "stopped", session.stop_reason or "Stop requested.")
                    return
                if round_index >= spec.max_rounds:
                    await self._finish_session(session_id, "stopped", "Reached max_rounds.")
                    return

                history = list(reversed(self.storage.list_rounds(session_id)))
                snapshot = self.storage.get_loop_snapshot()
                snapshot.status = "deciding"
                snapshot.message = f"Generating decision for round {round_index + 1}."
                self.storage.save_loop_snapshot(snapshot)

                decision = await self.decision_engine.decide(spec, context, history, current_params)
                round_record.agent_decision = decision
                self.storage.update_round(round_record)
                await self._publish(
                    "decision_made",
                    {
                        "session_id": session_id,
                        "round_id": round_record.id,
                        "action": decision.action,
                        "reason": decision.reason,
                    },
                )

                if decision.action == "stop":
                    await self._finish_session(session_id, "stopped", decision.reason)
                    return

                current_params = spec.merge_param_values(decision.next_params, base=current_params)
                round_index += 1
        except Exception as exc:
            await self._finish_session(session_id, "failed", str(exc))

    async def _handle_heartbeat(self, session_id: int, round_id: int, round_index: int, payload: Dict[str, Any]) -> None:
        snapshot = self.storage.get_loop_snapshot()
        snapshot.current_session_id = session_id
        snapshot.active_round_id = round_id
        snapshot.current_round_index = round_index
        snapshot.last_heartbeat_at = utc_now()
        snapshot.last_signal_at = payload.get("last_signal_at") or snapshot.last_signal_at
        if payload.get("stalled"):
            snapshot.status = "waiting_signal"
            snapshot.message = "Heartbeat detected a stall candidate; waiting for process completion."
        self.storage.save_loop_snapshot(snapshot)
        await self._publish("heartbeat", payload)

    async def _finish_session(self, session_id: int, status: str, message: str) -> None:
        session = self.storage.get_session(session_id)
        if session is not None:
            session.status = status
            session.stop_reason = message
            session.ended_at = utc_now()
            self.storage.update_session(session)

        snapshot = self.storage.get_loop_snapshot()
        snapshot.status = status
        snapshot.current_session_id = session_id
        snapshot.active_round_id = None
        snapshot.requested_stop = False
        snapshot.message = message
        self.storage.save_loop_snapshot(snapshot)
        await self._publish("loop_finished", {"session_id": session_id, "status": status, "message": message})

    def _initial_params(self, spec: ProjectSpec) -> Dict[str, Any]:
        latest_round = self.storage.get_latest_round()
        base = latest_round.param_values if latest_round and latest_round.status == "completed" else None
        return spec.merge_param_values(base=base)

    async def _publish(self, event_type: str, payload: Dict[str, Any]) -> None:
        await self.event_bus.publish(EventMessage(event_type=event_type, payload=payload))

    def _read_log(self, path: str) -> str:
        try:
            return Path(path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ""

    def _read_log_tail(self, path: str, line_count: int = 20) -> str:
        try:
            lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-line_count:])

    def _pretty_json(self, payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _clear_task_reference(self, task: asyncio.Task[None]) -> None:
        if self._loop_task is task:
            self._loop_task = None
