from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

from trainee.context_builder import ContextBuilder
from trainee.decision import DecisionEngine
from trainee.events import EventBus
from trainee.executor import TrainingExecutor
from trainee.ledger import LedgerExporter
from trainee.logging import get_logger
from trainee.models import EventMessage, LoopSnapshot, ProjectBundle, ProjectContext, ProjectSpec, PromptPreset, PromptPreview, RoundRecord, RunSession, utc_now
from trainee.parsers import discover_wandb_summary, missing_required_metrics, parse_metrics_from_sources
from trainee.prompt_documents import PromptDocumentLoader
from trainee.project_config import ProjectConfig, compile_project_spec, default_project_config, load_project_config, project_config_path
from trainee.providers import provider_settings_payload
from trainee.reporter import ReportGenerator
from trainee.research_state import ResearchStateBuilder
from trainee.settings import Settings
from trainee.storage import Storage

logger = get_logger(__name__)


class RuntimeService:
    def __init__(
        self,
        settings: Settings,
        storage: Storage,
        event_bus: EventBus,
        context_builder: Optional[ContextBuilder] = None,
        executor: Optional[TrainingExecutor] = None,
        decision_engine: Optional[DecisionEngine] = None,
        research_state_builder: Optional[ResearchStateBuilder] = None,
        prompt_document_loader: Optional[PromptDocumentLoader] = None,
        ledger_exporter: Optional[LedgerExporter] = None,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.event_bus = event_bus
        self.context_builder = context_builder or ContextBuilder()
        self.executor = executor or TrainingExecutor()
        self.decision_engine = decision_engine or DecisionEngine(settings)
        self.research_state_builder = research_state_builder or ResearchStateBuilder()
        self.prompt_document_loader = prompt_document_loader or PromptDocumentLoader()
        self.ledger_exporter = ledger_exporter or LedgerExporter(self.research_state_builder)
        self._loop_task: Optional[asyncio.Task[None]] = None
        self._active_stop_event: Optional[asyncio.Event] = None
        self._task_lock = asyncio.Lock()
        self._launch_context_preview: Optional[ProjectContext] = None

    def get_bundle(self) -> ProjectBundle:
        return ProjectBundle(
            spec=self.storage.get_project_spec(),
            context=self.storage.get_project_context(),
            loop=self.storage.get_loop_snapshot(),
        )

    def update_settings(self, settings: Settings) -> None:
        self.settings = settings
        self.decision_engine = DecisionEngine(settings)
        self._launch_context_preview = None

    def prepare_project_registration(self, spec: ProjectSpec) -> ProjectContext:
        self.executor.validate_paths(spec, self.settings.artifacts_dir)
        return self.context_builder.build(spec)

    async def register_project(
        self,
        spec: ProjectSpec,
        *,
        context: Optional[ProjectContext] = None,
    ) -> ProjectBundle:
        context = context or self.prepare_project_registration(spec)
        snapshot = self.storage.get_loop_snapshot()
        updated_snapshot: Optional[LoopSnapshot] = None
        if not self.loop_is_running():
            snapshot.status = "ready"
            snapshot.current_session_id = None
            snapshot.active_round_id = None
            snapshot.current_round_index = 0
            snapshot.requested_stop = False
            snapshot.message = "Project registered and context built."
            updated_snapshot = snapshot
        self.storage.save_project_registration(spec, context, updated_snapshot)
        await self._publish("project_registered", {"project_root": spec.project_root})
        return self.get_bundle()

    async def update_project_context(self, context: ProjectContext) -> ProjectBundle:
        self.storage.save_project_context(context)
        await self._publish("context_updated", {"warnings": context.warnings})
        return self.get_bundle()

    async def save_prompt_preset(
        self,
        name: str,
        metric_prompt: str,
        tuning_prompt: str,
        project_root: str,
        preset_id: Optional[str] = None,
    ) -> PromptPreset:
        preset_payload: Dict[str, Any] = {
            "project_root": project_root,
            "name": name,
            "metric_prompt": metric_prompt,
            "tuning_prompt": tuning_prompt,
        }
        if preset_id:
            preset_payload["id"] = preset_id
        preset = PromptPreset.model_validate(preset_payload)
        saved = self.storage.upsert_prompt_preset(preset)
        await self._publish("prompt_preset_saved", {"preset_id": saved.id, "project_root": saved.project_root})
        return saved

    async def start_loop(self, resume_session_id: Optional[int] = None) -> LoopSnapshot:
        async with self._task_lock:
            if self._loop_task and not self._loop_task.done():
                return self.storage.get_loop_snapshot()
            session, start_round, initial_params = self._create_session_for_start(resume_session_id)
            snapshot = LoopSnapshot(
                status="building_context",
                current_session_id=session.id,
                current_round_index=start_round - 1,
                requested_stop=False,
                message="Starting agent loop.",
            )
            self.storage.save_loop_snapshot(snapshot)
            self._loop_task = asyncio.create_task(
                self._run_session(session.id, start_round=start_round, initial_params=initial_params),
                name=f"agent-loop-{session.id}",
            )
            self._loop_task.add_done_callback(self._clear_task_reference)
            await self._publish(
                "loop_started",
                {"session_id": session.id, "resumed_from": session.resumed_from, "start_round": start_round},
            )
            return snapshot

    async def stop_loop(self, force: bool = False) -> LoopSnapshot:
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
        session.stop_reason = "User requested stop immediately." if force else "User requested stop after the current round."
        self.storage.update_session(session)
        snapshot.requested_stop = True
        if force and self._active_stop_event is not None:
            self._active_stop_event.set()
            snapshot.message = "Force stop requested; terminating the active round."
        else:
            snapshot.message = "Stop requested; the runtime will not start another round."
        self.storage.save_loop_snapshot(snapshot)
        await self._publish("loop_stop_requested", {"session_id": session.id, "force": force})
        return snapshot

    def loop_is_running(self) -> bool:
        return self._loop_task is not None and not self._loop_task.done()

    def reserve_image_analysis(self, session_id: Optional[int] = None) -> Optional[Dict[str, int]]:
        resolved_session_id = self._resolve_image_analysis_session_id(session_id)
        if resolved_session_id is None:
            return None
        used, limit = self.storage.reserve_session_image_analysis(
            resolved_session_id,
            self.settings.max_image_analyses_per_session,
        )
        return {"session_id": resolved_session_id, "used": used, "limit": limit}

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
        prompt_preview, prompt_preview_label = self._prompt_preview_for_dashboard(selected_run, rounds)
        prompt_presets = self.storage.list_prompt_presets(bundle.spec.project_root if bundle.spec else None)
        context = bundle.context
        context_is_preview = False
        if bundle.spec is None and context is None and self.settings.project_root is not None:
            context = self._launch_project_context_preview()
            context_is_preview = True
        project_config = self._dashboard_project_config(bundle.spec)
        return {
            "bundle": bundle,
            "spec": bundle.spec,
            "context": context,
            "context_is_preview": context_is_preview,
            "project_config": project_config,
            "project_config_path": str(project_config_path(bundle.spec.project_root)) if bundle.spec else "",
            "loop": bundle.loop,
            "sessions": sessions,
            "rounds": rounds,
            "selected_run": selected_run,
            "selected_log_excerpt": selected_log_excerpt,
            "selected_decision_json": self._pretty_json(selected_run.agent_decision.model_dump(mode="json")) if selected_run and selected_run.agent_decision else "",
            "selected_agent_trace_json": self._pretty_json(selected_run.agent_trace.model_dump(mode="json")) if selected_run and selected_run.agent_trace else "",
            "prompt_preview": prompt_preview,
            "prompt_preview_label": prompt_preview_label,
            "prompt_payload_json": self._pretty_json(prompt_preview.payload) if prompt_preview else "",
            "prompt_static_context_json": self._pretty_json(prompt_preview.static_context_json) if prompt_preview else "",
            "prompt_dynamic_state_json": self._pretty_json(prompt_preview.dynamic_state_json) if prompt_preview else "",
            "prompt_presets": prompt_presets,
            "prompt_presets_payload": [item.model_dump(mode="json") for item in prompt_presets],
            "runtime_settings": self._runtime_settings_payload(),
            "tunable_params_json": self._pretty_json([item.model_dump(mode="json") for item in bundle.spec.tunable_params] if bundle.spec else []),
            "signal_sources_json": self._pretty_json([item.model_dump(mode="json") for item in bundle.spec.signal_sources] if bundle.spec else []),
            "metric_specs_json": self._pretty_json([item.model_dump(mode="json") for item in bundle.spec.metric_specs] if bundle.spec else []),
        }

    def _dashboard_project_config(self, spec: Optional[ProjectSpec]) -> Optional[ProjectConfig]:
        root: Optional[Path] = Path(spec.project_root) if spec is not None else self.settings.project_root
        if root is None:
            return None
        try:
            return load_project_config(root)
        except (OSError, ValueError):
            return default_project_config(root)

    def _runtime_settings_payload(self) -> Dict[str, Any]:
        project_root = self.settings.project_root
        return {
            "launch_project_root": str(project_root) if project_root else "",
            "launch_launcher_template": self._default_launcher_template(),
            "data_dir": str(self.settings.data_dir),
            "project_data_dir": str(self.settings.project_data_dir),
            "database_path": str(self.settings.database_path),
            "artifacts_dir": str(self.settings.artifacts_dir),
            "config_path": str(self.settings.config_path),
            "global_config_path": str(self.settings.global_config_path),
            "system_prompt": self.settings.system_prompt,
            "loop_running": self.loop_is_running(),
            **provider_settings_payload(self.settings),
        }

    def _resolve_image_analysis_session_id(self, session_id: Optional[int]) -> Optional[int]:
        if session_id is not None:
            if self.storage.get_session(session_id) is None:
                raise ValueError(f"session {session_id} not found")
            return session_id

        snapshot = self.storage.get_loop_snapshot()
        if snapshot.current_session_id is not None:
            return snapshot.current_session_id

        latest_session = self.storage.get_latest_session()
        return latest_session.id if latest_session else None

    def _launch_project_context_preview(self) -> ProjectContext:
        if self._launch_context_preview is None:
            self._launch_context_preview = self.context_builder.build(self._launch_project_spec())
        return self._launch_context_preview

    def _launch_project_spec(self) -> ProjectSpec:
        if self.settings.project_root is None:
            raise ValueError("no launch project is configured")
        try:
            config = load_project_config(self.settings.project_root)
        except (OSError, ValueError):
            config = default_project_config(self.settings.project_root)
        return compile_project_spec(self.settings.project_root, config)

    def _default_launcher_template(self) -> str:
        if self.settings.project_root is None:
            return ""
        for relative_path in ("train.py", "main.py", "run.py"):
            if (self.settings.project_root / relative_path).is_file():
                return f"python {{project_root}}/{relative_path} {{extra_args}}"
        return ""

    def _create_session_for_start(
        self,
        resume_session_id: Optional[int],
    ) -> tuple[RunSession, int, Optional[Dict[str, Any]]]:
        if resume_session_id is None:
            spec = self.storage.get_project_spec()
            if spec is None:
                raise ValueError("register a project before starting the loop")
            context = self.storage.get_project_context()
            if context is None:
                context = self.context_builder.build(spec)
                self.storage.save_project_context(context)
            self.executor.validate_paths(spec, self.settings.artifacts_dir)
            session = self.storage.create_session(
                RunSession(status="running", project_spec=spec, project_context=context)
            )
            return session, 1, None

        source = self.storage.get_session(resume_session_id)
        if source is None:
            raise ValueError(f"session {resume_session_id} not found")
        if source.project_spec is None or source.project_context is None:
            raise ValueError(f"session {resume_session_id} cannot be resumed because it has no saved project snapshot")
        self.executor.validate_paths(source.project_spec, self.settings.artifacts_dir)
        research_rounds = self.storage.list_research_rounds(resume_session_id)
        last_completed = next(
            (item for item in reversed(research_rounds) if item.status == "completed"),
            None,
        )
        start_round = (last_completed.round_index + 1) if last_completed else 1
        initial_params = last_completed.param_values if last_completed else None
        session = self.storage.create_session(
            RunSession(
                status="running",
                resumed_from=resume_session_id,
                project_spec=source.project_spec,
                project_context=source.project_context,
            )
        )
        return session, start_round, initial_params

    # here is the main loop function
    async def _run_session(
        self,
        session_id: int,
        start_round: int = 1,
        initial_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            session_snapshot = self.storage.get_session(session_id)
            if session_snapshot is None:
                raise ValueError(f"session {session_id} disappeared")
            spec = session_snapshot.project_spec or self.storage.get_project_spec()
            if spec is None:
                raise ValueError("project spec disappeared before the loop started")
            context = session_snapshot.project_context or self.storage.get_project_context()
            if context is None:
                context = self.context_builder.build(spec)
                self.storage.save_project_context(context)

            snapshot = self.storage.get_loop_snapshot()
            snapshot.status = "ready"
            snapshot.message = "Project context is ready."
            self.storage.save_loop_snapshot(snapshot)

            current_params = spec.merge_param_values(base=initial_params) if initial_params else self._initial_params(spec)
            round_index = start_round

            if round_index > spec.max_rounds:
                await self._finish_session(session_id, "stopped", "Reached max_rounds.")
                return

            # !! agent loop here
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

                resolved_command = self.executor.render_command(
                    spec,
                    current_params,
                    session_id=session_id,
                    round_index=round_index,
                )
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

                stop_event = asyncio.Event()
                self._active_stop_event = stop_event
                try:
                    result = await self.executor.run_round(
                        spec=spec,
                        session_id=session_id,
                        round_index=round_index,
                        param_values=current_params,
                        artifacts_dir=self.settings.artifacts_dir,
                        heartbeat=lambda payload: self._handle_heartbeat(session_id, round_record.id or 0, round_index, payload),
                        stop_event=stop_event,
                    )
                finally:
                    if self._active_stop_event is stop_event:
                        self._active_stop_event = None

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

                wandb_summary_path, wandb_summary = discover_wandb_summary(spec, round_record.start_time)
                metrics, metric_log_paths = parse_metrics_from_sources(result.internal_log_path, spec, wandb_summary)
                for path in metric_log_paths:
                    if path not in round_record.log_paths:
                        round_record.log_paths.append(path)
                if wandb_summary_path and wandb_summary_path not in round_record.log_paths:
                    round_record.log_paths.append(wandb_summary_path)
                round_record.metrics = metrics

                if result.terminated_reason == "stop_requested":
                    round_record.status = "stopped"
                    round_record.error = "Training process was stopped by user request."
                elif result.terminated_reason == "timeout":
                    round_record.status = "timeout"
                    round_record.error = "Training process exceeded round_timeout_sec and was terminated."
                elif result.stalled:
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
                self._export_session_artifacts(session_id, spec)
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

                if round_record.status == "stopped":
                    await self._finish_session(session_id, "stopped", round_record.error or "Stopped.")
                    return
                if round_record.status != "completed":
                    await self._finish_session(session_id, "failed", round_record.error or round_record.status)
                    return

                session = self.storage.get_session(session_id)
                if session is None:
                    raise ValueError("session disappeared during evaluation")
                if session.requested_stop:
                    await self._finish_session(session_id, "stopped", session.stop_reason or "Stop requested.")
                    return

                history = self.storage.list_research_rounds(session_id)
                research_state = self.research_state_builder.build(spec, history)
                prompt_documents = self.prompt_document_loader.load(spec.project_root)
                snapshot = self.storage.get_loop_snapshot()
                snapshot.status = "deciding"
                snapshot.message = f"Evaluating round {round_index} and deciding the next action."
                self.storage.save_loop_snapshot(snapshot)

                decision_result = await self.decision_engine.decide_with_prompt(
                    spec=spec,
                    context=context,
                    research_state=research_state,
                    current_params=current_params,
                    prompt_documents=prompt_documents,
                )
                decision = decision_result.decision
                round_record.agent_decision = decision
                round_record.agent_trace = decision_result.agent_trace
                round_record.prompt_preview = decision_result.prompt_preview
                self.storage.update_round(round_record)
                self._export_session_artifacts(session_id, spec)
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
                if round_index >= spec.max_rounds:
                    await self._finish_session(session_id, "stopped", "Reached max_rounds.")
                    return

                current_params = spec.merge_param_values(decision.next_params, base=current_params)
                round_index += 1
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self._mark_active_round_failed(session_id, message)
            await self._finish_session(session_id, "failed", message)

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
        if session is not None and session.project_spec is not None:
            self._export_session_artifacts(session_id, session.project_spec)
        try:
            ReportGenerator(
                self.storage,
                self.research_state_builder,
                self.ledger_exporter,
            ).save_report(session_id, self.settings.artifacts_dir)
        except Exception:
            logger.exception("failed to write session report", extra={"_trainee_extra": {"session_id": session_id}})
        await self._publish("loop_finished", {"session_id": session_id, "status": status, "message": message})

    def _mark_active_round_failed(self, session_id: int, message: str) -> None:
        snapshot = self.storage.get_loop_snapshot()
        if snapshot.active_round_id is None:
            return
        round_record = self.storage.get_round(snapshot.active_round_id)
        if round_record is None or round_record.session_id != session_id:
            return
        if round_record.status != "running":
            return
        round_record.status = "failed"
        round_record.end_time = utc_now()
        round_record.error = message
        self.storage.update_round(round_record)

    def _initial_params(self, spec: ProjectSpec) -> Dict[str, Any]:
        return spec.merge_param_values()

    def _prompt_preview_for_dashboard(
        self,
        selected_run: Optional[RoundRecord],
        rounds: List[RoundRecord],
    ) -> tuple[Optional[PromptPreview], str]:
        if selected_run and selected_run.prompt_preview:
            return selected_run.prompt_preview, f"Recorded prompt for selected round #{selected_run.id}"

        latest_recorded = next((item for item in rounds if item.prompt_preview), None)
        if latest_recorded and latest_recorded.prompt_preview:
            return latest_recorded.prompt_preview, f"Latest recorded prompt from round #{latest_recorded.id}"

        live_preview = self._build_live_prompt_preview()
        if live_preview:
            return live_preview, "Next decision preview based on saved project state"
        return None, ""

    def _build_live_prompt_preview(self) -> Optional[PromptPreview]:
        snapshot = self.storage.get_loop_snapshot()
        session = self.storage.get_session(snapshot.current_session_id) if snapshot.current_session_id else None
        spec = session.project_spec if session and session.project_spec else self.storage.get_project_spec()
        context = session.project_context if session and session.project_context else self.storage.get_project_context()
        if spec is None or context is None or not spec.tunable_params:
            return None

        history = self.storage.list_research_rounds(snapshot.current_session_id) if snapshot.current_session_id else []
        if not history:
            latest_round = self.storage.get_latest_round()
            history = [latest_round] if latest_round and latest_round.status == "completed" else []

        current_params = history[-1].param_values if history else self._initial_params(spec)
        return self.decision_engine.build_prompt_preview(
            spec=spec,
            context=context,
            research_state=self.research_state_builder.build(spec, history),
            current_params=current_params,
            prompt_documents=self.prompt_document_loader.load(spec.project_root),
        )

    def _export_session_artifacts(self, session_id: int, spec: ProjectSpec) -> None:
        try:
            rounds = self.storage.list_research_rounds(session_id)
            self.ledger_exporter.export(
                session_id=session_id,
                spec=spec,
                rounds=rounds,
                output_dir=self.settings.artifacts_dir,
            )
        except Exception:
            logger.exception(
                "failed to write session ledger",
                extra={"_trainee_extra": {"session_id": session_id}},
            )

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
