from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote

import httpx
import uvicorn
import yaml

from trainee import __updated_at__, __version__
from trainee.context_builder import ContextBuilder
from trainee.doctor import format_doctor_report, run_doctor
from trainee.events import EventBus
from trainee.models import EventMessage, ProjectContext, PromptPreset
from trainee.orchestrator import RuntimeService
from trainee.output_discovery import OutputDiscoveryEngine, OutputDiscoveryResult
from trainee.provider_probe import ProviderProbeResult, probe_provider
from trainee.project_config import (
    ProjectRegistration,
    compile_project_spec,
    default_project_config,
    detect_project,
    load_project_config,
    project_config_path,
    render_project_config_yaml,
    save_tuning_config,
    tuning_config_path,
    render_tuning_config_yaml,
)
from trainee.providers import AgentDebugSettingsUpdate, ProviderSettingsUpdate, SystemPromptUpdate
from trainee.settings import load_settings
from trainee.storage import Storage
from trainee.tunable_discovery import (
    TunableDiscoveryApply,
    TunableDiscoveryEngine,
    TunableDiscoveryRequest,
    apply_tunable_suggestions,
    apply_tunable_suggestions_with_report,
)


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
RunProgressReporter = Callable[[EventMessage], None]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    method: str
    path: str
    input_schema: Mapping[str, Any]
    path_params: tuple[str, ...] = ()

    def to_manifest_item(self) -> dict[str, Any]:
        item = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.input_schema),
            },
            "http": {
                "method": self.method,
                "path": self.path,
            },
        }
        if self.path_params:
            item["http"]["path_params"] = list(self.path_params)
        return item


def _empty_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {}, "additionalProperties": False}


def _object_schema(properties: dict[str, Any], required: Sequence[str] = ()) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = list(required)
    return schema


TOOL_DEFINITIONS: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        name="project_register",
        description="Save project.yaml and tuning.yaml, register the compiled project, and rebuild its context.",
        method="POST",
        path="/api/project/register",
        input_schema=ProjectRegistration.model_json_schema(),
    ),
    ToolDefinition(
        name="project_update_context",
        description="Replace the saved project context after manual or agent edits.",
        method="POST",
        path="/api/project/context",
        input_schema=ProjectContext.model_json_schema(),
    ),
    ToolDefinition(
        name="runtime_system_prompt_get",
        description="Read the global system prompt used for training decisions.",
        method="GET",
        path="/api/runtime/system-prompt",
        input_schema=_empty_schema(),
    ),
    ToolDefinition(
        name="runtime_system_prompt_update",
        description="Replace the global system prompt and persist it to config.json.",
        method="POST",
        path="/api/runtime/system-prompt",
        input_schema=SystemPromptUpdate.model_json_schema(),
    ),
    ToolDefinition(
        name="project_get",
        description="Read the saved project spec, context, and loop snapshot.",
        method="GET",
        path="/api/project",
        input_schema=_empty_schema(),
    ),
    ToolDefinition(
        name="project_suggest_tunables",
        description="Suggest config-backed tunable params from the saved project context and baseline config.",
        method="POST",
        path="/api/project/tunables/suggest",
        input_schema=TunableDiscoveryRequest.model_json_schema(),
    ),
    ToolDefinition(
        name="project_apply_tunables",
        description="Apply reviewed tunable param suggestions to .trainee/tuning.yaml.",
        method="POST",
        path="/api/project/tunables/apply",
        input_schema=TunableDiscoveryApply.model_json_schema(),
    ),
    ToolDefinition(
        name="runtime_provider_get",
        description="Read the active LLM provider settings without exposing API keys.",
        method="GET",
        path="/api/runtime/provider",
        input_schema=_empty_schema(),
    ),
    ToolDefinition(
        name="runtime_provider_update",
        description="Update the active LLM provider settings and persist them to config.json.",
        method="POST",
        path="/api/runtime/provider",
        input_schema=ProviderSettingsUpdate.model_json_schema(),
    ),
    ToolDefinition(
        name="runtime_provider_test",
        description="Send a live probe request to the configured LLM provider without exposing API keys.",
        method="POST",
        path="/api/runtime/provider/test",
        input_schema=_empty_schema(),
    ),
    ToolDefinition(
        name="runtime_debug_get",
        description="Read whether Agent Debug trace collection is enabled.",
        method="GET",
        path="/api/runtime/debug",
        input_schema=_empty_schema(),
    ),
    ToolDefinition(
        name="runtime_debug_update",
        description="Enable or disable Agent Debug trace collection for future rounds.",
        method="POST",
        path="/api/runtime/debug",
        input_schema=AgentDebugSettingsUpdate.model_json_schema(),
    ),
    ToolDefinition(
        name="prompt_preview",
        description="Preview the next LLM decision prompt for the saved project or a selected run.",
        method="GET",
        path="/api/prompt-preview",
        input_schema=_object_schema({"run_id": {"type": "integer"}}),
    ),
    ToolDefinition(
        name="prompt_presets_list",
        description="List saved prompt presets, optionally scoped to one project root.",
        method="GET",
        path="/api/prompt-presets",
        input_schema=_object_schema({"project_root": {"type": "string"}}),
    ),
    ToolDefinition(
        name="prompt_presets_save",
        description="Create or update a prompt preset.",
        method="POST",
        path="/api/prompt-presets",
        input_schema=PromptPreset.model_json_schema(),
    ),
    ToolDefinition(
        name="loop_start",
        description="Start the training automation loop for the registered project, optionally resuming a prior session.",
        method="POST",
        path="/api/loop/start",
        input_schema=_object_schema({"resume_session_id": {"type": "integer"}}),
    ),
    ToolDefinition(
        name="loop_stop",
        description="Request the active loop to stop after the current round, or immediately when force is true.",
        method="POST",
        path="/api/loop/stop",
        input_schema=_object_schema({"force": {"type": "boolean"}}),
    ),
    ToolDefinition(
        name="loop_get",
        description="Read the current loop snapshot.",
        method="GET",
        path="/api/loop",
        input_schema=_empty_schema(),
    ),
    ToolDefinition(
        name="runs_list",
        description="List run sessions, rounds, and loop state.",
        method="GET",
        path="/api/runs",
        input_schema=_empty_schema(),
    ),
    ToolDefinition(
        name="runs_get",
        description="Read one run round by id.",
        method="GET",
        path="/api/runs/{run_id}",
        input_schema=_object_schema({"run_id": {"type": "integer"}}, required=("run_id",)),
        path_params=("run_id",),
    ),
    ToolDefinition(
        name="session_report",
        description="Generate a Markdown report for one run session.",
        method="GET",
        path="/api/sessions/{session_id}/report",
        input_schema=_object_schema({"session_id": {"type": "integer"}}, required=("session_id",)),
        path_params=("session_id",),
    ),
)

TOOL_INDEX = {definition.name: definition for definition in TOOL_DEFINITIONS}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = getattr(args, "command", None)

    if command is None or command == "serve":
        host = getattr(args, "host", "127.0.0.1")
        port = getattr(args, "port", 8000)
        reload = getattr(args, "reload", False)
        run_web_service(host=host, port=port, reload=reload)
        return 0

    if command == "version":
        print(f"Trainee {__version__}")
        print(f"Last updated: {__updated_at__}")
        return 0

    if command == "webui":
        host = getattr(args, "host", "127.0.0.1")
        port = getattr(args, "port", 8000)
        reload = getattr(args, "reload", False)
        open_browser = not getattr(args, "no_open", False)
        run_web_service(host=host, port=port, reload=reload, open_browser=open_browser)
        return 0

    if command == "tools":
        payload = build_tool_manifest(tool_name=getattr(args, "name", None), base_url=args.base_url)
        _print_json(payload)
        return 0

    if command == "init":
        try:
            result = asyncio.run(
                initialize_project(
                    Path(args.project_root),
                    force=args.force,
                    baseline_config=args.baseline_config,
                    skip_provider_test=args.skip_provider_test,
                )
            )
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _print_init_result(result)
        return 0

    if command == "prepare":
        try:
            result = asyncio.run(
                prepare_project_async(
                    Path(args.project_root),
                    replace=args.replace,
                    skip_provider_test=args.skip_provider_test,
                )
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _print_prepare_result(result)
        return 0

    if command == "run":
        try:
            security_mode = "unsafe" if args.unsafe else ("guarded" if args.guarded else None)
            if args.dry_run:
                report = run_doctor(Path(args.project_root), security_mode=security_mode)
                print(format_doctor_report(report), end="")
                return 1 if report.has_failures else 0
            result = asyncio.run(
                run_project(
                    Path(args.project_root),
                    security_mode=security_mode,
                    progress_reporter=_RunProgressPrinter(),
                )
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _print_run_result(result)
        return 0 if result["status"] != "failed" else 1

    if command == "doctor":
        report = run_doctor(Path(args.project_root), skip_provider_test=args.skip_provider_test)
        print(format_doctor_report(report), end="")
        return 1 if report.has_failures else 0

    if command == "tunables-discover":
        try:
            result = asyncio.run(
                discover_tunables(
                    Path(args.project_root),
                    apply=args.apply,
                    replace=args.replace,
                    limit=args.limit,
                )
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _print_tunable_discovery_result(result)
        return 0

    if command == "call":
        try:
            payload = load_tool_input(args.input)
            result = call_tool(args.name, payload, base_url=args.base_url, timeout=args.timeout)
        except (OSError, ValueError, RuntimeError, httpx.HTTPError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _print_json(result)
        return 0

    if command == "report":
        try:
            text = fetch_report(args.session_id, base_url=args.base_url, timeout=args.timeout, output=args.output)
        except (OSError, RuntimeError, httpx.HTTPError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.output is None:
            print(text, end="")
        return 0

    parser.error(f"unknown command: {command}")
    return 2


def build_tool_manifest(tool_name: str | None = None, base_url: str = DEFAULT_BASE_URL) -> dict[str, Any]:
    definitions = TOOL_DEFINITIONS if tool_name is None else (_get_tool(tool_name),)
    return {
        "base_url": base_url,
        "tools": [definition.to_manifest_item() for definition in definitions],
    }


def run_web_service(host: str = "127.0.0.1", port: int = 8000, reload: bool = False, open_browser: bool = False) -> None:
    if open_browser:
        webbrowser.open(_webui_url(host, port))
    uvicorn.run("trainee.app:app", host=host, port=port, reload=reload)


def load_tool_input(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return {}

    if raw == "-":
        text = sys.stdin.read()
    elif raw.startswith("@"):
        text = Path(raw[1:]).expanduser().read_text(encoding="utf-8")
    else:
        text = raw

    if not text.strip():
        return {}

    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("tool input must be a JSON object")
    return payload


def call_tool(name: str, payload: Mapping[str, Any], base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0) -> Any:
    definition = _get_tool(name)
    path = _render_tool_path(definition, payload)
    request_payload = {key: value for key, value in payload.items() if key not in definition.path_params and value is not None}

    try:
        with httpx.Client(base_url=base_url, timeout=timeout) as client:
            if definition.method == "GET":
                response = client.get(path, params=request_payload)
            else:
                response = client.request(definition.method, path, json=request_payload or None)
    except httpx.ConnectError as exc:
        raise RuntimeError(f"could not connect to {base_url}; start the service with `uv run trainee serve`") from exc

    if response.is_error:
        raise RuntimeError(f"{definition.method} {path} returned {response.status_code}: {_response_body(response)}")

    try:
        return response.json()
    except ValueError:
        return {"text": response.text}


def fetch_report(
    session_id: int,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 30.0,
    output: str | None = None,
) -> str:
    try:
        with httpx.Client(base_url=base_url, timeout=timeout) as client:
            response = client.get(f"/api/sessions/{session_id}/report")
    except httpx.ConnectError as exc:
        raise RuntimeError(f"could not connect to {base_url}; start the service with `uv run trainee serve`") from exc

    if response.is_error:
        raise RuntimeError(f"GET /api/sessions/{session_id}/report returned {response.status_code}: {_response_body(response)}")

    text = response.text
    if output is not None:
        Path(output).expanduser().write_text(text, encoding="utf-8")
    return text


async def initialize_project(
    project_root: Path,
    force: bool = False,
    baseline_config: str | None = None,
    skip_provider_test: bool = False,
) -> dict[str, Any]:
    project_root = project_root.expanduser().resolve()
    if not project_root.exists() or not project_root.is_dir():
        raise ValueError(f"project root does not exist or is not a directory: {project_root}")

    provider_test = None if skip_provider_test else await _require_provider_ready(project_root)

    trainee_dir = project_root / ".trainee"
    trainee_dir.mkdir(parents=True, exist_ok=True)
    (trainee_dir / "runs").mkdir(parents=True, exist_ok=True)
    (trainee_dir / "logs").mkdir(parents=True, exist_ok=True)

    discovery = detect_project(project_root)
    generated_config = default_project_config(
        project_root,
        discovery,
        baseline_config=baseline_config,
    )
    config_path = project_config_path(project_root)
    tuning_path = tuning_config_path(project_root)
    generate_config = force or not config_path.is_file()
    config = generated_config if generate_config else load_project_config(project_root)
    spec = compile_project_spec(project_root, config)
    context = ContextBuilder().build(spec)
    files_read = _launch_read_targets(project_root)

    outputs = {
        config_path: _render_project_yaml(config),
        tuning_path: render_tuning_config_yaml(config.tuning),
        trainee_dir / "context.md": _render_context_markdown(context),
        trainee_dir / "README.md": _render_launch_readme(project_root, discovery),
    }
    already_initialized = all(path.exists() for path in outputs)
    written: list[Path] = []
    skipped: list[Path] = []
    for path, content in outputs.items():
        if path.exists() and not force:
            skipped.append(path)
            continue
        path.write_text(content, encoding="utf-8")
        written.append(path)

    return {
        "project_root": project_root,
        "trainee_dir": trainee_dir,
        "config_path": config_path,
        "tuning_config_path": tuning_path,
        "config": config,
        "spec": spec,
        "force": force,
        "files_read": files_read,
        "files_written": written,
        "files_skipped": skipped,
        "already_initialized": already_initialized,
        "launcher_template": spec.launcher_template,
        "discovery": discovery,
        "tunable_discovery": None,
        "applied_tunables": [],
        "provider_test": provider_test,
        "warnings": context.warnings,
    }


def init_project(
    project_root: Path,
    force: bool = False,
    baseline_config: str | None = None,
    skip_provider_test: bool = False,
) -> dict[str, Any]:
    return asyncio.run(
        initialize_project(
            project_root,
            force=force,
            baseline_config=baseline_config,
            skip_provider_test=skip_provider_test,
        )
    )


async def prepare_project_async(
    project_root: Path,
    *,
    replace: bool = False,
    skip_provider_test: bool = False,
) -> dict[str, Any]:
    project_root = project_root.expanduser().resolve()
    config_path = project_config_path(project_root)
    tuning_path = tuning_config_path(project_root)
    context_path = project_root / ".trainee" / "context.md"
    config = load_project_config(project_root)
    spec = compile_project_spec(project_root, config)
    context = ContextBuilder().build(spec)
    settings = load_settings(repo_root=project_root, project_root=project_root)
    provider_test = None if skip_provider_test else await _require_provider_ready(project_root, settings=settings)
    output_discovery = OutputDiscoveryResult()
    if config.output is None:
        output_discovery = await OutputDiscoveryEngine(settings).suggest(spec, context)
    output_applied = False
    if config.launch.baseline_config and config.output is None and output_discovery.output is not None:
        output_conflicts = [
            item.name
            for item in config.tuning.params
            if item.config_path == output_discovery.output.config_path
        ]
        if output_conflicts and not replace:
            output_discovery = output_discovery.model_copy(
                update={
                    "output": None,
                    "warnings": [
                        *output_discovery.warnings,
                        "detected output path is already listed in tuning.yaml; rerun with --replace or edit tuning.yaml.",
                    ],
                }
            )
        else:
            config = config.model_copy(update={"output": output_discovery.output})
            output_applied = True
            spec = compile_project_spec(project_root, config)
            context = ContextBuilder().build(spec)

    tunable_discovery = None
    applied_tunables = []
    should_discover_tunables = bool(
        config.launch.baseline_config
        and (replace or not tuning_path.is_file() or not config.tuning.params)
    )
    if should_discover_tunables:
        tunable_discovery = await TunableDiscoveryEngine(settings).suggest(
            spec,
            context,
            fixed_args=[*config.launch.args, *config.run.fixed_args],
        )
        config, applied_tunables = apply_tunable_suggestions(
            config,
            tunable_discovery.suggestions,
            replace=replace or not config.tuning.params,
        )
        spec = compile_project_spec(project_root, config)
        context = ContextBuilder().build(spec)

    files_written: list[Path] = []
    files_unchanged: list[Path] = []
    if output_applied:
        if _write_if_changed(config_path, _render_project_yaml(config)):
            files_written.append(config_path)
        else:
            files_unchanged.append(config_path)
    if should_discover_tunables:
        if _write_if_changed(tuning_path, render_tuning_config_yaml(config.tuning)):
            files_written.append(tuning_path)
        else:
            files_unchanged.append(tuning_path)
    if _write_if_changed(context_path, _render_context_markdown(context)):
        files_written.append(context_path)
    else:
        files_unchanged.append(context_path)

    return {
        "project_root": project_root,
        "config_path": config_path,
        "tuning_config_path": tuning_path,
        "context_path": context_path,
        "config": config,
        "spec": spec,
        "replace": replace,
        "files_written": files_written,
        "files_unchanged": files_unchanged,
        "output_discovery": output_discovery,
        "output_applied": output_applied,
        "tunable_discovery": tunable_discovery,
        "applied_tunables": applied_tunables,
        "provider_test": provider_test,
        "warnings": context.warnings,
    }


def prepare_project(project_root: Path, *, replace: bool = False, skip_provider_test: bool = False) -> dict[str, Any]:
    return asyncio.run(prepare_project_async(project_root, replace=replace, skip_provider_test=skip_provider_test))


class _RunProgressPrinter:
    def __init__(self, max_rounds: int | None = None, heartbeat_interval_sec: float = 30.0) -> None:
        self.max_rounds = max_rounds
        self.heartbeat_interval_sec = heartbeat_interval_sec
        self._last_heartbeat_by_round: dict[int, float] = {}

    def set_max_rounds(self, max_rounds: int) -> None:
        self.max_rounds = max_rounds

    def __call__(self, event: EventMessage) -> None:
        if event.event_type == "loop_started":
            self._print_loop_started(event.payload)
        elif event.event_type == "round_started":
            self._print_round_started(event.payload)
        elif event.event_type == "heartbeat":
            self._print_heartbeat(event.payload)
        elif event.event_type == "round_finished":
            self._print_round_finished(event.payload)
        elif event.event_type == "decision_made":
            self._print_decision(event.payload)
        elif event.event_type == "loop_finished":
            self._print_loop_finished(event.payload)

    def _print_loop_started(self, payload: Mapping[str, Any]) -> None:
        session_id = payload.get("session_id", "?")
        line = f"Session {session_id} started"
        if self.max_rounds is not None:
            line += f"; max_rounds={self.max_rounds}"
        print(line, flush=True)

    def _print_round_started(self, payload: Mapping[str, Any]) -> None:
        round_index = self._round_index(payload)
        round_id = payload.get("round_id")
        line = f"{self._round_label(round_index)} started"
        if round_id is not None:
            line += f" (#{round_id})"
        print(line, flush=True)

    def _print_heartbeat(self, payload: Mapping[str, Any]) -> None:
        round_index = self._round_index(payload)
        now = time.monotonic()
        previous = self._last_heartbeat_by_round.get(round_index)
        if previous is not None and now - previous < self.heartbeat_interval_sec:
            return
        self._last_heartbeat_by_round[round_index] = now

        last_signal_at = payload.get("last_signal_at")
        round_dir = payload.get("round_dir")
        line = f"{self._round_label(round_index)} running"
        if last_signal_at:
            line += f"; last_signal_at={last_signal_at}"
        elif round_dir:
            line += f"; workspace={round_dir}"
        print(line, flush=True)

    def _print_round_finished(self, payload: Mapping[str, Any]) -> None:
        round_index = self._round_index(payload)
        status = payload.get("status", "finished")
        line = f"{self._round_label(round_index)} {status}"
        metrics = payload.get("metrics")
        if isinstance(metrics, Mapping) and metrics:
            line += f"; metrics: {self._format_metrics(metrics)}"
        print(line, flush=True)

    def _print_decision(self, payload: Mapping[str, Any]) -> None:
        action = payload.get("action", "unknown")
        reason = str(payload.get("reason") or "").strip()
        line = f"Decision: {action}"
        if reason:
            line += f" - {self._truncate(reason)}"
        print(line, flush=True)

    def _print_loop_finished(self, payload: Mapping[str, Any]) -> None:
        status = payload.get("status", "finished")
        message = str(payload.get("message") or "").strip()
        line = f"Run finished: {status}"
        if message:
            line += f" - {self._truncate(message)}"
        print(line, flush=True)

    def _round_label(self, round_index: int) -> str:
        if self.max_rounds is None:
            return f"Round {round_index}"
        return f"Round {round_index}/{self.max_rounds}"

    def _round_index(self, payload: Mapping[str, Any]) -> int:
        try:
            return int(payload.get("round_index") or 0)
        except (TypeError, ValueError):
            return 0

    def _format_metrics(self, metrics: Mapping[str, Any]) -> str:
        rendered = [
            f"{name}={self._format_metric_value(value)}"
            for name, value in list(metrics.items())[:3]
        ]
        if len(metrics) > 3:
            rendered.append("...")
        return ", ".join(rendered)

    def _format_metric_value(self, value: Any) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, (int, float)):
            return f"{value:g}"
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def _truncate(self, value: str, max_length: int = 180) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3].rstrip() + "..."


async def _drain_progress_events(
    queue: asyncio.Queue[EventMessage] | None,
    progress_reporter: RunProgressReporter | None,
) -> None:
    if queue is None or progress_reporter is None:
        return
    while True:
        try:
            event = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        progress_reporter(event)


async def run_project(
    project_root: Path,
    security_mode: str | None = None,
    progress_reporter: RunProgressReporter | None = None,
) -> dict[str, Any]:
    project_root = project_root.expanduser().resolve()
    provider_test = await _require_provider_ready(project_root)
    report = run_doctor(project_root, security_mode=security_mode, provider_test_result=provider_test)
    if report.has_failures:
        raise ValueError("preflight failed:\n" + format_doctor_report(report))
    spec = compile_project_spec(project_root, load_project_config(project_root), security_mode=security_mode)

    settings = load_settings(repo_root=project_root, project_root=project_root)
    storage = Storage(settings.database_path)
    event_bus = EventBus()
    runtime = RuntimeService(settings, storage, event_bus)
    progress_queue = await event_bus.subscribe() if progress_reporter is not None else None
    if isinstance(progress_reporter, _RunProgressPrinter):
        progress_reporter.set_max_rounds(spec.max_rounds)
    try:
        await runtime.register_project(spec)
        snapshot = await runtime.start_loop()
        await _drain_progress_events(progress_queue, progress_reporter)
        while runtime.loop_is_running():
            await asyncio.sleep(0.2)
            snapshot = storage.get_loop_snapshot()
            await _drain_progress_events(progress_queue, progress_reporter)
        await _drain_progress_events(progress_queue, progress_reporter)
        session = storage.get_latest_session()
        return {
            "project_root": project_root,
            "data_dir": settings.data_dir,
            "artifacts_dir": settings.artifacts_dir,
            "session_id": session.id if session else snapshot.current_session_id,
            "status": session.status if session else snapshot.status,
            "message": session.stop_reason if session and session.stop_reason else snapshot.message,
            "security_mode": spec.security_mode,
        }
    finally:
        if progress_queue is not None:
            await event_bus.unsubscribe(progress_queue)
        storage.close()


async def _require_provider_ready(project_root: Path, *, settings: Any = None) -> ProviderProbeResult:
    settings = settings or load_settings(repo_root=project_root, project_root=project_root)
    result = await probe_provider(settings)
    if not result.ok:
        raise ValueError("LLM provider test failed: " + _format_provider_test_result(result))
    return result


def _format_provider_test_result(result: ProviderProbeResult) -> str:
    parts = [
        f"provider={result.provider}",
        f"model={result.model}",
        f"status={result.status}",
    ]
    if result.http_status is not None:
        parts.append(f"http_status={result.http_status}")
    summary = result.failure_summary()
    if summary:
        parts.append(summary)
    return "; ".join(parts)


async def discover_tunables(
    project_root: Path,
    *,
    apply: bool = False,
    replace: bool = False,
    limit: int = 32,
) -> dict[str, Any]:
    project_root = project_root.expanduser().resolve()
    config = load_project_config(project_root)
    spec = compile_project_spec(project_root, config)
    context = ContextBuilder().build(spec)
    settings = load_settings(repo_root=project_root, project_root=project_root)
    result = await TunableDiscoveryEngine(settings).suggest(
        spec,
        context,
        limit=limit,
        fixed_args=[*config.launch.args, *config.run.fixed_args],
    )
    applied = []
    skipped = []
    if apply:
        updated, applied, skipped = apply_tunable_suggestions_with_report(config, result.suggestions, replace=replace)
        save_tuning_config(project_root, updated.tuning)
    return {
        "project_root": project_root,
        "config_path": project_config_path(project_root),
        "tuning_config_path": tuning_config_path(project_root),
        "result": result,
        "applied": applied,
        "skipped": skipped,
        "apply": apply,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trainee agent runtime.")
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Run the local web service.")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    serve_parser.add_argument("--port", default=8000, type=int, help="Bind port.")
    serve_parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload mode.")
    serve_parser.set_defaults(command="serve")

    webui_parser = subparsers.add_parser("webui", help="Run the local web service and open the Web UI.")
    webui_parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    webui_parser.add_argument("--port", default=8000, type=int, help="Bind port.")
    webui_parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload mode.")
    webui_parser.add_argument("--no-open", action="store_true", help="Start the service without opening a browser.")
    webui_parser.set_defaults(command="webui")

    version_parser = subparsers.add_parser("version", help="Print the installed version and last update date/time.")
    version_parser.set_defaults(command="version")

    init_parser = subparsers.add_parser("init", help="Initialize Trainee project files in a training project directory.")
    init_parser.add_argument("project_root", nargs="?", default=".", help="Training project directory. Defaults to the current directory.")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing files under .trainee.")
    init_parser.add_argument(
        "--baseline-config",
        help="Initialize launch.baseline_config with a project-local training config path.",
    )
    init_parser.add_argument(
        "--skip-provider-test",
        action="store_true",
        help="Skip the live LLM provider test for offline setup.",
    )
    init_parser.set_defaults(command="init")

    prepare_parser = subparsers.add_parser("prepare", help="Infer runtime output and tunable config from launch.baseline_config.")
    prepare_parser.add_argument("project_root", nargs="?", default=".", help="Training project directory. Defaults to the current directory.")
    prepare_parser.add_argument("--replace", action="store_true", help="Replace existing tuning.yaml params with fresh suggestions.")
    prepare_parser.add_argument(
        "--skip-provider-test",
        action="store_true",
        help="Skip the live LLM provider test for offline setup.",
    )
    prepare_parser.set_defaults(command="prepare")

    run_parser = subparsers.add_parser("run", help="Run the training loop from .trainee/project.yaml.")
    run_parser.add_argument("project_root", nargs="?", default=".", help="Training project directory. Defaults to the current directory.")
    security_group = run_parser.add_mutually_exclusive_group()
    security_group.add_argument("--guarded", action="store_true", help="Run with the default bubblewrap sandbox.")
    security_group.add_argument("--unsafe", action="store_true", help="Run without bubblewrap isolation.")
    run_parser.add_argument("--dry-run", action="store_true", help="Validate project.yaml and print the baseline command without starting a session.")
    run_parser.set_defaults(command="run")

    doctor_parser = subparsers.add_parser("doctor", help="Check whether a training project is ready for Trainee.")
    doctor_parser.add_argument("project_root", nargs="?", default=".", help="Training project directory. Defaults to the current directory.")
    doctor_parser.add_argument(
        "--skip-provider-test",
        action="store_true",
        help="Skip the live LLM provider test for offline setup.",
    )
    doctor_parser.set_defaults(command="doctor")

    tunables_parser = subparsers.add_parser("tunables", help="Inspect or update tunable parameter configuration.")
    tunables_subparsers = tunables_parser.add_subparsers(dest="tunables_command", required=True)
    discover_parser = tunables_subparsers.add_parser("discover", help="Suggest config-backed tuning params from project context.")
    discover_parser.add_argument("project_root", nargs="?", default=".", help="Training project directory. Defaults to the current directory.")
    discover_parser.add_argument("--limit", default=32, type=int, help="Maximum suggestions to return.")
    discover_parser.add_argument("--apply", action="store_true", help="Write suggestions into .trainee/tuning.yaml.")
    discover_parser.add_argument("--replace", action="store_true", help="With --apply, replace existing tuning.yaml params.")
    discover_parser.set_defaults(command="tunables-discover")

    tools_parser = subparsers.add_parser("tools", help="Print tool-call compatible function schemas.")
    tools_parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL to include in the manifest.")
    tools_parser.add_argument("--name", choices=sorted(TOOL_INDEX), help="Print only one tool definition.")
    tools_parser.set_defaults(command="tools")

    call_parser = subparsers.add_parser("call", help="Call one tool against a running Trainee service.")
    call_parser.add_argument("name", choices=sorted(TOOL_INDEX), help="Tool name to invoke.")
    call_parser.add_argument("--input", "-i", help="JSON object, @path/to/input.json, or - for stdin.")
    call_parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Running Trainee service URL.")
    call_parser.add_argument("--timeout", default=30.0, type=float, help="HTTP timeout in seconds.")
    call_parser.set_defaults(command="call")

    report_parser = subparsers.add_parser("report", help="Print or save a Markdown report for one session.")
    report_parser.add_argument("session_id", type=int, help="Session id to report.")
    report_parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Running Trainee service URL.")
    report_parser.add_argument("--timeout", default=30.0, type=float, help="HTTP timeout in seconds.")
    report_parser.add_argument("--output", "-o", help="Optional file path for the Markdown report.")
    report_parser.set_defaults(command="report")

    return parser


def _get_tool(name: str) -> ToolDefinition:
    try:
        return TOOL_INDEX[name]
    except KeyError as exc:
        raise ValueError(f"unknown tool {name!r}; run `trainee tools` to list available tools") from exc


def _render_tool_path(definition: ToolDefinition, payload: Mapping[str, Any]) -> str:
    path = definition.path
    for param in definition.path_params:
        if param not in payload:
            raise ValueError(f"tool {definition.name!r} requires input field {param!r}")
        path = path.replace("{" + param + "}", quote(str(payload[param]), safe=""))
    return path


def _webui_url(host: str, port: int) -> str:
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    if ":" in browser_host and not browser_host.startswith("["):
        browser_host = f"[{browser_host}]"
    return f"http://{browser_host}:{port}/"


def _response_body(response: httpx.Response) -> str:
    try:
        return json.dumps(response.json(), ensure_ascii=False)
    except ValueError:
        return response.text


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _print_init_result(result: Mapping[str, Any]) -> None:
    project_root = Path(result["project_root"])
    config = result["config"]
    spec = result["spec"]
    discovery = result["discovery"]
    print("Trainee init")
    print(f"- Project: {project_root}")
    if result["force"]:
        print("- Status: regenerated project files (--force)")
    elif result["already_initialized"] and not result["files_written"]:
        print("- Status: already initialized; kept existing project files")
    elif result["files_skipped"]:
        print("- Status: added missing project files; kept existing files")
    else:
        print("- Status: initialized new project files")

    print("")
    print("Files")
    for path in result["files_read"]:
        print(f"- Read: {Path(path).relative_to(project_root)}")
    if not result["files_read"]:
        print("- Read: no README, entrypoint, or config candidates found")
    for path in result["files_written"]:
        print(f"- Wrote: {Path(path).relative_to(project_root)}")
    for path in result["files_skipped"]:
        print(f"- Kept: {Path(path).relative_to(project_root)}")
    print(f"- Config: {Path(result['config_path']).relative_to(project_root)}")
    print(f"- Tuning: {Path(result['tuning_config_path']).relative_to(project_root)}")

    print("")
    print("Discovery")
    detected_environment = discovery.environment
    if discovery.env_name:
        detected_environment += f" ({discovery.env_name})"
    print(f"- Environment: {detected_environment}")
    print(f"- Entrypoints: {_joined_or_none(discovery.entrypoints)}")
    print(f"- Data candidates: {_joined_or_none(discovery.data_dirs)}")
    print(f"- Config candidates: {_joined_or_none(discovery.config_files)}")
    print(f"- Training limit candidates: {_format_command_args(discovery.limit_flags)}")

    print("")
    print("Effective configuration")
    configured_environment = config.launch.environment
    if config.launch.env_name:
        configured_environment += f" ({config.launch.env_name})"
    timeout = (
        f"{_format_number(config.run.timeout_minutes)} minutes"
        if config.run.timeout_minutes is not None
        else "disabled"
    )
    print(f"- Environment: {configured_environment}")
    print(f"- Working directory: {spec.working_dir}")
    print(f"- Security: {spec.security_mode}")
    print(f"- Budget: max_rounds={spec.max_rounds}, timeout={timeout}")
    print(f"- Data inputs: {_format_data_inputs(config.data)}")
    print(f"- Baseline config: {config.launch.baseline_config or 'not set'}")
    print(f"- Launch arguments: {_format_command_args(config.launch.args)}")
    print(f"- Fixed arguments: {_format_command_args(config.run.fixed_args)}")
    print(f"- Tunable parameters: {_format_tunable_params(spec.tunable_params)}")
    metric_summary = _format_metrics(spec.metric_specs)
    if not spec.metric_specs:
        metric_summary += " (built-in loss/total_loss parsing only)"
    print(f"- Metrics: {metric_summary}")
    print(f"- Runtime: round_timeout={timeout}, wandb={'enabled' if spec.wandb_enabled else 'disabled'}")
    print(
        "- Activity monitor: "
        f"every {_format_number(spec.heartbeat_interval_sec)}s, "
        f"sources={_format_signal_sources(spec.signal_sources)}"
    )
    print(f"- Log paths: {_joined_or_none(spec.log_paths)}")
    launcher = result["launcher_template"] or "set launch.command in .trainee/project.yaml"
    print(f"- Launcher: {launcher}")
    _print_provider_test_summary(result.get("provider_test"))

    print("")
    print("Next")
    print("- Review: .trainee/project.yaml, .trainee/tuning.yaml, and .trainee/context.md")
    print("- Prepare: set launch.baseline_config, then run `trainee prepare`")
    print("- Validate: trainee doctor or trainee run --dry-run")
    print("- Next: run `trainee prepare`, review generated config, then run `trainee doctor`")
    for warning in result["warnings"]:
        print(f"- Warning: {warning}")


def _print_prepare_result(result: Mapping[str, Any]) -> None:
    project_root = Path(result["project_root"])
    config = result["config"]
    spec = result["spec"]
    print("Trainee prepare")
    print(f"- Project: {project_root}")
    print(f"- Baseline config: {config.launch.baseline_config or 'not set'}")
    print(f"- Replace tuning: {'true' if result['replace'] else 'false'}")

    print("")
    print("Files")
    for path in result["files_written"]:
        print(f"- Wrote: {Path(path).relative_to(project_root)}")
    for path in result["files_unchanged"]:
        print(f"- Kept: {Path(path).relative_to(project_root)}")
    print(f"- Config: {Path(result['config_path']).relative_to(project_root)}")
    print(f"- Tuning: {Path(result['tuning_config_path']).relative_to(project_root)}")

    print("")
    print("Output config")
    output_discovery: OutputDiscoveryResult = result["output_discovery"]
    if not config.launch.baseline_config:
        print("- Status: skipped (launch.baseline_config is not set)")
    elif config.output is not None and not config.output.config_path:
        print("- Status: skipped (output.config_path is not set)")
    elif config.output is not None and not result["output_applied"]:
        print(f"- Status: configured ({config.output.config_path} -> {config.output.path})")
    elif result["output_applied"] and output_discovery.output is not None:
        detected = output_discovery.candidates[0] if output_discovery.candidates else None
        if detected is not None:
            print(f"- Detected: {detected.config_path} = {detected.current_value}")
        print(f"- Applied: project.yaml output.config_path = {output_discovery.output.config_path}")
        print(f"- Runtime path: {output_discovery.output.path}")
    else:
        print("- Status: skipped (no single confident output path detected)")
    for candidate in output_discovery.candidates[:5]:
        print(f"- Candidate: {candidate.config_path}={candidate.current_value} ({candidate.reason})")
    for warning in output_discovery.warnings:
        print(f"- Warning: {warning}")

    print("")
    print("Tunable discovery")
    tunable_discovery = result["tunable_discovery"]
    if tunable_discovery is None:
        if not config.launch.baseline_config:
            reason = "launch.baseline_config is not set"
        elif config.tuning.params and not result["replace"]:
            reason = "tuning.yaml already has params"
        else:
            reason = "not needed"
        print(f"- Status: skipped ({reason})")
    else:
        source = tunable_discovery.source
        if source == "llm":
            source += f" ({tunable_discovery.provider}:{tunable_discovery.model})"
        print(f"- Source: {source}")
        print(f"- Generated: {len(result['applied_tunables'])} parameter(s) in tuning.yaml params")
        for warning in tunable_discovery.warnings:
            print(f"- Warning: {warning}")

    print("")
    print("Effective configuration")
    print(f"- Output: {spec.output.config_path + ' -> ' + spec.output.path if spec.output else 'not set'}")
    print(f"- Tunable parameters: {_format_tunable_params(spec.tunable_params)}")
    _print_provider_test_summary(result.get("provider_test"))

    print("")
    print("Next")
    print("- Review: .trainee/project.yaml, .trainee/tuning.yaml, and .trainee/context.md")
    print("- Validate: trainee doctor or trainee run --dry-run")
    print("- Next: run `trainee doctor`, then run `trainee run`")
    for warning in result["warnings"]:
        print(f"- Warning: {warning}")


def _joined_or_none(values: Sequence[Any]) -> str:
    rendered = [str(value) for value in values]
    return ", ".join(rendered) if rendered else "none"


def _print_provider_test_summary(result: Any) -> None:
    print("")
    print("Provider")
    if result is None:
        print("- Live test: skipped")
        return
    print(f"- Live test: {'ok' if result.ok else 'failed'}")
    print(f"- Provider: {result.provider}")
    print(f"- Model: {result.model}")
    print(f"- Status: {result.status}")
    if result.http_status is not None:
        print(f"- HTTP: {result.http_status}")
    if not result.ok and result.failure_summary():
        print(f"- Error: {result.failure_summary()}")


def _format_command_args(values: Sequence[Any]) -> str:
    rendered = []
    for item in values:
        if item.value is None or item.value is True:
            rendered.append(item.flag)
        elif item.value is not False:
            rendered.append(f"{item.flag}={item.value}")
    return _joined_or_none(rendered)


def _format_data_inputs(values: Sequence[Any]) -> str:
    rendered = [
        f"{item.path} via {item.flag}" if item.flag else item.path
        for item in values
    ]
    return _joined_or_none(rendered)


def _format_tunable_params(values: Sequence[Any]) -> str:
    rendered = []
    for item in values:
        target = item.config_path or item.flag or "unset"
        target_kind = "config" if item.config_path else "cli"
        details = [f"{target_kind}:{target}", item.type]
        if item.default is not None:
            details.append(f"default={item.default}")
        if item.min_value is not None or item.max_value is not None:
            details.append(f"range=[{item.min_value}, {item.max_value}]")
        if item.choices:
            details.append(f"choices={','.join(item.choices)}")
        rendered.append(f"{item.name} ({', '.join(details)})")
    return _joined_or_none(rendered)


def _format_metrics(values: Sequence[Any]) -> str:
    rendered = [
        f"{item.name} via {item.source} ({item.goal}, required={'true' if item.required else 'false'})"
        for item in values
    ]
    return _joined_or_none(rendered)


def _format_signal_sources(values: Sequence[Any]) -> str:
    rendered = []
    for item in values:
        paths = item.configured_paths()
        rendered.append(f"{item.type}({', '.join(paths)})" if paths else item.type)
    return "; ".join(rendered) if rendered else "process output"


def _format_number(value: float) -> str:
    return f"{value:g}"


def _print_run_result(result: Mapping[str, Any]) -> None:
    print("Trainee run")
    print(f"- Project: {result['project_root']}")
    print(f"- Security: {result['security_mode']}")
    print(f"- Data: {result['data_dir']}")
    print(f"- Artifacts: {result['artifacts_dir']}")
    if result["session_id"] is not None:
        print(f"- Session: {result['session_id']}")
    print(f"- Status: {result['status']}")
    if result["message"]:
        print(f"- Message: {result['message']}")


def _print_tunable_discovery_result(result: Mapping[str, Any]) -> None:
    discovery = result["result"]
    suggestions = [item.model_dump(mode="json", exclude_none=True) for item in discovery.suggestions]
    candidates = [item.model_dump(mode="json", exclude_none=True) for item in discovery.candidates]
    print("Trainee tunable discovery")
    print(f"- Project: {result['project_root']}")
    print(f"- Baseline config: {discovery.baseline_config_path or 'not set'}")
    print(f"- Source: {discovery.source}" + (f" ({discovery.provider}:{discovery.model})" if discovery.source == "llm" else ""))
    print(f"- Candidates: {len(candidates)}")
    print(f"- Suggestions: {len(suggestions)}")
    for warning in discovery.warnings:
        print(f"- Warning: {warning}")
    if result["apply"]:
        print(f"- Applied: {len(result['applied'])} to {Path(result['tuning_config_path']).relative_to(result['project_root'])}")
        for item in result.get("skipped", []):
            print(f"- Skipped: {item.name} ({item.target}) - {item.reason}")
    print("")
    print(yaml.safe_dump(suggestions, sort_keys=False, allow_unicode=True), end="")


def _write_if_changed(path: Path, content: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def _launch_read_targets(project_root: Path) -> list[Path]:
    targets: list[Path] = []
    for name in ("README.md", "README.rst", "README.txt", "readme.md", "train.py", "main.py", "run.py"):
        path = project_root / name
        if path.is_file():
            targets.append(path)
    for pattern in ("*.yaml", "*.yml", "*.json", "*.toml"):
        for path in sorted(project_root.glob(pattern)):
            if path.is_file() and not path.name.startswith(".") and path not in targets:
                targets.append(path)
            if len(targets) >= 12:
                return targets
    return targets


def _render_context_markdown(context: ProjectContext) -> str:
    sections = [
        ("Project Summary", context.project_summary),
        ("Training Entrypoint", context.training_entrypoint_summary),
        ("Data And Logs", context.data_summary),
        ("Parameters", context.parameter_summary),
        ("Result Reading", context.result_reading_summary),
    ]
    lines = ["# Trainee Project Context", ""]
    for title, body in sections:
        lines.extend([f"## {title}", "", body or "Not detected yet.", ""])
    if context.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in context.warnings)
        lines.append("")
    return "\n".join(lines)


def _render_project_yaml(config: Any) -> str:
    return render_project_config_yaml(config)


def _render_launch_readme(project_root: Path, discovery: Any) -> str:
    entrypoints = ", ".join(f"`{item}`" for item in discovery.entrypoints) or "none detected"
    data_dirs = ", ".join(f"`{item}`" for item in discovery.data_dirs) or "none detected"
    config_files = ", ".join(f"`{item}`" for item in discovery.config_files) or "none detected"
    limit_flags = ", ".join(f"`{item.flag} {item.value}`" for item in discovery.limit_flags) or "none detected"
    return "\n".join(
        [
            "# Trainee Project Files",
            "",
            f"Project root: `{project_root}`",
            "",
            "- `project.yaml`: stable project/run contract.",
            "- `tuning.yaml`: tunable parameter whitelist and tuning strategy.",
            "- `context.md`: generated project understanding for review.",
            "- `logs/`, `runs/`, and `artifacts/`: writable runtime outputs for guarded runs.",
            "- The decision system prompt is global in `~/.trainee/config.json` and editable in the Web UI.",
            "",
            "Detected candidates:",
            "",
            f"- Environment: `{discovery.environment}`" + (f" (`{discovery.env_name}`)" if discovery.env_name else ""),
            f"- Entrypoints: {entrypoints}",
            f"- Data directories: {data_dirs}",
            f"- Config files: {config_files}",
            f"- Training-limit arguments: {limit_flags}",
            "",
            "Recommended workflow:",
            "",
            "1. Edit `project.yaml`: select `launch.baseline_config` and confirm data, environment, command, fixed limits, and metrics.",
            "2. Review `context.md` for the generated project understanding.",
            "3. Set `output.config_path`, run `trainee prepare` to fill an empty `tuning.yaml`, then review both files.",
            "4. Run `trainee doctor` or `trainee run --dry-run` and inspect the baseline command.",
            "5. Run `trainee run`.",
            "",
            "`run.fixed_args` are constant in every round. Only `tuning.yaml` params may be changed by the agent.",
            "Fixed arguments exclude matching tunable names, flags, and config path keys from discovery.",
            "Detected config files are suggestions only; Trainee never selects one automatically.",
            "Use `advanced.shell_command` only when the structured launcher cannot express the command.",
            "Guarded runs make the host filesystem read-only and keep writes inside this `.trainee/` directory.",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
