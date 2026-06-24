from __future__ import annotations

import argparse
import asyncio
import json
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

import httpx
import uvicorn

from trainee.context_builder import ContextBuilder
from trainee.doctor import format_doctor_report, run_doctor
from trainee.events import EventBus
from trainee.models import ProjectContext, ProjectSpec, PromptPreset
from trainee.orchestrator import RuntimeService
from trainee.providers import ProviderSettingsUpdate
from trainee.settings import load_settings
from trainee.storage import Storage


DEFAULT_BASE_URL = "http://127.0.0.1:8000"


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
        description="Register an external training project and rebuild its project context.",
        method="POST",
        path="/api/project/register",
        input_schema=ProjectSpec.model_json_schema(),
    ),
    ToolDefinition(
        name="project_update_context",
        description="Replace the saved project context after manual or agent edits.",
        method="POST",
        path="/api/project/context",
        input_schema=ProjectContext.model_json_schema(),
    ),
    ToolDefinition(
        name="project_get",
        description="Read the saved project spec, context, and loop snapshot.",
        method="GET",
        path="/api/project",
        input_schema=_empty_schema(),
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
            result = init_project(Path(args.project_root), force=args.force)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _print_init_result(result)
        return 0

    if command == "run":
        try:
            security_mode = "unsafe" if args.unsafe else "guarded"
            result = asyncio.run(run_project(Path(args.project_root), security_mode=security_mode))
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _print_run_result(result)
        return 0 if result["status"] != "failed" else 1

    if command == "doctor":
        report = run_doctor(Path(args.project_root))
        print(format_doctor_report(report), end="")
        return 1 if report.has_failures else 0

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


def init_project(project_root: Path, force: bool = False) -> dict[str, Any]:
    project_root = project_root.expanduser().resolve()
    if not project_root.exists() or not project_root.is_dir():
        raise ValueError(f"project root does not exist or is not a directory: {project_root}")

    trainee_dir = project_root / ".trainee"
    trainee_dir.mkdir(parents=True, exist_ok=True)
    (trainee_dir / "runs").mkdir(parents=True, exist_ok=True)
    (trainee_dir / "logs").mkdir(parents=True, exist_ok=True)

    spec = ProjectSpec(
        project_root=str(project_root),
        working_dir=str(project_root),
        launcher_template=_default_launcher_template(project_root),
        log_paths=[".trainee/logs/**/*.log", ".trainee/runs/**/*.log"],
        signal_sources=[
            {"type": "stdout"},
            {"type": "log_file_mtime", "paths": [".trainee/logs/**/*.log", ".trainee/runs/**/*.log"]},
        ],
    )
    context = ContextBuilder().build(spec)
    files_read = _launch_read_targets(project_root)

    outputs = {
        trainee_dir / "project.json": json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        trainee_dir / "context.md": _render_context_markdown(context),
        trainee_dir / "README.md": _render_launch_readme(project_root),
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
        "files_read": files_read,
        "files_written": written,
        "files_skipped": skipped,
        "already_initialized": already_initialized,
        "launcher_template": spec.launcher_template,
        "warnings": context.warnings,
    }


def launch_project(project_root: Path, force: bool = False) -> dict[str, Any]:
    return init_project(project_root, force=force)


async def run_project(project_root: Path, security_mode: str = "guarded") -> dict[str, Any]:
    project_root = project_root.expanduser().resolve()
    spec_path = project_root / ".trainee" / "project.json"
    if not spec_path.is_file():
        raise ValueError(f"project config not found: {spec_path}; run `trainee init` first")

    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{spec_path} must contain a JSON object")
    spec = ProjectSpec.model_validate(payload).model_copy(update={"security_mode": security_mode})

    settings = load_settings(repo_root=project_root, project_root=project_root)
    storage = Storage(settings.database_path)
    runtime = RuntimeService(settings, storage, EventBus())
    try:
        await runtime.register_project(spec)
        snapshot = await runtime.start_loop()
        while runtime.loop_is_running():
            await asyncio.sleep(0.2)
            snapshot = storage.get_loop_snapshot()
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
        storage.close()


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

    init_parser = subparsers.add_parser(
        "init",
        aliases=["launch"],
        help="Initialize Trainee project files in a training project directory.",
    )
    init_parser.add_argument("project_root", nargs="?", default=".", help="Training project directory. Defaults to the current directory.")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing files under .trainee.")
    init_parser.set_defaults(command="init")

    run_parser = subparsers.add_parser("run", help="Run the training loop from .trainee/project.json.")
    run_parser.add_argument("project_root", nargs="?", default=".", help="Training project directory. Defaults to the current directory.")
    security_group = run_parser.add_mutually_exclusive_group()
    security_group.add_argument("--guarded", action="store_true", help="Run with the default bubblewrap sandbox.")
    security_group.add_argument("--unsafe", action="store_true", help="Run without bubblewrap isolation.")
    run_parser.set_defaults(command="run")

    doctor_parser = subparsers.add_parser("doctor", help="Check whether a training project is ready for Trainee.")
    doctor_parser.add_argument("project_root", nargs="?", default=".", help="Training project directory. Defaults to the current directory.")
    doctor_parser.set_defaults(command="doctor")

    tools_parser = subparsers.add_parser("tools", help="Print tool-call compatible function schemas.")
    tools_parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL to include in the manifest.")
    tools_parser.add_argument("--name", choices=sorted(TOOL_INDEX), help="Print only one tool definition.")
    tools_parser.set_defaults(command="tools")

    call_parser = subparsers.add_parser("call", aliases=["tool"], help="Call one tool against a running Trainee service.")
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
    print("Trainee init")
    print(f"- Project: {project_root}")
    if result["already_initialized"] and not result["files_written"]:
        print("- Status: already initialized; kept existing project files")
    for path in result["files_read"]:
        print(f"- Read: {Path(path).relative_to(project_root)}")
    if not result["files_read"]:
        print("- Read: no README, entrypoint, or config candidates found")
    for path in result["files_written"]:
        print(f"- Wrote: {Path(path).relative_to(project_root)}")
    for path in result["files_skipped"]:
        print(f"- Kept: {Path(path).relative_to(project_root)}")
    launcher = result["launcher_template"] or "set launcher_template in .trainee/project.json"
    print(f"- Launcher: {launcher}")
    print("- Next: review .trainee/context.md and .trainee/project.json, then run `trainee run`")
    for warning in result["warnings"]:
        print(f"- Warning: {warning}")


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


def _default_launcher_template(project_root: Path) -> str:
    for relative_path in ("train.py", "main.py", "run.py"):
        if (project_root / relative_path).is_file():
            return f"python {{project_root}}/{relative_path} {{extra_args}}"
    return ""


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


def _render_launch_readme(project_root: Path) -> str:
    return "\n".join(
        [
            "# Trainee Project Files",
            "",
            f"Project root: `{project_root}`",
            "",
            "- `project.json`: editable project registration draft.",
            "- `context.md`: generated project understanding for review.",
            "- `logs/`, `runs/`, and `artifacts/`: writable runtime outputs for guarded runs.",
            "",
            "Recommended workflow:",
            "",
            "1. Review and edit `context.md` so Trainee has the right project goal, constraints, and evaluation criteria.",
            "2. Review and edit `project.json`, especially `launcher_template`, `tunable_params`, and `metric_specs`.",
            "3. Run `trainee doctor` to check the project setup.",
            "4. Run `trainee run` after the draft looks correct.",
            "",
            "Launcher template variables: `{project_root}`, `{working_dir}`, `{trainee_dir}`, `{session_id}`, `{round_index}`, `{session_dir}`, `{round_dir}`, `{config_path}`, `{extra_args}`.",
            "`{session_dir}` and `{round_dir}` point inside `.trainee/runs/`; `{config_path}` defaults to `{round_dir}/config.yaml`.",
            "Guarded runs make the host filesystem read-only and keep writes inside this `.trainee/` directory.",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
