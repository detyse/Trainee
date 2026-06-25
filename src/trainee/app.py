from __future__ import annotations

import asyncio
import base64
import html
import json
import shlex
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.datastructures import FormData

from trainee.events import EventBus
from trainee.logging import configure_logging, get_logger
from trainee.models import ProjectContext, ProjectSpec, PromptPreset
from trainee.orchestrator import RuntimeService
from trainee.project_config import (
    AdvancedConfig,
    CommandArg,
    DataInput,
    LaunchConfig,
    MetricsConfig,
    ProjectConfig,
    ProjectRegistration,
    RunConfig,
    TuningConfig,
    compile_project_spec,
    load_project_config,
    normalized_project_config,
    project_config_path,
    project_config_from_spec,
    restore_project_config,
    save_project_config,
)
from trainee.providers import (
    DEFAULT_ANTHROPIC_BASE_URL,
    DEFAULT_ANTHROPIC_MAX_TOKENS,
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_ANTHROPIC_VERSION,
    DEFAULT_LLM_TIMEOUT_SEC,
    DEFAULT_MOONSHOT_BASE_URL,
    DEFAULT_MOONSHOT_MODEL,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    AgentDebugSettingsUpdate,
    ProviderSettingsUpdate,
    SystemPromptUpdate,
    active_model,
    build_provider_config_payload,
    provider_is_configured,
    provider_settings_payload,
    provider_update_from_form,
)
from trainee.reporter import ReportGenerator
from trainee.settings import Settings, load_settings, save_global_config
from trainee.storage import ImageAnalysisLimitExceeded, Storage
from trainee.tunable_discovery import (
    TunableDiscoveryApply,
    TunableDiscoveryEngine,
    TunableDiscoveryRequest,
    apply_tunable_suggestions,
)

MAX_LLM_TEST_IMAGE_BYTES = 5 * 1024 * 1024
logger = get_logger(__name__)


def _yaml_dump(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if hasattr(item, "model_dump"):
            return normalize(item.model_dump(mode="json", exclude_none=True))
        if isinstance(item, dict):
            return {key: normalize(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        return item

    return yaml.safe_dump(
        normalize(value),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).rstrip()


def build_app(settings: Optional[Settings] = None) -> FastAPI:                  # turn fastapi
    app_settings = settings or load_settings()
    templates = Jinja2Templates(directory=str(app_settings.template_dir))       # 
    templates.env.filters["toyaml"] = _yaml_dump

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app_settings.data_dir.mkdir(parents=True, exist_ok=True)
        app_settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
        storage = Storage(app_settings.database_path)
        event_bus = EventBus()
        runtime = RuntimeService(app_settings, storage, event_bus)
        app.state.started_at_monotonic = time.monotonic()
        app.state.settings = app_settings
        app.state.storage = storage
        app.state.event_bus = event_bus
        app.state.runtime = runtime
        try:
            yield
        finally:
            storage.close()

    app = FastAPI(title="Trainee", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(app_settings.static_dir)), name="static")

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        started = time.monotonic()
        response = await call_next(request)
        logger.info(
            "request",
            extra={
                "_trainee_extra": {
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round((time.monotonic() - started) * 1000, 2),
                }
            },
        )
        return response

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request, run_id: Optional[int] = None) -> HTMLResponse:
        runtime = get_runtime(request)
        payload = runtime.dashboard_payload(selected_run_id=run_id)
        return templates.TemplateResponse(request, "index.html", {"request": request, **payload, "health": _health_payload(request)})

    @app.get("/llm-test", response_class=HTMLResponse)
    async def llm_test(request: Request) -> HTMLResponse:
        settings = request.app.state.settings
        runtime = get_runtime(request)
        latest_session = runtime.storage.get_latest_session()
        return templates.TemplateResponse(
            request,
            "llm_test.html",
            {
                "request": request,
                "provider": settings.llm_provider,
                "model": active_model(settings),
                "configured": provider_is_configured(settings),
                "max_image_mb": MAX_LLM_TEST_IMAGE_BYTES // (1024 * 1024),
                "image_analysis_session_id": latest_session.id if latest_session else None,
                "image_analysis_limit": settings.max_image_analyses_per_session,
            },
        )

    @app.get("/fragments/project", response_class=HTMLResponse)
    async def project_fragment(request: Request) -> HTMLResponse:
        runtime = get_runtime(request)
        payload = runtime.dashboard_payload()
        return templates.TemplateResponse(request, "partials/project_section.html", {"request": request, **payload})

    @app.get("/fragments/context", response_class=HTMLResponse)
    async def context_fragment(request: Request) -> HTMLResponse:
        runtime = get_runtime(request)
        payload = runtime.dashboard_payload()
        return templates.TemplateResponse(request, "partials/context_section.html", {"request": request, **payload})

    @app.get("/fragments/loop", response_class=HTMLResponse)
    async def loop_fragment(request: Request) -> HTMLResponse:
        runtime = get_runtime(request)
        payload = runtime.dashboard_payload()
        return templates.TemplateResponse(request, "partials/loop_section.html", {"request": request, **payload})

    @app.get("/fragments/runs", response_class=HTMLResponse)
    async def runs_fragment(request: Request) -> HTMLResponse:
        runtime = get_runtime(request)
        payload = runtime.dashboard_payload()
        return templates.TemplateResponse(request, "partials/runs_section.html", {"request": request, **payload})

    @app.get("/fragments/run-detail", response_class=HTMLResponse)
    async def run_detail_fragment(request: Request, run_id: Optional[int] = None) -> HTMLResponse:
        runtime = get_runtime(request)
        payload = runtime.dashboard_payload(selected_run_id=run_id)
        return templates.TemplateResponse(request, "partials/run_detail_section.html", {"request": request, **payload})

    @app.get("/fragments/prompt", response_class=HTMLResponse)
    async def prompt_fragment(request: Request, run_id: Optional[int] = None) -> HTMLResponse:
        runtime = get_runtime(request)
        payload = runtime.dashboard_payload(selected_run_id=run_id)
        return templates.TemplateResponse(request, "partials/prompt_section.html", {"request": request, **payload})

    @app.get("/fragments/runtime", response_class=HTMLResponse)
    async def runtime_fragment(request: Request) -> HTMLResponse:
        runtime = get_runtime(request)
        payload = runtime.dashboard_payload()
        return templates.TemplateResponse(
            request,
            "partials/runtime_section.html",
            {"request": request, **payload, "health": _health_payload(request)},
        )

    @app.post("/api/project/register")
    async def api_register_project(request: Request, payload: Dict[str, Any]) -> JSONResponse:
        runtime = get_runtime(request)
        try:
            if "launch" in payload:
                registration = ProjectRegistration.model_validate(payload)
                project_root = Path(registration.project_root).expanduser().resolve()
                config = ProjectConfig.model_validate(registration.model_dump(exclude={"project_root"}))
            else:
                legacy_spec = ProjectSpec.model_validate(payload)
                project_root = Path(legacy_spec.project_root).expanduser().resolve()
                config = project_config_from_spec(legacy_spec)
            bundle = await _register_project_config(runtime, project_root, config)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(bundle.model_dump(mode="json"))

    @app.post("/api/project/context")
    async def api_update_context(request: Request, context: ProjectContext) -> JSONResponse:
        runtime = get_runtime(request)
        bundle = await runtime.update_project_context(context)
        return JSONResponse(bundle.model_dump(mode="json"))

    @app.get("/api/runtime/system-prompt")
    async def api_get_system_prompt(request: Request) -> JSONResponse:
        runtime = get_runtime(request)
        return JSONResponse(
            {
                "system_prompt": runtime.settings.system_prompt,
                "config_path": str(runtime.settings.global_config_path),
            }
        )

    @app.post("/api/runtime/system-prompt")
    async def api_update_system_prompt(request: Request, update: SystemPromptUpdate) -> JSONResponse:
        runtime = get_runtime(request)
        try:
            updated_settings = _update_system_prompt_settings(request, runtime, update.system_prompt)
        except OSError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            status_code = 409 if runtime.loop_is_running() else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc
        return JSONResponse(
            {
                "system_prompt": updated_settings.system_prompt,
                "config_path": str(updated_settings.global_config_path),
            }
        )

    @app.get("/api/project")
    async def api_get_project(request: Request) -> JSONResponse:
        runtime = get_runtime(request)
        payload = runtime.dashboard_payload()
        bundle = runtime.get_bundle().model_dump(mode="json")
        bundle["config"] = payload["project_config"].model_dump(mode="json") if payload["project_config"] else None
        bundle["config_path"] = payload["project_config_path"]
        return JSONResponse(bundle)

    @app.post("/api/project/tunables/suggest")
    async def api_suggest_tunables(request: Request, payload: Optional[Dict[str, Any]] = None) -> JSONResponse:
        runtime = get_runtime(request)
        try:
            project_root, config, limit = _project_config_from_discovery_payload(runtime, payload)
            normalized = normalized_project_config(project_root, config)
            spec = compile_project_spec(project_root, normalized)
            context = runtime.context_builder.build(spec)
            result = await TunableDiscoveryEngine(runtime.settings).suggest(
                spec,
                context,
                limit=limit,
                fixed_args=config.run.fixed_args,
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(result.model_dump(mode="json"))

    @app.post("/api/project/tunables/apply")
    async def api_apply_tunables(request: Request, payload: TunableDiscoveryApply) -> JSONResponse:
        runtime = get_runtime(request)
        try:
            project_root = Path(payload.project_root).expanduser().resolve()
            config = load_project_config(project_root)
            updated, applied = apply_tunable_suggestions(
                config,
                payload.suggestions,
                replace=payload.replace,
            )
            bundle = await _register_project_config(runtime, project_root, updated)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(
            {
                "applied": [item.model_dump(mode="json") for item in applied],
                "bundle": bundle.model_dump(mode="json"),
            }
        )

    @app.get("/api/health")
    async def api_health(request: Request) -> JSONResponse:
        return JSONResponse(_health_payload(request))

    @app.get("/api/runtime/provider")
    async def api_get_provider_settings(request: Request) -> JSONResponse:
        runtime = get_runtime(request)
        return JSONResponse(provider_settings_payload(runtime.settings))

    @app.post("/api/runtime/provider")
    async def api_update_provider_settings(
        request: Request,
        update: ProviderSettingsUpdate,
    ) -> JSONResponse:
        runtime = get_runtime(request)
        try:
            updated_settings = _update_provider_settings(request, runtime, update)
        except OSError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            status_code = 409 if runtime.loop_is_running() else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc
        return JSONResponse(provider_settings_payload(updated_settings))

    @app.get("/api/runtime/debug")
    async def api_get_agent_debug_settings(request: Request) -> JSONResponse:
        runtime = get_runtime(request)
        return JSONResponse({"agent_debug_enabled": runtime.settings.agent_debug_enabled})

    @app.post("/api/runtime/debug")
    async def api_update_agent_debug_settings(
        request: Request,
        update: AgentDebugSettingsUpdate,
    ) -> JSONResponse:
        runtime = get_runtime(request)
        try:
            updated_settings = _update_agent_debug_settings(request, runtime, update.agent_debug_enabled)
        except OSError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            status_code = 409 if runtime.loop_is_running() else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc
        return JSONResponse({"agent_debug_enabled": updated_settings.agent_debug_enabled})

    @app.get("/api/prompt-preview")
    async def api_get_prompt_preview(request: Request, run_id: Optional[int] = None) -> JSONResponse:
        runtime = get_runtime(request)
        payload = runtime.dashboard_payload(selected_run_id=run_id)
        preview = payload["prompt_preview"]
        return JSONResponse(
            {
                "label": payload["prompt_preview_label"],
                "prompt_preview": preview.model_dump(mode="json") if preview else None,
            }
        )

    @app.get("/api/prompt-presets")
    async def api_get_prompt_presets(request: Request, project_root: Optional[str] = None) -> JSONResponse:
        runtime = get_runtime(request)
        presets = runtime.storage.list_prompt_presets(project_root)
        return JSONResponse({"presets": [item.model_dump(mode="json") for item in presets]})

    @app.post("/api/prompt-presets")
    async def api_save_prompt_preset(request: Request, preset: PromptPreset) -> JSONResponse:
        runtime = get_runtime(request)
        saved = await runtime.save_prompt_preset(
            name=preset.name,
            metric_prompt=preset.metric_prompt,
            tuning_prompt=preset.tuning_prompt,
            project_root=preset.project_root,
            preset_id=preset.id,
        )
        return JSONResponse(saved.model_dump(mode="json"))

    @app.post("/api/loop/start")
    async def api_start_loop(request: Request, resume_session_id: Optional[int] = None) -> JSONResponse:
        runtime = get_runtime(request)
        with suppress(Exception):
            payload = await request.json()
            if isinstance(payload, dict) and payload.get("resume_session_id") is not None:
                resume_session_id = int(payload["resume_session_id"])
        try:
            snapshot = await runtime.start_loop(resume_session_id=resume_session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(snapshot.model_dump(mode="json"))

    @app.post("/api/loop/stop")
    async def api_stop_loop(request: Request, force: bool = False) -> JSONResponse:
        runtime = get_runtime(request)
        with suppress(Exception):
            payload = await request.json()
            if isinstance(payload, dict) and "force" in payload:
                force = _coerce_bool(payload["force"])
        snapshot = await runtime.stop_loop(force=force)
        return JSONResponse(snapshot.model_dump(mode="json"))

    @app.get("/api/loop")
    async def api_get_loop(request: Request) -> JSONResponse:
        runtime = get_runtime(request)
        return JSONResponse(runtime.get_bundle().loop.model_dump(mode="json"))

    @app.get("/api/runs")
    async def api_get_runs(request: Request) -> JSONResponse:
        runtime = get_runtime(request)
        payload = runtime.dashboard_payload()
        return JSONResponse(
            {
                "sessions": [item.model_dump(mode="json") for item in payload["sessions"]],
                "rounds": [item.model_dump(mode="json") for item in payload["rounds"]],
                "loop": payload["loop"].model_dump(mode="json"),
            }
        )

    @app.get("/api/runs/{run_id}")
    async def api_get_run(request: Request, run_id: int) -> JSONResponse:
        runtime = get_runtime(request)
        payload = runtime.dashboard_payload(selected_run_id=run_id)
        selected_run = payload["selected_run"]
        if selected_run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return JSONResponse(selected_run.model_dump(mode="json"))

    @app.get("/api/sessions/{session_id}/report")
    async def api_get_session_report(request: Request, session_id: int, format: str = "markdown"):
        runtime = get_runtime(request)
        try:
            report = ReportGenerator(runtime.storage).generate_session_report(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if format == "html":
            return HTMLResponse(f"<pre>{html.escape(report)}</pre>")
        return PlainTextResponse(report, media_type="text/markdown")

    @app.get("/api/events")
    async def api_events(request: Request) -> StreamingResponse:
        runtime = get_runtime(request)
        event_bus = runtime.event_bus

        async def event_stream() -> AsyncIterator[str]:
            queue = await event_bus.subscribe()
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15)
                        yield _format_sse("message", event.model_dump(mode="json"))
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                    if await request.is_disconnected():
                        break
            finally:
                await event_bus.unsubscribe(queue)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/api/llm/test")
    async def api_llm_test(
        request: Request,
        prompt: str = Form(...),
        session_id: Optional[int] = Form(None),
        image: Optional[UploadFile] = File(None),
    ) -> JSONResponse:
        runtime = get_runtime(request)
        try:
            image_payload = await _read_llm_test_image(image)
            image_analysis_usage = runtime.reserve_image_analysis(session_id) if image_payload is not None else None
            result = await runtime.decision_engine.probe(prompt, image=image_payload)
            if image_analysis_usage is not None:
                result["image_analysis"] = image_analysis_usage
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ImageAnalysisLimitExceeded as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return JSONResponse(result)

    @app.post("/ui/project/register")
    async def ui_register_project(request: Request) -> HTMLResponse:
        runtime = get_runtime(request)
        form = await request.form()
        try:
            project_root, config = _project_config_from_form(form)
            review_response = await _render_tunable_review_response(
                request,
                templates,
                runtime,
                project_root,
                config,
                reviewed=_form_checkbox(form, "tunable_reviewed"),
            )
            if review_response is not None:
                return review_response
            await _register_project_config(runtime, project_root, config)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return await _render_refresh_response(request, templates, runtime)

    @app.post("/ui/project/tunables/suggest")
    async def ui_suggest_tunables(request: Request) -> HTMLResponse:
        runtime = get_runtime(request)
        form = await request.form()
        try:
            project_root, config = _project_config_from_form(form)
            normalized = normalized_project_config(project_root, config)
            spec = compile_project_spec(project_root, normalized)
            context = runtime.context_builder.build(spec)
            result = await TunableDiscoveryEngine(runtime.settings).suggest(
                spec,
                context,
                fixed_args=config.run.fixed_args,
            )
            suggested_config, _ = apply_tunable_suggestions(config, result.suggestions)
            suggested_spec = compile_project_spec(project_root, normalized_project_config(project_root, suggested_config))
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        payload = runtime.dashboard_payload()
        payload["project_config"] = suggested_config
        payload["spec"] = suggested_spec
        payload["project_config_path"] = str(project_config_path(project_root))
        payload["tunable_discovery"] = result
        return templates.TemplateResponse(request, "partials/project_section.html", {"request": request, **payload})

    @app.post("/ui/prompt-presets/save")
    async def ui_save_prompt_preset(request: Request) -> HTMLResponse:
        runtime = get_runtime(request)
        form = await request.form()
        try:
            project_root, config = _project_config_from_form(form)
            bundle = await _register_project_config(runtime, project_root, config)
            spec = bundle.spec
            if spec is None:
                raise ValueError("project registration did not produce a project spec")
            preset_id = _form_str(form, "prompt_preset_id")
            preset_name = _form_str(form, "prompt_preset_name")
            if not preset_name and preset_id:
                existing = runtime.storage.get_prompt_preset(preset_id)
                preset_name = existing.name if existing else ""
            await runtime.save_prompt_preset(
                name=preset_name or "Default Prompt",
                metric_prompt=spec.metric_prompt,
                tuning_prompt=spec.tuning_prompt,
                project_root=spec.project_root,
                preset_id=preset_id or None,
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return await _render_refresh_response(request, templates, runtime)

    @app.post("/ui/prompt-presets/apply")
    async def ui_apply_prompt_preset(request: Request) -> HTMLResponse:
        runtime = get_runtime(request)
        form = await request.form()
        preset_id = _form_str(form, "prompt_preset_id")
        if not preset_id:
            raise HTTPException(status_code=400, detail="select a prompt preset first")
        preset = runtime.storage.get_prompt_preset(preset_id)
        if preset is None:
            raise HTTPException(status_code=404, detail="prompt preset not found")
        try:
            project_root, config = _project_config_from_form(form)
            config.metrics.prompt = preset.metric_prompt
            config.advanced.tuning_prompt = preset.tuning_prompt
            await _register_project_config(runtime, project_root, config)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return await _render_refresh_response(request, templates, runtime)

    @app.post("/ui/project/context")
    async def ui_update_context(
        request: Request,
        project_summary: str = Form(""),
        training_entrypoint_summary: str = Form(""),
        data_summary: str = Form(""),
        parameter_summary: str = Form(""),
        result_reading_summary: str = Form(""),
        warnings_json: str = Form("[]"),
    ) -> HTMLResponse:
        runtime = get_runtime(request)
        context = ProjectContext(
            project_summary=project_summary,
            training_entrypoint_summary=training_entrypoint_summary,
            data_summary=data_summary,
            parameter_summary=parameter_summary,
            result_reading_summary=result_reading_summary,
            warnings=_parse_json_field(warnings_json, "warnings_json"),
        )
        await runtime.update_project_context(context)
        return await _render_refresh_response(request, templates, runtime)

    @app.post("/ui/runtime/system-prompt")
    async def ui_update_system_prompt(
        request: Request,
        system_prompt: str = Form(""),
    ) -> HTMLResponse:
        runtime = get_runtime(request)
        try:
            _update_system_prompt_settings(request, runtime, system_prompt)
        except OSError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            status_code = 409 if runtime.loop_is_running() else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc
        return await _render_refresh_response(request, templates, runtime)

    @app.post("/ui/runtime/provider")
    async def ui_update_provider_settings(
        request: Request,
        llm_provider: str = Form("none"),
        llm_timeout_sec: float = Form(DEFAULT_LLM_TIMEOUT_SEC),
        openai_api_key: str = Form(""),
        clear_openai_api_key: Optional[str] = Form(None),
        openai_base_url: str = Form(DEFAULT_OPENAI_BASE_URL),
        openai_model: str = Form(DEFAULT_OPENAI_MODEL),
        moonshot_api_key: str = Form(""),
        clear_moonshot_api_key: Optional[str] = Form(None),
        moonshot_base_url: str = Form(DEFAULT_MOONSHOT_BASE_URL),
        moonshot_model: str = Form(DEFAULT_MOONSHOT_MODEL),
        anthropic_api_key: str = Form(""),
        clear_anthropic_api_key: Optional[str] = Form(None),
        anthropic_base_url: str = Form(DEFAULT_ANTHROPIC_BASE_URL),
        anthropic_model: str = Form(DEFAULT_ANTHROPIC_MODEL),
        anthropic_version: str = Form(DEFAULT_ANTHROPIC_VERSION),
        anthropic_max_tokens: int = Form(DEFAULT_ANTHROPIC_MAX_TOKENS),
    ) -> HTMLResponse:
        runtime = get_runtime(request)
        update = provider_update_from_form(
            llm_provider=llm_provider,
            llm_timeout_sec=llm_timeout_sec,
            openai_api_key=openai_api_key,
            clear_openai_api_key=clear_openai_api_key is not None,
            openai_base_url=openai_base_url,
            openai_model=openai_model,
            moonshot_api_key=moonshot_api_key,
            clear_moonshot_api_key=clear_moonshot_api_key is not None,
            moonshot_base_url=moonshot_base_url,
            moonshot_model=moonshot_model,
            anthropic_api_key=anthropic_api_key,
            clear_anthropic_api_key=clear_anthropic_api_key is not None,
            anthropic_base_url=anthropic_base_url,
            anthropic_model=anthropic_model,
            anthropic_version=anthropic_version,
            anthropic_max_tokens=anthropic_max_tokens,
        )
        try:
            _update_provider_settings(request, runtime, update)
        except (OSError, ValueError) as exc:
            status_code = 409 if runtime.loop_is_running() else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc
        return await _render_refresh_response(request, templates, runtime)

    @app.post("/ui/runtime/debug")
    async def ui_update_agent_debug_settings(
        request: Request,
        agent_debug_enabled: Optional[str] = Form(None),
    ) -> HTMLResponse:
        runtime = get_runtime(request)
        try:
            _update_agent_debug_settings(request, runtime, agent_debug_enabled is not None)
        except (OSError, ValueError) as exc:
            status_code = 409 if runtime.loop_is_running() else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc
        return await _render_refresh_response(request, templates, runtime)

    @app.post("/ui/loop/start")
    async def ui_start_loop(request: Request) -> HTMLResponse:
        runtime = get_runtime(request)
        try:
            await runtime.start_loop()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return await _render_refresh_response(request, templates, runtime)

    @app.post("/ui/loop/stop")
    async def ui_stop_loop(request: Request) -> HTMLResponse:
        runtime = get_runtime(request)
        await runtime.stop_loop()
        return await _render_refresh_response(request, templates, runtime)

    return app


def get_runtime(request: Request) -> RuntimeService:
    return request.app.state.runtime  # type: ignore[return-value]


def _update_provider_settings(
    request: Request,
    runtime: RuntimeService,
    update: ProviderSettingsUpdate,
) -> Settings:
    if runtime.loop_is_running():
        raise ValueError("stop the loop before changing provider settings")
    provider_payload = build_provider_config_payload(update)
    save_global_config(runtime.settings.global_config_path, provider_payload)
    updated_settings = load_settings(
        repo_root=runtime.settings.repo_root,
        data_dir=runtime.settings.data_dir,
        project_root=runtime.settings.project_root,
        global_config_path=runtime.settings.global_config_path,
    )
    request.app.state.settings = updated_settings
    runtime.update_settings(updated_settings)
    return updated_settings


def _update_agent_debug_settings(
    request: Request,
    runtime: RuntimeService,
    enabled: bool,
) -> Settings:
    if runtime.loop_is_running():
        raise ValueError("stop the loop before changing Agent Debug settings")
    save_global_config(
        runtime.settings.global_config_path,
        {"agent_debug_enabled": enabled},
    )
    updated_settings = load_settings(
        repo_root=runtime.settings.repo_root,
        data_dir=runtime.settings.data_dir,
        project_root=runtime.settings.project_root,
        global_config_path=runtime.settings.global_config_path,
    )
    request.app.state.settings = updated_settings
    runtime.update_settings(updated_settings)
    return updated_settings


def _update_system_prompt_settings(
    request: Request,
    runtime: RuntimeService,
    system_prompt: str,
) -> Settings:
    if runtime.loop_is_running():
        raise ValueError("stop the loop before changing the system prompt")
    if not system_prompt.strip():
        raise ValueError("system_prompt cannot be blank")
    save_global_config(
        runtime.settings.global_config_path,
        {"system_prompt": system_prompt},
    )
    updated_settings = load_settings(
        repo_root=runtime.settings.repo_root,
        data_dir=runtime.settings.data_dir,
        project_root=runtime.settings.project_root,
        global_config_path=runtime.settings.global_config_path,
    )
    request.app.state.settings = updated_settings
    runtime.update_settings(updated_settings)
    return updated_settings


def _health_payload(request: Request) -> Dict[str, Any]:
    runtime = get_runtime(request)
    db_ok = False
    with suppress(Exception):
        db_ok = runtime.storage.ping()
    snapshot = runtime.get_bundle().loop
    uptime = time.monotonic() - getattr(request.app.state, "started_at_monotonic", time.monotonic())
    status = "healthy" if db_ok else "degraded"
    return {
        "status": status,
        "uptime_sec": round(uptime, 3),
        "loop_state": snapshot.status,
        "current_round": snapshot.current_round_index,
        "llm_provider": runtime.settings.llm_provider,
        "db_ok": db_ok,
    }


async def _render_tunable_review_response(
    request: Request,
    templates: Jinja2Templates,
    runtime: RuntimeService,
    project_root: Path,
    config: ProjectConfig,
    *,
    reviewed: bool,
) -> Optional[HTMLResponse]:
    if reviewed or config.tuning.params or not config.launch.baseline_config:
        return None

    normalized = normalized_project_config(project_root, config)
    spec = compile_project_spec(project_root, normalized)
    context = runtime.prepare_project_registration(spec)
    result = await TunableDiscoveryEngine(runtime.settings).suggest(
        spec,
        context,
        fixed_args=config.run.fixed_args,
    )
    suggested_config, applied = apply_tunable_suggestions(config, result.suggestions)
    if not applied:
        return None

    suggested_spec = compile_project_spec(project_root, normalized_project_config(project_root, suggested_config))
    payload = runtime.dashboard_payload()
    payload["project_config"] = suggested_config
    payload["spec"] = suggested_spec
    payload["project_config_path"] = str(project_config_path(project_root))
    payload["tunable_discovery"] = result
    return templates.TemplateResponse(request, "partials/project_section.html", {"request": request, **payload})


async def _register_project_config(
    runtime: RuntimeService,
    project_root: Path,
    config: ProjectConfig,
):
    normalized = normalized_project_config(project_root, config)
    spec = compile_project_spec(project_root, normalized)
    context = runtime.prepare_project_registration(spec)
    path = project_config_path(project_root)
    previous = path.read_bytes() if path.is_file() else None
    try:
        save_project_config(project_root, normalized)
        return await runtime.register_project(spec, context=context)
    except Exception:
        restore_project_config(project_root, previous)
        raise


def _project_config_from_discovery_payload(
    runtime: RuntimeService,
    payload: Optional[Dict[str, Any]],
) -> tuple[Path, ProjectConfig, int]:
    payload = payload or {}
    limit = int(payload.get("limit", TunableDiscoveryRequest().limit))
    limit = max(1, min(32, limit))
    if "launch" in payload:
        registration = ProjectRegistration.model_validate(payload)
        project_root = Path(registration.project_root).expanduser().resolve()
        return project_root, ProjectConfig.model_validate(registration.model_dump(exclude={"project_root"})), limit

    request_payload = TunableDiscoveryRequest.model_validate(payload)
    if request_payload.project_root:
        project_root = Path(request_payload.project_root).expanduser().resolve()
    else:
        bundle = runtime.get_bundle()
        if bundle.spec is not None:
            project_root = Path(bundle.spec.project_root).expanduser().resolve()
        elif runtime.settings.project_root is not None:
            project_root = runtime.settings.project_root
        else:
            raise ValueError("project_root is required before tunable discovery")
    return project_root, load_project_config(project_root), limit


def _project_config_from_form(form: FormData) -> tuple[Path, ProjectConfig]:
    project_root = Path(_form_str(form, "project_root")).expanduser().resolve()
    legacy_launcher = _form_str(form, "launcher_template")
    if legacy_launcher:
        def legacy_json(name: str) -> Any:
            try:
                return json.loads(_form_str(form, name, "[]") or "[]")
            except json.JSONDecodeError as exc:
                raise ValueError(f"{name} must be valid JSON") from exc

        spec = ProjectSpec(
            project_root=str(project_root),
            working_dir=_form_str(form, "working_dir", str(project_root)),
            launcher_template=legacy_launcher,
            security_mode=_form_str(form, "security_mode", "guarded"),
            data_paths=legacy_json("data_paths_json"),
            log_paths=legacy_json("log_paths_json"),
            signal_sources=legacy_json("signal_sources_json"),
            wandb_enabled=_form_checkbox(form, "wandb_enabled"),
            heartbeat_interval_sec=_form_float(form, "heartbeat_interval_sec", 5.0),
            stall_timeout_sec=_form_float(form, "stall_timeout_sec", 120.0),
            max_rounds=_form_int(form, "max_rounds", 3),
            tunable_params=legacy_json("tunable_params_json"),
            metric_specs=legacy_json("metric_specs_json"),
            metric_prompt=_form_str(form, "metric_prompt"),
            tuning_prompt=_form_str(form, "tuning_prompt"),
        )
        return project_root, project_config_from_spec(spec)

    command = shlex.split(_form_str(form, "launch_command"))
    data = []
    for line in _nonempty_lines(_form_str(form, "data_lines")):
        path, separator, flag = line.partition("|")
        data.append(DataInput(path=path.strip(), flag=flag.strip() if separator and flag.strip() else None))

    advanced_payload = _parse_yaml_mapping(_form_str(form, "advanced_yaml"), "advanced_yaml")
    advanced_payload.update(
        {
            "security_mode": _form_str(form, "security_mode", "guarded"),
            "working_dir": _form_str(form, "working_dir", ".") or ".",
            "wandb_enabled": _form_checkbox(form, "wandb_enabled"),
            "tuning_prompt": _form_str(form, "tuning_prompt"),
        }
    )
    timeout_raw = _form_str(form, "timeout_minutes", "60").strip()
    config = ProjectConfig(
        data=data,
        launch=LaunchConfig(
            environment=_form_str(form, "launch_environment", "system"),
            env_name=_form_str(form, "launch_env_name") or None,
            command=command,
            baseline_config=_form_str(form, "baseline_config") or None,
            args=_parse_arg_lines(_form_str(form, "launch_args_lines")),
        ),
        run=RunConfig(
            max_rounds=_form_int(form, "max_rounds", 3),
            timeout_minutes=float(timeout_raw) if timeout_raw else None,
            fixed_args=_parse_arg_lines(_form_str(form, "fixed_args_lines")),
        ),
        tuning=TuningConfig(params=_parse_yaml_list(_form_str(form, "tunable_params_yaml"), "tunable_params_yaml")),
        metrics=MetricsConfig(
            specs=_parse_yaml_list(_form_str(form, "metric_specs_yaml"), "metric_specs_yaml"),
            prompt=_form_str(form, "metric_prompt"),
        ),
        advanced=AdvancedConfig.model_validate(advanced_payload),
    )
    return project_root, config


def _parse_arg_lines(raw: str) -> list[CommandArg]:
    args: list[CommandArg] = []
    for line in _nonempty_lines(raw):
        flag, separator, value = line.partition("=")
        args.append(CommandArg(flag=flag.strip(), value=_coerce_scalar(value.strip()) if separator else None))
    return args


def _parse_yaml_list(raw: str, field_name: str) -> list[Any]:
    if not raw.strip():
        return []
    try:
        payload = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"{field_name} must be valid YAML") from exc
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError(f"{field_name} must be a YAML list")
    return payload


def _parse_yaml_mapping(raw: str, field_name: str) -> dict[str, Any]:
    if not raw.strip():
        return {}
    try:
        payload = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"{field_name} must be valid YAML") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must be a YAML mapping")
    return payload


def _nonempty_lines(raw: str) -> list[str]:
    return [line.strip() for line in raw.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _coerce_scalar(raw: str) -> Any:
    if not raw:
        return ""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _form_str(form: FormData, key: str, default: str = "") -> str:
    value = form.get(key, default)
    return str(value) if value is not None else default


def _form_float(form: FormData, key: str, default: float) -> float:
    raw = _form_str(form, key, str(default))
    return float(raw or default)


def _form_int(form: FormData, key: str, default: int) -> int:
    raw = _form_str(form, key, str(default))
    return int(raw or default)


def _form_checkbox(form: FormData, key: str) -> bool:
    value = form.get(key)
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "0", "false", "off", "none"}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"", "0", "false", "off", "none"}


def _format_sse(event_name: str, payload: Dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _read_llm_test_image(image: Optional[UploadFile]) -> Optional[Dict[str, str]]:
    if image is None or not image.filename:
        return None

    media_type = image.content_type or "application/octet-stream"
    if not media_type.startswith("image/"):
        raise ValueError("image must be an image file")

    raw = await image.read()
    if len(raw) > MAX_LLM_TEST_IMAGE_BYTES:
        raise ValueError(f"image must be {MAX_LLM_TEST_IMAGE_BYTES // (1024 * 1024)}MB or smaller")
    if not raw:
        raise ValueError("image is empty")

    return {
        "media_type": media_type,
        "data": base64.b64encode(raw).decode("ascii"),
    }


async def _render_refresh_response(request: Request, templates: Jinja2Templates, runtime: RuntimeService) -> HTMLResponse:
    if request.headers.get("HX-Request") == "true":
        payload = runtime.dashboard_payload()
        return templates.TemplateResponse(request, "partials/oob_dashboard.html", {"request": request, **payload, "health": _health_payload(request)})
    return RedirectResponse("/", status_code=303)


app = build_app()
