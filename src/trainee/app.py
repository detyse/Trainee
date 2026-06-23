from __future__ import annotations

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from trainee.events import EventBus
from trainee.models import ProjectContext, ProjectSpec
from trainee.orchestrator import RuntimeService
from trainee.settings import Settings, load_settings
from trainee.storage import Storage

MAX_LLM_TEST_IMAGE_BYTES = 5 * 1024 * 1024


def build_app(settings: Optional[Settings] = None) -> FastAPI:                  # turn fastapi
    app_settings = settings or load_settings()
    templates = Jinja2Templates(directory=str(app_settings.template_dir))       # 

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app_settings.data_dir.mkdir(parents=True, exist_ok=True)
        app_settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
        storage = Storage(app_settings.database_path)
        event_bus = EventBus()
        runtime = RuntimeService(app_settings, storage, event_bus)
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

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request, run_id: Optional[int] = None) -> HTMLResponse:
        runtime = get_runtime(request)
        payload = runtime.dashboard_payload(selected_run_id=run_id)
        return templates.TemplateResponse(request, "index.html", {"request": request, **payload})

    @app.get("/llm-test", response_class=HTMLResponse)
    async def llm_test(request: Request) -> HTMLResponse:
        settings = request.app.state.settings
        model = _llm_display_model(settings)
        configured = (
            settings.llm_provider == "openai" and bool(settings.openai_api_key)
        ) or (
            settings.llm_provider == "anthropic" and bool(settings.anthropic_api_key)
        )
        return templates.TemplateResponse(
            request,
            "llm_test.html",
            {
                "request": request,
                "provider": settings.llm_provider,
                "model": model,
                "configured": configured,
                "max_image_mb": MAX_LLM_TEST_IMAGE_BYTES // (1024 * 1024),
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

    @app.post("/api/project/register")
    async def api_register_project(request: Request, spec: ProjectSpec) -> JSONResponse:
        runtime = get_runtime(request)
        bundle = await runtime.register_project(spec)
        return JSONResponse(bundle.model_dump(mode="json"))

    @app.post("/api/project/context")
    async def api_update_context(request: Request, context: ProjectContext) -> JSONResponse:
        runtime = get_runtime(request)
        bundle = await runtime.update_project_context(context)
        return JSONResponse(bundle.model_dump(mode="json"))

    @app.get("/api/project")
    async def api_get_project(request: Request) -> JSONResponse:
        runtime = get_runtime(request)
        return JSONResponse(runtime.get_bundle().model_dump(mode="json"))

    @app.post("/api/loop/start")
    async def api_start_loop(request: Request) -> JSONResponse:
        runtime = get_runtime(request)
        try:
            snapshot = await runtime.start_loop()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(snapshot.model_dump(mode="json"))

    @app.post("/api/loop/stop")
    async def api_stop_loop(request: Request) -> JSONResponse:
        runtime = get_runtime(request)
        snapshot = await runtime.stop_loop()
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
        image: Optional[UploadFile] = File(None),
    ) -> JSONResponse:
        runtime = get_runtime(request)
        try:
            image_payload = await _read_llm_test_image(image)
            result = await runtime.decision_engine.probe(prompt, image=image_payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return JSONResponse(result)

    @app.post("/ui/project/register")
    async def ui_register_project(
        request: Request,
        project_root: str = Form(...),
        working_dir: str = Form(...),
        launcher_template: str = Form(...),
        data_paths_json: str = Form("[]"),
        log_paths_json: str = Form("[]"),
        wandb_enabled: Optional[str] = Form(None),
        heartbeat_interval_sec: float = Form(5.0),
        stall_timeout_sec: float = Form(120.0),
        max_rounds: int = Form(3),
        tunable_params_json: str = Form("[]"),
        metric_specs_json: str = Form("[]"),
        metric_prompt: str = Form(""),
        tuning_prompt: str = Form(""),
    ) -> HTMLResponse:
        runtime = get_runtime(request)
        try:
            spec = ProjectSpec(
                project_root=project_root,
                working_dir=working_dir,
                launcher_template=launcher_template,
                data_paths=_parse_json_field(data_paths_json, "data_paths_json"),
                log_paths=_parse_json_field(log_paths_json, "log_paths_json"),
                wandb_enabled=wandb_enabled is not None,
                heartbeat_interval_sec=heartbeat_interval_sec,
                stall_timeout_sec=stall_timeout_sec,
                max_rounds=max_rounds,
                tunable_params=_parse_json_field(tunable_params_json, "tunable_params_json"),
                metric_specs=_parse_json_field(metric_specs_json, "metric_specs_json"),
                metric_prompt=metric_prompt,
                tuning_prompt=tuning_prompt,
            )
            await runtime.register_project(spec)
        except ValueError as exc:
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


def _parse_json_field(raw: str, field_name: str) -> Any:
    try:
        return json.loads(raw or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON") from exc


def _format_sse(event_name: str, payload: Dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _llm_display_model(settings: Settings) -> str:
    if settings.llm_provider == "openai":
        return settings.openai_model
    if settings.llm_provider == "anthropic":
        return settings.anthropic_model
    return "none"


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
        return templates.TemplateResponse(request, "partials/oob_dashboard.html", {"request": request, **payload})
    return RedirectResponse("/", status_code=303)


app = build_app()
