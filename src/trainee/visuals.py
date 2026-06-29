from __future__ import annotations

import base64
import glob
import json
import mimetypes
from pathlib import Path
from typing import Any, Callable, Optional

from trainee.decision import DecisionEngine
from trainee.executor import TrainingExecutor
from trainee.models import ProjectSpec, VisualAnalysisResult, VisualPlotObservation
from trainee.security import project_trainee_dir
from trainee.storage import ImageAnalysisLimitExceeded

MAX_VISUAL_IMAGE_BYTES = 5 * 1024 * 1024

ReserveImageAnalysis = Callable[[], Optional[dict[str, int]]]


class VisualAnalyzer:
    def __init__(
        self,
        executor: Optional[TrainingExecutor] = None,
        *,
        max_image_bytes: int = MAX_VISUAL_IMAGE_BYTES,
    ) -> None:
        self.executor = executor or TrainingExecutor()
        self.max_image_bytes = max_image_bytes

    async def analyze_round(
        self,
        *,
        spec: ProjectSpec,
        session_id: int,
        round_index: int,
        decision_engine: DecisionEngine,
        reserve_image_analysis: ReserveImageAnalysis,
    ) -> Optional[VisualAnalysisResult]:
        if not spec.visuals.enabled:
            return None

        try:
            image_paths = self.select_images(spec, session_id=session_id, round_index=round_index)
        except ValueError as exc:
            return VisualAnalysisResult(status="failed", error=str(exc))
        if not image_paths:
            return VisualAnalysisResult(status="no_images")

        usage: list[dict[str, int]] = []
        plots: list[VisualPlotObservation] = []
        errors: list[str] = []
        status = "completed"

        for path in image_paths:
            try:
                reserved = reserve_image_analysis()
                if reserved is not None:
                    usage.append(reserved)
                image = self._read_image(path)
                prompt = self._analysis_prompt(spec, path)
                response = await decision_engine.analyze_image(prompt, image)
                plots.append(self._plot_observation(spec, path, response.get("content", "")))
            except ImageAnalysisLimitExceeded as exc:
                errors.append(str(exc))
                status = "limit_reached" if not plots else "partial"
                break
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                errors.append(f"{path}: {message}")
                plots.append(
                    VisualPlotObservation(
                        path=self._display_path(spec, path),
                        error=message,
                    )
                )

        if not plots and errors and status == "completed":
            status = "failed"
        elif errors and status == "completed":
            status = "partial"

        return VisualAnalysisResult(
            status=status,
            image_paths=[self._display_path(spec, path) for path in image_paths],
            plots=plots,
            overall_visual_summary=self._overall_summary(plots),
            decision_relevant_observations=self._decision_observations(plots),
            image_analysis_usage=usage,
            error="; ".join(errors) if errors else None,
        )

    def select_images(self, spec: ProjectSpec, *, session_id: int, round_index: int) -> list[Path]:
        variables = self._template_vars(spec, session_id=session_id, round_index=round_index)
        project_root = Path(spec.project_root).expanduser().resolve()
        working_dir = Path(spec.working_dir).expanduser().resolve()
        matches: list[Path] = []

        for raw_pattern in spec.visuals.patterns:
            rendered = self._format_pattern(raw_pattern, variables)
            pattern = Path(rendered).expanduser()
            if not pattern.is_absolute():
                pattern = working_dir / pattern
            for item in glob.glob(str(pattern), recursive=True):
                path = Path(item).expanduser().resolve()
                if not path.is_file():
                    continue
                self._ensure_within(path, project_root, "visuals.patterns")
                matches.append(path)

        deduped: list[Path] = []
        seen: set[str] = set()
        for path in matches:
            marker = str(path)
            if marker in seen:
                continue
            seen.add(marker)
            deduped.append(path)

        if spec.visuals.selection == "newest":
            deduped.sort(key=self._mtime_ns, reverse=True)
        return deduped[: spec.visuals.max_images_per_round]

    def _template_vars(self, spec: ProjectSpec, *, session_id: int, round_index: int) -> dict[str, str]:
        workspace = self.executor.round_workspace(spec, session_id, round_index)
        project_root = Path(spec.project_root).expanduser().resolve()
        working_dir = Path(spec.working_dir).expanduser().resolve()
        round_output_dir = self._round_output_dir(spec, workspace.round_dir, session_id, round_index)
        return {
            "project_root": str(project_root),
            "working_dir": str(working_dir),
            "trainee_dir": str(project_trainee_dir(project_root)),
            "session_id": str(session_id),
            "round_index": str(round_index),
            "session_dir": str(workspace.session_dir),
            "round_dir": str(workspace.round_dir),
            "round_output_dir": str(round_output_dir),
            "config_path": str(workspace.config_path),
        }

    def _round_output_dir(self, spec: ProjectSpec, round_dir: Path, session_id: int, round_index: int) -> Path:
        if spec.output is None:
            return round_dir / "outputs"
        workspace = self.executor.round_workspace(spec, session_id, round_index)
        rendered = self.executor.render_output_path(spec, workspace)
        path = Path(rendered).expanduser()
        if not path.is_absolute():
            path = Path(spec.project_root).expanduser().resolve() / path
        return path.resolve()

    def _format_pattern(self, pattern: str, variables: dict[str, str]) -> str:
        try:
            return pattern.format_map(variables)
        except KeyError as exc:
            name = str(exc.args[0])
            allowed = ", ".join(sorted(variables))
            raise ValueError(f"visuals.patterns references unknown template variable {name!r}; allowed: {allowed}") from exc

    def _read_image(self, path: Path) -> dict[str, str]:
        raw = path.read_bytes()
        if len(raw) > self.max_image_bytes:
            raise ValueError(f"image must be {self.max_image_bytes // (1024 * 1024)}MB or smaller")
        if not raw:
            raise ValueError("image is empty")
        media_type = mimetypes.guess_type(str(path))[0] or "image/png"
        if not media_type.startswith("image/"):
            raise ValueError(f"unsupported image media type: {media_type}")
        return {"media_type": media_type, "data": base64.b64encode(raw).decode("ascii")}

    def _analysis_prompt(self, spec: ProjectSpec, path: Path) -> str:
        return (
            "Analyze this image from the latest completed training round.\n\n"
            f"Project visual prompt: {spec.visuals.prompt}\n"
            f"Image path: {self._display_path(spec, path)}\n\n"
            "Interpret it as an ordinary ML training/evaluation plot unless visible labels indicate otherwise.\n"
            "Use visible text first: title, axis labels, legends, filenames, and directory names.\n"
            "Then use curve shape and visual patterns.\n\n"
            "Common meanings:\n"
            "- decreasing loss/error usually suggests optimization progress\n"
            "- flat validation metric may indicate a plateau\n"
            "- training improves while validation worsens may indicate overfitting\n"
            "- spikes or oscillation may indicate instability or overly aggressive settings\n"
            "- NaN-like, blank, or collapsed plots indicate failed or invalid runs\n"
            "- qualitative render/result plots should be judged conservatively\n\n"
            "Do not choose next parameters. If the plot is unclear, say it is unclear and lower confidence.\n"
            "Return JSON only with keys: likely_meaning, visible_signals, concerns, "
            "decision_relevant_observations, confidence."
        )

    def _plot_observation(self, spec: ProjectSpec, path: Path, content: str) -> VisualPlotObservation:
        raw = content.strip()
        display_path = self._display_path(spec, path)
        try:
            payload = self._extract_json(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            return VisualPlotObservation(
                path=display_path,
                raw_response=raw,
                error=f"visual analysis response could not be parsed: {exc}",
            )

        if isinstance(payload.get("plots"), list) and payload["plots"]:
            first = payload["plots"][0]
            if isinstance(first, dict):
                payload = first

        return VisualPlotObservation(
            path=display_path,
            likely_meaning=str(payload.get("likely_meaning") or payload.get("summary") or ""),
            visible_signals=self._string_list(payload.get("visible_signals")),
            concerns=self._string_list(payload.get("concerns")),
            decision_relevant_observations=self._string_list(payload.get("decision_relevant_observations")),
            confidence=self._confidence(payload.get("confidence")),
            raw_response=raw,
        )

    def _overall_summary(self, plots: list[VisualPlotObservation]) -> str:
        summaries = [
            f"{Path(item.path).name}: {item.likely_meaning}"
            for item in plots
            if item.likely_meaning
        ]
        return "; ".join(summaries)

    def _decision_observations(self, plots: list[VisualPlotObservation]) -> list[str]:
        values: list[str] = []
        for item in plots:
            for value in item.decision_relevant_observations:
                if value not in values:
                    values.append(value)
        if values:
            return values
        for item in plots:
            for value in [*item.visible_signals, *item.concerns]:
                if value not in values:
                    values.append(value)
        return values

    def _extract_json(self, content: str) -> dict[str, Any]:
        if content.startswith("{"):
            payload = json.loads(content)
        else:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("no JSON object found in visual analysis response")
            payload = json.loads(content[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("visual analysis response must be a JSON object")
        return payload

    def _string_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []

    def _confidence(self, value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, confidence))

    def _display_path(self, spec: ProjectSpec, path: Path) -> str:
        project_root = Path(spec.project_root).expanduser().resolve()
        try:
            return path.resolve().relative_to(project_root).as_posix()
        except ValueError:
            return str(path)

    def _mtime_ns(self, path: Path) -> int:
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return 0

    def _ensure_within(self, path: Path, root: Path, field_name: str) -> None:
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{field_name} must stay within project_root: {path}") from exc
