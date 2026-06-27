from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

from trainee.decision import DecisionEngine
from trainee.models import OutputConfig, ProjectContext, ProjectSpec
from trainee.prompt_documents import PromptDocument, PromptDocumentLoader
from trainee.providers import active_model, provider_is_configured
from trainee.settings import Settings


class OutputCandidate(BaseModel):
    config_path: str
    current_value: Any = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    evidence: list[str] = Field(default_factory=list)


class OutputDiscoveryResult(BaseModel):
    source: str = "llm"
    provider: str = "none"
    model: str = "none"
    output: Optional[OutputConfig] = None
    candidates: list[OutputCandidate] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class OutputDiscoveryEngine:
    def __init__(self, settings: Settings, decision_engine: Optional[DecisionEngine] = None) -> None:
        self.settings = settings
        self.decision_engine = decision_engine or DecisionEngine(settings)
        self.prompt_document_loader = PromptDocumentLoader()

    async def suggest(self, spec: ProjectSpec, context: ProjectContext) -> OutputDiscoveryResult:
        if spec.output is not None:
            return OutputDiscoveryResult(
                output=spec.output,
                warnings=["output is already configured; keeping existing project.yaml value."],
            )
        if not spec.baseline_config_path:
            return OutputDiscoveryResult(warnings=["launch.baseline_config is not set."])
        if not provider_is_configured(self.settings):
            return OutputDiscoveryResult(warnings=["LLM provider is not configured; output discovery skipped."])

        try:
            leaves = _flatten_scalars(_load_yaml_mapping(Path(spec.baseline_config_path)))
            prompt_documents = self.prompt_document_loader.load(spec.project_root)
            completion = await self.decision_engine._provider_complete(  # noqa: SLF001 - provider adapter reuse
                _system_prompt(),
                _user_prompt(spec, context, leaves, prompt_documents),
            )
            return _result_from_payload(
                _extract_json_object(completion.content),
                leaves,
                provider=self.settings.llm_provider,
                model=active_model(self.settings),
            )
        except Exception as exc:
            return OutputDiscoveryResult(warnings=[f"LLM output discovery failed: {type(exc).__name__}: {exc}"])


def _result_from_payload(payload: dict[str, Any], leaves: dict[str, Any], *, provider: str, model: str) -> OutputDiscoveryResult:
    candidates = [_candidate_from_payload(item, leaves) for item in payload.get("candidates", []) if isinstance(item, dict)]
    candidates = [item for item in candidates if item is not None]
    selected = payload.get("selected")
    selected_candidate = _candidate_from_payload(selected, leaves) if isinstance(selected, dict) else None
    warnings = [str(item) for item in payload.get("warnings", []) if isinstance(item, str)]
    if isinstance(selected, dict) and selected_candidate is None:
        warnings.append("selected output config_path is not an existing scalar key.")

    if selected_candidate is None:
        return OutputDiscoveryResult(provider=provider, model=model, candidates=candidates, warnings=warnings)
    if selected_candidate.config_path not in leaves:
        return OutputDiscoveryResult(
            provider=provider,
            model=model,
            candidates=candidates,
            warnings=[*warnings, f"selected output config_path does not exist: {selected_candidate.config_path}"],
        )
    current_value = leaves[selected_candidate.config_path]
    if not isinstance(current_value, str):
        return OutputDiscoveryResult(
            provider=provider,
            model=model,
            candidates=candidates,
            warnings=[*warnings, f"selected output config_path is not a string path: {selected_candidate.config_path}"],
        )
    if selected_candidate.confidence < 0.7:
        return OutputDiscoveryResult(
            provider=provider,
            model=model,
            candidates=candidates,
            warnings=[*warnings, "selected output candidate confidence is below 0.7."],
        )
    if _looks_like_input_path(selected_candidate.config_path, current_value):
        return OutputDiscoveryResult(
            provider=provider,
            model=model,
            candidates=candidates,
            warnings=[*warnings, f"selected output config_path looks like an input/data path: {selected_candidate.config_path}"],
        )
    if selected_candidate not in candidates:
        candidates.insert(0, selected_candidate)
    return OutputDiscoveryResult(
        provider=provider,
        model=model,
        output=OutputConfig(config_path=selected_candidate.config_path),
        candidates=candidates,
        warnings=warnings,
    )


def _candidate_from_payload(payload: dict[str, Any], leaves: dict[str, Any]) -> OutputCandidate | None:
    path = payload.get("config_path")
    if not isinstance(path, str) or path not in leaves:
        return None
    return OutputCandidate(
        config_path=path,
        current_value=leaves[path],
        confidence=float(payload.get("confidence", 0.0)),
        reason=str(payload.get("reason", "")),
        evidence=[str(item) for item in payload.get("evidence", []) if isinstance(item, str)],
    )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"baseline config is invalid YAML: {path}: {exc}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"baseline config must contain a YAML mapping: {path}")
    return payload


def _flatten_scalars(payload: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(payload, dict):
        flattened: dict[str, Any] = {}
        for key, value in payload.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_scalars(value, child_prefix))
        return flattened
    if isinstance(payload, list):
        flattened = {}
        for index, value in enumerate(payload):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            flattened.update(_flatten_scalars(value, child_prefix))
        return flattened
    return {prefix: payload}


def _looks_like_input_path(config_path: str, value: str) -> bool:
    tokens = set(_tokens(config_path)) | set(_tokens(value))
    output_tokens = {"output", "outputs", "result", "results", "run", "runs", "log", "logs", "checkpoint", "checkpoints", "artifact", "artifacts", "render"}
    input_tokens = {"data", "dataset", "datasets", "image", "images", "annotation", "annotations", "label", "labels", "mask", "masks", "camera", "cameras"}
    return bool(tokens & input_tokens) and not bool(tokens & output_tokens)


def _tokens(value: str) -> list[str]:
    return [token for token in re.split(r"[^0-9a-zA-Z_]+", value.lower()) if token]


def _extract_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("{"):
        payload = json.loads(text)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no JSON object found in completion")
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("completion JSON must be an object")
    return payload


def _system_prompt() -> str:
    return (
        "You identify the training program's output directory field in a baseline YAML config. "
        "Select only an existing scalar config key. Do not invent keys or runtime paths. Return JSON only."
    )


def _user_prompt(
    spec: ProjectSpec,
    context: ProjectContext,
    leaves: dict[str, Any],
    prompt_documents: list[PromptDocument],
) -> str:
    payload = {
        "goal": "Find the config key that controls where the run writes logs, checkpoints, renders, params, or result artifacts.",
        "rules": [
            "Select only an existing key from scalar_config_leaves.",
            "Prefer output.*, output_dir, save_dir, result_dir, log_dir, or checkpoint_dir.",
            "Reject data/input/image/annotation/mask/camera/dataset paths.",
            "If no single confident output directory key exists, return selected=null.",
            "Trainee will assign the runtime output value; only choose the config key.",
        ],
        "project_context": context.model_dump(mode="json"),
        "baseline_config_path": spec.baseline_config_path,
        "prompt_documents": [item.model_dump(mode="json") for item in prompt_documents],
        "scalar_config_leaves": {
            path: value
            for path, value in leaves.items()
            if value is None or isinstance(value, (str, int, float, bool))
        },
        "output_schema": {
            "selected": {
                "config_path": "existing key from scalar_config_leaves",
                "current_value": "current value",
                "confidence": "0..1",
                "reason": "short reason",
                "evidence": ["short evidence"],
            },
            "candidates": [
                {
                    "config_path": "existing key",
                    "current_value": "current value",
                    "confidence": "0..1",
                    "reason": "short reason",
                    "evidence": ["short evidence"],
                }
            ],
            "warnings": ["optional warning"],
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
