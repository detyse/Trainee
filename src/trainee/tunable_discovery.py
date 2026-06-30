from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from trainee.llm import LLMClient
from trainee.models import ParamType, ProjectContext, ProjectSpec, TunableParam
from trainee.project_config import (
    CommandArg,
    ProjectConfig,
    TuningConfig,
    fixed_arg_exclusions,
    tunable_excluded_by_fixed_args,
)
from trainee.prompt_documents import PromptDocument, PromptDocumentLoader
from trainee.providers import active_model, provider_is_configured
from trainee.settings import Settings


CandidateApplicability = Literal["auto_applyable", "needs_review", "unsupported"]
CandidateRisk = Literal["low", "medium", "high", "unknown"]
CandidateTargetKind = Literal["config_path", "cli_flag", "note"]


class TunableParamSuggestion(BaseModel):
    name: str
    flag: Optional[str] = None
    config_path: Optional[str] = None
    type: ParamType = "float"
    default: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[list[str]] = None
    reason: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_tunable_shape(self) -> "TunableParamSuggestion":
        self.to_tunable_param()
        return self

    def to_tunable_param(self) -> TunableParam:
        return TunableParam(
            name=self.name,
            flag=self.flag,
            config_path=self.config_path,
            type=self.type,
            default=self.default,
            min_value=self.min_value,
            max_value=self.max_value,
            choices=self.choices,
        )


class TunableCandidate(BaseModel):
    name: str
    target_kind: CandidateTargetKind = "config_path"
    target: str
    type: ParamType = "float"
    default: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[list[str]] = None
    applicability: CandidateApplicability = "needs_review"
    risk: CandidateRisk = "unknown"
    reason: str = ""
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class TunableDiscoveryRequest(BaseModel):
    project_root: Optional[str] = None
    limit: int = Field(default=32, ge=1, le=32)


class TunableDiscoveryApply(BaseModel):
    project_root: str
    suggestions: list[TunableParamSuggestion]
    replace: bool = False


class TunableDiscoveryResult(BaseModel):
    source: Literal["llm", "heuristic"]
    provider: str = "none"
    model: str = "none"
    baseline_config_path: Optional[str] = None
    candidates: list[TunableCandidate] = Field(default_factory=list)
    suggestions: list[TunableParamSuggestion] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TunableApplySkip(BaseModel):
    name: str
    target: str
    reason: str


class TunableDiscoveryEngine:
    def __init__(self, settings: Settings, llm_client: Optional[LLMClient] = None) -> None:
        self.settings = settings
        self.llm_client = llm_client or LLMClient(settings)
        self.prompt_document_loader = PromptDocumentLoader()

    async def suggest(
        self,
        spec: ProjectSpec,
        context: ProjectContext,
        *,
        limit: int = 32,
        fixed_args: Iterable[CommandArg] = (),
    ) -> TunableDiscoveryResult:
        exclusions = fixed_arg_exclusions(fixed_args)
        prompt_documents = self.prompt_document_loader.load(spec.project_root)
        heuristic = suggest_tunable_params_heuristic(
            spec,
            context,
            limit=limit,
            exclusions=exclusions,
            prompt_documents=prompt_documents,
        )
        if not provider_is_configured(self.settings):
            return heuristic

        try:
            baseline = _load_baseline_mapping(spec)
            leaves = _flatten_scalars(baseline)
            completion = await self.llm_client.complete_active(
                _discovery_system_prompt(),
                _discovery_user_prompt(spec, context, heuristic, leaves, limit, prompt_documents),
            )
            payload = _extract_json_object(completion.content)
            candidates = _candidates_from_payload(payload, leaves, spec, limit)
            suggestions = _suggestions_from_candidates(candidates, leaves, spec, limit, exclusions)
        except Exception as exc:
            return heuristic.model_copy(
                update={
                    "warnings": [
                        *heuristic.warnings,
                        f"LLM tunable discovery failed; using heuristic fallback: {type(exc).__name__}: {exc}",
                    ]
                }
            )

        if not suggestions:
            if candidates:
                return TunableDiscoveryResult(
                    source="llm",
                    provider=self.settings.llm_provider,
                    model=active_model(self.settings),
                    baseline_config_path=spec.baseline_config_path,
                    candidates=candidates,
                    suggestions=[],
                    warnings=[*heuristic.warnings, "LLM returned candidates, but none were auto-applyable."],
                )
            return heuristic.model_copy(
                update={"warnings": [*heuristic.warnings, "LLM did not return any valid tunable candidates."]}
            )

        return TunableDiscoveryResult(
            source="llm",
            provider=self.settings.llm_provider,
            model=active_model(self.settings),
            baseline_config_path=spec.baseline_config_path,
            candidates=candidates,
            suggestions=suggestions,
            warnings=heuristic.warnings,
        )


def suggest_tunable_params_heuristic(
    spec: ProjectSpec,
    context: ProjectContext,
    *,
    limit: int = 32,
    exclusions: Optional[set[str]] = None,
    prompt_documents: Iterable[PromptDocument] = (),
) -> TunableDiscoveryResult:
    warnings: list[str] = []
    exclusions = exclusions or set()
    if not spec.baseline_config_path:
        return TunableDiscoveryResult(
            source="heuristic",
            baseline_config_path=None,
            warnings=["launch.baseline_config is required before tunable params can be discovered from config."],
        )

    try:
        baseline = _load_baseline_mapping(spec)
    except ValueError as exc:
        return TunableDiscoveryResult(
            source="heuristic",
            baseline_config_path=spec.baseline_config_path,
            warnings=[str(exc)],
        )

    leaves = _flatten_scalars(baseline)
    existing_targets = _existing_targets(spec)
    context_text = _context_text(spec, context, prompt_documents)
    scored: list[tuple[float, TunableParamSuggestion]] = []
    for path, value in leaves.items():
        if path in existing_targets:
            continue
        suggestion = _heuristic_suggestion(path, value, context_text)
        if suggestion is None or tunable_excluded_by_fixed_args(
            suggestion.to_tunable_param(),
            exclusions,
        ):
            continue
        score = _score_path(path, value, context_text)
        scored.append((score, suggestion))

    scored.sort(key=lambda item: (-item[0], item[1].config_path or item[1].name))
    return TunableDiscoveryResult(
        source="heuristic",
        provider="none",
        model="none",
        baseline_config_path=spec.baseline_config_path,
        candidates=_candidates_from_suggestions([item[1] for item in scored[:limit]]),
        suggestions=[item[1] for item in scored[:limit]],
        warnings=warnings,
    )


def apply_tunable_suggestions(
    config: ProjectConfig,
    suggestions: Iterable[TunableParamSuggestion],
    *,
    replace: bool = False,
) -> tuple[ProjectConfig, list[TunableParam]]:
    updated, applied, _ = apply_tunable_suggestions_with_report(config, suggestions, replace=replace)
    return updated, applied


def apply_tunable_suggestions_with_report(
    config: ProjectConfig,
    suggestions: Iterable[TunableParamSuggestion],
    *,
    replace: bool = False,
) -> tuple[ProjectConfig, list[TunableParam], list[TunableApplySkip]]:
    params = [] if replace else list(config.tuning.params)
    existing_names = {item.name for item in params}
    existing_targets = {(item.config_path or item.flag or "") for item in params}
    exclusions = fixed_arg_exclusions([*config.launch.args, *config.run.fixed_args])
    applied: list[TunableParam] = []
    skipped: list[TunableApplySkip] = []

    for suggestion in suggestions:
        param = _normalize_applied_param(suggestion.to_tunable_param(), existing_names)
        target = param.config_path or param.flag or ""
        if target in existing_targets:
            skipped.append(TunableApplySkip(name=param.name, target=target, reason="target already exists"))
            continue
        if tunable_excluded_by_fixed_args(param, exclusions):
            skipped.append(TunableApplySkip(name=param.name, target=target, reason="excluded by fixed args"))
            continue
        params.append(param)
        applied.append(param)
        existing_names.add(param.name)
        existing_targets.add(target)

    return config.model_copy(update={"tuning": TuningConfig(params=params)}), applied, skipped


def _normalize_applied_param(param: TunableParam, existing_names: set[str]) -> TunableParam:
    payload = param.model_dump(mode="python")
    if param.config_path:
        payload["default"] = None
    if param.name in existing_names:
        payload["name"] = _unique_name(_param_name(param.config_path) if param.config_path else param.name, existing_names)
    return TunableParam.model_validate(payload)


def _unique_name(base: str, existing_names: set[str]) -> str:
    name = base or "tunable_param"
    if name not in existing_names:
        return name
    index = 2
    while f"{name}_{index}" in existing_names:
        index += 1
    return f"{name}_{index}"


def _load_baseline_mapping(spec: ProjectSpec) -> dict[str, Any]:
    if not spec.baseline_config_path:
        raise ValueError("baseline_config_path is required")
    path = Path(spec.baseline_config_path).expanduser().resolve()
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


def _existing_targets(spec: ProjectSpec) -> set[str]:
    targets = {
        item.config_path
        for item in spec.tunable_params
        if item.config_path
    } | {
        item.flag
        for item in spec.tunable_params
        if item.flag
    }
    if spec.output is not None:
        targets.add(spec.output.config_path)
    return targets


def _heuristic_suggestion(path: str, value: Any, context_text: str) -> Optional[TunableParamSuggestion]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    score = _score_path(path, value, context_text)
    if score <= 0:
        return None
    param_type: ParamType = "int" if isinstance(value, int) and not isinstance(value, bool) else "float"
    min_value, max_value = _bounds_for(value, path)
    return TunableParamSuggestion(
        name=_param_name(path),
        config_path=path,
        type=param_type,
        default=value,
        min_value=min_value,
        max_value=max_value,
        reason=_reason_for(path),
        confidence=min(0.95, max(0.35, 0.25 + score / 12.0)),
    )


def _score_path(path: str, value: Any, context_text: str) -> float:
    lowered = path.lower()
    if any(token in lowered for token in _EXCLUDED_TOKENS):
        return -5.0
    if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) == 0.0:
        if not any(token in lowered for token in ("weight", "lambda", "alpha", "beta")):
            return -2.0

    score = 0.0
    if "term_weights" in lowered:
        score += 6.0
    if any(token in lowered for token in ("weight", "lambda", "coef", "regular", "penalty", "prior")):
        score += 4.0
    if any(token in lowered for token in ("theta", "stretch", "bone", "scale", "mask", "silhouette", "temp")):
        score += 2.0
    if any(token in lowered for token in ("lr", "learning_rate", "max_iters", "tolerance", "sigma")):
        score += 1.5
    for token in _path_tokens(path):
        if len(token) >= 4 and token in context_text:
            score += 0.25
    return score


def _bounds_for(value: int | float, path: str) -> tuple[Optional[float], Optional[float]]:
    lowered = path.lower()
    numeric = float(value)
    if isinstance(value, int) and not isinstance(value, bool):
        if "max_iters" in lowered or "epochs" in lowered:
            return max(1.0, math.floor(numeric / 4.0)), max(1.0, math.ceil(numeric * 4.0))
        return max(0.0, math.floor(numeric / 5.0)), max(1.0, math.ceil(numeric * 5.0))
    if numeric > 0:
        if any(token in lowered for token in ("epsilon", "tolerance")):
            return numeric / 10.0, numeric * 10.0
        return max(0.0, numeric / 5.0), numeric * 5.0
    return 0.0, 1.0


def _param_name(path: str) -> str:
    parts = [part for part in path.split(".") if not part.isdigit()]
    if "stages" in parts:
        stage_index = parts.index("stages")
        if stage_index + 1 < len(parts):
            raw = "_".join([parts[stage_index + 1], parts[-1]])
        else:
            raw = "_".join(parts[-3:])
    elif len(parts) >= 2 and parts[-2] in {"term_weights", "keypoint_weights", "bone_weights"}:
        raw = "_".join(parts[-2:])
    elif len(parts) >= 3 and parts[-3] == "stages":
        raw = "_".join(parts[-2:])
    else:
        raw = "_".join(parts[-3:])
    name = re.sub(r"[^0-9a-zA-Z]+", "_", raw).strip("_").lower()
    return name or "tunable_param"


def _reason_for(path: str) -> str:
    lowered = path.lower()
    if "term_weights" in lowered:
        return "Loss term weight in the baseline config; changing it can rebalance fitting objectives."
    if "max_iters" in lowered:
        return "Iteration budget field; changing it can trade runtime for convergence."
    if any(token in lowered for token in ("lr", "learning_rate")):
        return "Learning-rate style numeric field."
    return "Numeric config field with tuning-related naming/context."


def _context_text(
    spec: ProjectSpec,
    context: ProjectContext,
    prompt_documents: Iterable[PromptDocument] = (),
) -> str:
    return " ".join(
        [
            spec.tuning_prompt,
            context.project_summary,
            context.training_entrypoint_summary,
            context.parameter_summary,
            context.result_reading_summary,
            *[item.text for item in prompt_documents],
        ]
    ).lower()


def _path_tokens(path: str) -> list[str]:
    return [token for token in re.split(r"[^a-zA-Z0-9]+", path.lower()) if token]


def _discovery_system_prompt() -> str:
    return (
        "You are a training-configuration reviewer. "
        "Propose a broad but auditable set of candidate tunable parameters. "
        "Mark candidates that can safely become tuning.yaml params as auto_applyable; "
        "mark uncertain, non-numeric, CLI, or structural changes as needs_review or unsupported. "
        "Return JSON only."
    )


def _discovery_user_prompt(
    spec: ProjectSpec,
    context: ProjectContext,
    heuristic: TunableDiscoveryResult,
    leaves: dict[str, Any],
    limit: int,
    prompt_documents: Iterable[PromptDocument],
) -> str:
    scalar_leaves = {
        path: value
        for path, value in leaves.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }
    numeric_leaves = {
        path: value
        for path, value in leaves.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    payload = {
        "task": "Choose candidate tuning.yaml params for a two-stage approval workflow.",
        "limit": limit,
        "project_context": context.model_dump(mode="json"),
        "tuning_prompt": spec.tuning_prompt,
        "metric_specs": [item.model_dump(mode="json") for item in spec.metric_specs],
        "existing_tunable_params": [item.model_dump(mode="json") for item in spec.tunable_params],
        "prompt_documents": [item.model_dump(mode="json") for item in prompt_documents],
        "scalar_config_leaves": scalar_leaves,
        "numeric_config_leaves": numeric_leaves,
        "heuristic_suggestions": [item.model_dump(mode="json") for item in heuristic.suggestions],
        "output_schema": {
            "candidates": [
                {
                    "name": "string",
                    "target_kind": "config_path|cli_flag|note",
                    "target": "config path, CLI flag, or short note",
                    "type": "int|float|str|bool",
                    "default": "current scalar value",
                    "min_value": "reasonable lower bound",
                    "max_value": "reasonable upper bound",
                    "choices": ["optional string choices"],
                    "applicability": "auto_applyable|needs_review|unsupported",
                    "risk": "low|medium|high|unknown",
                    "reason": "why this parameter is safe/useful or why review is needed",
                    "evidence": ["short supporting signals from config/context"],
                    "confidence": "0..1",
                }
            ]
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _candidates_from_suggestions(suggestions: Iterable[TunableParamSuggestion]) -> list[TunableCandidate]:
    candidates: list[TunableCandidate] = []
    for suggestion in suggestions:
        target = suggestion.config_path or suggestion.flag or suggestion.name
        target_kind: CandidateTargetKind = "config_path" if suggestion.config_path else "cli_flag"
        candidates.append(
            TunableCandidate(
                name=suggestion.name,
                target_kind=target_kind,
                target=target,
                type=suggestion.type,
                default=suggestion.default,
                min_value=suggestion.min_value,
                max_value=suggestion.max_value,
                choices=suggestion.choices,
                applicability="auto_applyable",
                risk="low",
                reason=suggestion.reason,
                evidence=["heuristic_suggestion"],
                confidence=suggestion.confidence,
            )
        )
    return candidates


def _candidates_from_payload(
    payload: dict[str, Any],
    leaves: dict[str, Any],
    spec: ProjectSpec,
    limit: int,
) -> list[TunableCandidate]:
    raw = payload.get("candidates", payload.get("suggestions", []))
    if not isinstance(raw, list):
        return []
    existing_targets = _existing_targets(spec)
    candidates: list[TunableCandidate] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_candidate_payload(item, leaves)
        target = normalized.get("target")
        if not isinstance(target, str) or not target or target in seen:
            continue
        if target in existing_targets:
            continue
        try:
            candidate = TunableCandidate.model_validate(normalized)
        except ValueError:
            continue
        candidates.append(candidate)
        seen.add(target)
        if len(candidates) >= limit:
            break
    return candidates


def _normalize_candidate_payload(item: dict[str, Any], leaves: dict[str, Any]) -> dict[str, Any]:
    payload = dict(item)
    if "target" not in payload:
        if isinstance(payload.get("config_path"), str):
            payload["target"] = payload["config_path"]
            payload.setdefault("target_kind", "config_path")
        elif isinstance(payload.get("flag"), str):
            payload["target"] = payload["flag"]
            payload.setdefault("target_kind", "cli_flag")
    payload.setdefault("target_kind", "config_path")
    target = payload.get("target")
    if isinstance(target, str) and payload.get("target_kind") == "config_path" and target in leaves:
        value = leaves[target]
        payload.setdefault("default", value)
        if isinstance(value, bool):
            payload.setdefault("type", "bool")
        elif isinstance(value, int):
            payload.setdefault("type", "int")
        elif isinstance(value, float):
            payload.setdefault("type", "float")
        elif isinstance(value, str):
            payload.setdefault("type", "str")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            min_value, max_value = _bounds_for(value, target)
            payload.setdefault("min_value", min_value)
            payload.setdefault("max_value", max_value)
    evidence = payload.get("evidence")
    if isinstance(evidence, str):
        payload["evidence"] = [evidence]
    if isinstance(target, str):
        payload.setdefault(
            "name",
            _param_name(target)
            if payload.get("target_kind") == "config_path"
            else re.sub(r"[^0-9a-zA-Z]+", "_", target.lstrip("-")).strip("_").lower(),
        )
    payload.setdefault("applicability", "needs_review")
    payload.setdefault("risk", "unknown")
    payload.setdefault("reason", "")
    payload.setdefault("evidence", [])
    return payload


def _suggestions_from_candidates(
    candidates: Iterable[TunableCandidate],
    leaves: dict[str, Any],
    spec: ProjectSpec,
    limit: int,
    exclusions: set[str],
) -> list[TunableParamSuggestion]:
    existing_targets = _existing_targets(spec)
    suggestions: list[TunableParamSuggestion] = []
    seen: set[str] = set()
    for candidate in candidates:
        path = candidate.target
        if candidate.applicability != "auto_applyable" or candidate.target_kind != "config_path":
            continue
        if path not in leaves or path in existing_targets or path in seen:
            continue
        value = leaves[path]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        payload = {
            "name": candidate.name or _param_name(path),
            "config_path": path,
            "type": "int" if isinstance(value, int) else "float",
            "default": value,
            "min_value": candidate.min_value,
            "max_value": candidate.max_value,
            "reason": candidate.reason,
            "confidence": candidate.confidence,
        }
        min_value, max_value = _bounds_for(value, path)
        payload["min_value"] = min_value if payload["min_value"] is None else payload["min_value"]
        payload["max_value"] = max_value if payload["max_value"] is None else payload["max_value"]
        try:
            suggestion = TunableParamSuggestion.model_validate(payload)
        except ValueError:
            continue
        if tunable_excluded_by_fixed_args(suggestion.to_tunable_param(), exclusions):
            continue
        suggestions.append(suggestion)
        seen.add(path)
        if len(suggestions) >= limit:
            break
    return suggestions


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


_EXCLUDED_TOKENS = {
    "path",
    "root",
    "dir",
    "output",
    "render",
    "camera",
    "views",
    "image",
    "color",
    "opacity",
    "ambient",
    "diffuse",
    "specular",
    "shininess",
    "debug",
    "batch_id",
    "device",
    "fps",
    "frame_ids",
    "initialization",
    "translation_scale",
    "scale_scale",
}
