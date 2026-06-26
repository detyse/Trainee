from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


ParamType = Literal["int", "float", "str", "bool"]
SignalSourceType = Literal["stdout", "stderr", "log_file_mtime", "heartbeat_json"]
MetricSource = Literal["log_regex", "stdout_regex", "log_file_regex", "jsonl", "wandb_summary"]
MetricGoal = Literal["min", "max"]
DecisionAction = Literal["continue", "stop"]
SecurityMode = Literal["guarded", "unsafe"]


class TunableParam(BaseModel):
    name: str
    flag: Optional[str] = None
    config_path: Optional[str] = None
    type: ParamType = "float"
    default: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[str]] = None

    @field_validator("flag")
    @classmethod
    def validate_flag(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if not value.startswith("-"):
            raise ValueError("flag must start with '-' or '--'")
        return value

    @field_validator("config_path")
    @classmethod
    def validate_config_path(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if any(not part for part in normalized.split(".")):
            raise ValueError("config_path must use non-empty dot-separated segments")
        return normalized

    @model_validator(mode="after")
    def validate_bounds(self) -> "TunableParam":
        if not self.flag and not self.config_path:
            raise ValueError("tunable param requires flag or config_path")
        if self.min_value is not None and self.max_value is not None and self.min_value > self.max_value:
            raise ValueError("min_value cannot be greater than max_value")
        if self.choices and self.type != "str":
            raise ValueError("choices are only supported for string parameters")
        if self.default is not None:
            self.normalize_value(self.default)
        return self

    def normalize_value(self, value: Any) -> Any:
        if self.type == "int":
            normalized = int(value)
        elif self.type == "float":
            normalized = float(value)
        elif self.type == "bool":
            if isinstance(value, bool):
                normalized = value
            elif isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes", "on"}:
                    normalized = True
                elif lowered in {"false", "0", "no", "off"}:
                    normalized = False
                else:
                    raise ValueError(f"invalid boolean value for {self.name}: {value}")
            else:
                normalized = bool(value)
        else:
            normalized = str(value)

        if self.choices and str(normalized) not in self.choices:
            raise ValueError(f"value for {self.name} must be one of {self.choices}")
        if self.type in {"int", "float"}:
            numeric = float(normalized)
            if self.min_value is not None and numeric < self.min_value:
                raise ValueError(f"value for {self.name} cannot be lower than {self.min_value}")
            if self.max_value is not None and numeric > self.max_value:
                raise ValueError(f"value for {self.name} cannot be greater than {self.max_value}")
        return normalized

    def target_key(self) -> str:
        if self.config_path:
            return f"config:{self.config_path}"
        if self.flag:
            return f"cli:{self.flag}"
        return f"name:{self.name}"


class MetricSpec(BaseModel):
    name: str
    source: MetricSource = "log_regex"
    key_or_pattern: str
    path: Optional[str] = None
    paths: List[str] = Field(default_factory=list)
    goal: MetricGoal = "min"
    required: bool = True

    @model_validator(mode="after")
    def validate_source_paths(self) -> "MetricSpec":
        if self.source in {"log_file_regex", "jsonl"} and not self.path and not self.paths:
            raise ValueError(f"{self.source} requires path or paths")
        if self.source in {"log_regex", "stdout_regex", "log_file_regex"}:
            try:
                pattern = re.compile(self.key_or_pattern)
            except re.error as exc:
                raise ValueError(f"{self.source} key_or_pattern is invalid regex: {exc}") from exc
            if "value" not in pattern.groupindex and pattern.groups < 1:
                raise ValueError(
                    f"{self.source} key_or_pattern must contain a capture group "
                    "or a named (?P<value>...) group"
                )
        return self


class SignalSource(BaseModel):
    type: SignalSourceType = "log_file_mtime"
    path: Optional[str] = None
    paths: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_source_paths(self) -> "SignalSource":
        if self.type in {"log_file_mtime", "heartbeat_json"} and not self.path and not self.paths:
            raise ValueError(f"{self.type} requires path or paths")
        return self

    def configured_paths(self) -> List[str]:
        values = list(self.paths)
        if self.path:
            values.append(self.path)
        return values


class ProjectSpec(BaseModel):
    project_root: str
    working_dir: str
    launcher_template: str
    security_mode: SecurityMode = "guarded"
    data_paths: List[str] = Field(default_factory=list)
    log_paths: List[str] = Field(default_factory=list)
    signal_sources: List[SignalSource] = Field(default_factory=list)
    wandb_enabled: bool = False
    heartbeat_interval_sec: float = 5.0
    stall_timeout_sec: float = 120.0
    kill_on_stall: bool = True
    round_timeout_sec: Optional[float] = None
    max_rounds: int = 3
    tunable_params: List[TunableParam] = Field(default_factory=list)
    baseline_config_path: Optional[str] = None
    metric_specs: List[MetricSpec] = Field(default_factory=list)
    metric_prompt: str = ""
    tuning_prompt: str = ""

    @model_validator(mode="after")
    def validate_project(self) -> "ProjectSpec":
        if self.heartbeat_interval_sec <= 0:
            raise ValueError("heartbeat_interval_sec must be positive")
        if self.stall_timeout_sec <= 0:
            raise ValueError("stall_timeout_sec must be positive")
        if self.round_timeout_sec is not None and self.round_timeout_sec <= 0:
            raise ValueError("round_timeout_sec must be positive when set")
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be positive")
        names = [item.name for item in self.tunable_params]
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        if duplicate_names:
            raise ValueError("tunable_params names must be unique: " + ", ".join(duplicate_names))
        targets = [item.target_key() for item in self.tunable_params]
        duplicate_targets = sorted({target for target in targets if targets.count(target) > 1})
        if duplicate_targets:
            raise ValueError("tunable_params targets must be unique: " + ", ".join(duplicate_targets))
        return self

    def param_index(self) -> Dict[str, TunableParam]:
        return {item.name: item for item in self.tunable_params}

    def default_params(self) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {}
        for item in self.tunable_params:
            if item.default is not None:
                defaults[item.name] = item.normalize_value(item.default)
        return defaults

    def merge_param_values(self, overrides: Optional[Dict[str, Any]] = None, base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        values: Dict[str, Any] = dict(self.default_params())
        if base:
            for key, value in base.items():
                if key in self.param_index():
                    values[key] = self.param_index()[key].normalize_value(value)
        if overrides:
            unknown = sorted(set(overrides) - set(self.param_index()))
            if unknown:
                raise ValueError(f"unknown tunable params: {', '.join(unknown)}")
            for key, value in overrides.items():
                values[key] = self.param_index()[key].normalize_value(value)
        return values

    def metric_index(self) -> Dict[str, MetricSpec]:
        return {item.name: item for item in self.metric_specs}

    def signal_log_paths(self) -> List[str]:
        paths: List[str] = []
        for source in self.signal_sources:
            if source.type in {"log_file_mtime", "heartbeat_json"}:
                paths.extend(source.configured_paths())
        if not self.signal_sources:
            paths.extend(self.log_paths)
        return _dedupe(paths)

    def process_output_is_signal(self) -> bool:
        if not self.signal_sources:
            return True
        return any(item.type in {"stdout", "stderr"} for item in self.signal_sources)

    def metric_log_paths(self) -> List[str]:
        paths: List[str] = []
        for metric in self.metric_specs:
            if metric.source in {"log_file_regex", "jsonl"}:
                paths.extend(metric.paths)
                if metric.path:
                    paths.append(metric.path)
        return _dedupe(paths)

    def fallback_log_paths_for_metrics(self) -> List[str]:
        return list(self.log_paths)

    def uses_generated_config(self) -> bool:
        return bool(self.baseline_config_path)


def _dedupe(values: List[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


class ProjectContext(BaseModel):
    project_summary: str = ""
    training_entrypoint_summary: str = ""
    data_summary: str = ""
    parameter_summary: str = ""
    result_reading_summary: str = ""
    warnings: List[str] = Field(default_factory=list)


class PromptPreview(BaseModel):
    provider: str = "none"
    model: str = ""
    status: str = "preview"
    system_prompt: str = ""
    user_prompt: str = ""
    payload: Dict[str, Any] = Field(default_factory=dict)
    static_context_json: Dict[str, Any] = Field(default_factory=dict)
    dynamic_state_json: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)


class PromptPreset(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    project_root: str = ""
    name: str
    metric_prompt: str = ""
    tuning_prompt: str = ""
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("preset name cannot be empty")
        return normalized


class AgentDecision(BaseModel):
    action: DecisionAction
    next_params: Dict[str, Any] = Field(default_factory=dict)
    reason: str
    focus_metrics: List[str] = Field(default_factory=list)
    hypothesis: str = ""
    change_summary: str = ""
    latest_round_judgement: str = "inconclusive"
    compare_to_baseline: str = ""
    compare_to_best: str = ""
    expected_effect: str = ""
    avoid_repeating: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class AgentTrace(BaseModel):
    provider: str
    model: Optional[str] = None
    status: str
    http_status: Optional[int] = None
    request_id: Optional[str] = None
    raw_response_body: Optional[str] = None
    error_body: Optional[str] = None
    raw_output: Optional[str] = None
    extracted_json: Optional[Dict[str, Any]] = None
    parse_error: Optional[str] = None
    validation_error: Optional[str] = None
    provider_error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    fallback_reason: Optional[str] = None
    created_at: str = Field(default_factory=utc_now)


class RunSession(BaseModel):
    id: Optional[int] = None
    status: str = "running"
    started_at: str = Field(default_factory=utc_now)
    ended_at: Optional[str] = None
    stop_reason: Optional[str] = None
    requested_stop: bool = False
    current_round: int = 0
    resumed_from: Optional[int] = None
    project_spec: Optional[ProjectSpec] = None
    project_context: Optional[ProjectContext] = None
    image_analysis_count: int = 0


class RoundRecord(BaseModel):
    id: Optional[int] = None
    session_id: Optional[int] = None
    round_index: int
    resolved_command: str
    param_values: Dict[str, Any] = Field(default_factory=dict)
    status: str
    start_time: str = Field(default_factory=utc_now)
    end_time: Optional[str] = None
    exit_code: Optional[int] = None
    log_paths: List[str] = Field(default_factory=list)
    wandb_run_url: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    agent_decision: Optional[AgentDecision] = None
    agent_trace: Optional[AgentTrace] = None
    prompt_preview: Optional[PromptPreview] = None
    error: Optional[str] = None


class LoopSnapshot(BaseModel):
    status: str = "idle"
    current_session_id: Optional[int] = None
    active_round_id: Optional[int] = None
    current_round_index: int = 0
    requested_stop: bool = False
    last_heartbeat_at: Optional[str] = None
    last_signal_at: Optional[str] = None
    message: str = ""


class EventMessage(BaseModel):
    event_type: str
    created_at: str = Field(default_factory=utc_now)
    payload: Dict[str, Any] = Field(default_factory=dict)


class ProjectBundle(BaseModel):
    spec: Optional[ProjectSpec] = None
    context: Optional[ProjectContext] = None
    loop: LoopSnapshot = Field(default_factory=LoopSnapshot)
