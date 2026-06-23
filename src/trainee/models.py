from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


ParamType = Literal["int", "float", "str", "bool"]
MetricSource = Literal["log_regex", "wandb_summary"]
MetricGoal = Literal["min", "max"]
DecisionAction = Literal["continue", "stop"]


class TunableParam(BaseModel):
    name: str
    flag: str
    type: ParamType = "float"
    default: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[str]] = None

    @field_validator("flag")
    @classmethod
    def validate_flag(cls, value: str) -> str:
        if not value.startswith("-"):
            raise ValueError("flag must start with '-' or '--'")
        return value

    @model_validator(mode="after")
    def validate_bounds(self) -> "TunableParam":
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


class MetricSpec(BaseModel):
    name: str
    source: MetricSource = "log_regex"
    key_or_pattern: str
    goal: MetricGoal = "min"
    required: bool = True


class ProjectSpec(BaseModel):
    project_root: str
    working_dir: str
    launcher_template: str
    data_paths: List[str] = Field(default_factory=list)
    log_paths: List[str] = Field(default_factory=list)
    wandb_enabled: bool = False
    heartbeat_interval_sec: float = 5.0
    stall_timeout_sec: float = 120.0
    kill_on_stall: bool = True
    round_timeout_sec: Optional[float] = None
    max_rounds: int = 3
    tunable_params: List[TunableParam] = Field(default_factory=list)
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
