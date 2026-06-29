from __future__ import annotations

import os
import re
import shlex
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from trainee.config_paths import get_config_value
from trainee.models import MetricSpec, OutputConfig, ProjectSpec, SecurityMode, SignalSource, TunableParam, VisualsConfig


LaunchEnvironment = Literal["system", "uv", "venv", "conda"]


class CommandArg(BaseModel):
    flag: str
    value: Any = None

    @field_validator("flag")
    @classmethod
    def validate_flag(cls, value: str) -> str:
        if not value.startswith("-"):
            raise ValueError("flag must start with '-' or '--'")
        return value


class DataInput(BaseModel):
    path: str
    flag: Optional[str] = None

    @field_validator("flag")
    @classmethod
    def validate_flag(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.startswith("-"):
            raise ValueError("data flag must start with '-' or '--'")
        return value


class LaunchConfig(BaseModel):
    environment: LaunchEnvironment = "system"
    env_name: Optional[str] = None
    command: list[str] = Field(default_factory=list)
    baseline_config: Optional[str] = None
    args: list[CommandArg] = Field(default_factory=list)

    @field_validator("baseline_config")
    @classmethod
    def normalize_baseline_config(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def validate_environment(self) -> "LaunchConfig":
        if self.environment == "conda" and not (self.env_name or "").strip():
            raise ValueError("launch.env_name is required for conda")
        if not self.command:
            raise ValueError("launch.command cannot be empty")
        return self


class RunConfig(BaseModel):
    max_rounds: int = 3
    timeout_minutes: Optional[float] = 60.0
    fixed_args: list[CommandArg] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_run(self) -> "RunConfig":
        if self.max_rounds <= 0:
            raise ValueError("run.max_rounds must be positive")
        if self.timeout_minutes is not None and self.timeout_minutes <= 0:
            raise ValueError("run.timeout_minutes must be positive when set")
        return self


class TuningConfig(BaseModel):
    version: Literal[1] = 1
    params: list[TunableParam] = Field(default_factory=list)


class MetricsConfig(BaseModel):
    specs: list[MetricSpec] = Field(default_factory=list)
    prompt: str = ""


class AdvancedConfig(BaseModel):
    security_mode: SecurityMode = "guarded"
    working_dir: Optional[str] = None
    heartbeat_interval_sec: float = 5.0
    signal_sources: list[SignalSource] = Field(
        default_factory=lambda: [
            SignalSource(type="stdout"),
            SignalSource(type="log_file_mtime", paths=[".trainee/logs/**/*.log", ".trainee/runs/**/*.log"]),
        ]
    )
    log_paths: list[str] = Field(default_factory=lambda: [".trainee/logs/**/*.log", ".trainee/runs/**/*.log"])
    wandb_enabled: bool = False
    tuning_prompt: str = ""
    shell_command: Optional[str] = None


class ProjectConfig(BaseModel):
    version: Literal[1] = 1
    data: list[DataInput] = Field(default_factory=list)
    launch: LaunchConfig
    run: RunConfig = Field(default_factory=RunConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    output: Optional[OutputConfig] = None
    visuals: VisualsConfig = Field(default_factory=VisualsConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)

    @model_validator(mode="after")
    def validate_fixed_args_exclude_tunables(self) -> "ProjectConfig":
        exclusions = fixed_arg_exclusions([*self.launch.args, *self.run.fixed_args])
        conflicts = [
            item.name
            for item in self.tuning.params
            if tunable_excluded_by_fixed_args(item, exclusions)
        ]
        if conflicts:
            raise ValueError("tuning.yaml params must not include fixed launch/run args: " + ", ".join(conflicts))
        if self.output is not None and self.output.config_path:
            output_conflicts = [
                item.name
                for item in self.tuning.params
                if item.config_path == self.output.config_path
            ]
            if output_conflicts:
                raise ValueError(
                    "output.config_path must not also be listed in tuning.yaml params: "
                    + ", ".join(output_conflicts)
                )
        return self


class ProjectRegistration(ProjectConfig):
    project_root: str


class _ProjectConfigYamlDumper(yaml.SafeDumper):
    pass


def _represent_empty_null(dumper: yaml.Dumper, data: None) -> yaml.nodes.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:null", "")


_ProjectConfigYamlDumper.add_representer(type(None), _represent_empty_null)


def fixed_arg_exclusions(fixed_args: Iterable[CommandArg]) -> set[str]:
    exclusions: set[str] = set()
    for item in fixed_args:
        exclusions.add(item.flag)
        exclusions.update(_param_aliases(item.flag))
    return exclusions


def tunable_excluded_by_fixed_args(param: TunableParam, exclusions: set[str]) -> bool:
    candidates = {param.name, *_param_aliases(param.name)}
    if param.flag:
        candidates.add(param.flag)
        candidates.update(_param_aliases(param.flag))
    if param.config_path:
        candidates.update(_param_aliases(param.config_path))
        for segment in param.config_path.split("."):
            candidates.update(_param_aliases(segment))
    return bool(candidates & exclusions)


@dataclass(frozen=True)
class ProjectDiscovery:
    environment: LaunchEnvironment
    env_name: Optional[str]
    entrypoints: list[str] = field(default_factory=list)
    data_dirs: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    limit_flags: list[CommandArg] = field(default_factory=list)


def project_config_path(project_root: str | Path) -> Path:
    return Path(project_root).expanduser().resolve() / ".trainee" / "project.yaml"


def tuning_config_path(project_root: str | Path) -> Path:
    return Path(project_root).expanduser().resolve() / ".trainee" / "tuning.yaml"


def load_project_config(project_root: str | Path) -> ProjectConfig:
    path = project_config_path(project_root)
    if not path.is_file():
        raise ValueError(f"project config not found: {path}; run `trainee init` first")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path} is invalid YAML: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    if "tuning" in payload:
        raise ValueError(f"{path} must not contain tuning; use {tuning_config_path(project_root)}")
    config = ProjectConfig.model_validate(payload)
    tuning_path = tuning_config_path(project_root)
    if tuning_path.is_file():
        config = config.model_copy(update={"tuning": load_tuning_config(project_root)})
    return config


def load_tuning_config(project_root: str | Path) -> TuningConfig:
    path = tuning_config_path(project_root)
    if not path.is_file():
        return TuningConfig()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path} is invalid YAML: {exc}") from exc
    if payload is None:
        return TuningConfig()
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return TuningConfig.model_validate(payload)


def save_project_config(project_root: str | Path, config: ProjectConfig) -> Path:
    root = Path(project_root).expanduser().resolve()
    path = project_config_path(root)
    normalized = normalized_project_config(root, config)
    _atomic_write(
        path,
        render_project_config_yaml(normalized).encode("utf-8"),
    )
    save_tuning_config(root, normalized.tuning)
    return path


def save_tuning_config(project_root: str | Path, tuning: TuningConfig) -> Path:
    root = Path(project_root).expanduser().resolve()
    path = tuning_config_path(root)
    _atomic_write(
        path,
        render_tuning_config_yaml(tuning).encode("utf-8"),
    )
    return path


def restore_project_config(project_root: str | Path, previous: Optional[bytes]) -> None:
    path = project_config_path(project_root)
    if previous is None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    _atomic_write(path, previous)


def restore_tuning_config(project_root: str | Path, previous: Optional[bytes]) -> None:
    path = tuning_config_path(project_root)
    if previous is None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    _atomic_write(path, previous)


def render_project_config_yaml(config: ProjectConfig) -> str:
    payload = config.model_dump(mode="json", exclude_none=True)
    payload.pop("tuning", None)
    payload.pop("output", None)
    launch = config.launch.model_dump(mode="json", exclude_none=True)
    launch_payload: dict[str, Any] = {
        "environment": launch["environment"],
        "env_name": config.launch.env_name,
    }
    launch_payload["command"] = launch["command"]
    launch_payload["baseline_config"] = config.launch.baseline_config
    launch_payload["args"] = launch["args"]
    payload["launch"] = launch_payload
    output_payload = {"config_path": config.output.config_path if config.output else None}
    ordered_payload: dict[str, Any] = {}
    for key in ("version", "data", "launch", "run"):
        if key in payload:
            ordered_payload[key] = payload[key]
    ordered_payload["output"] = output_payload
    for key in ("visuals", "metrics", "advanced"):
        if key in payload:
            ordered_payload[key] = payload[key]
    for key, value in payload.items():
        if key not in ordered_payload:
            ordered_payload[key] = value
    return yaml.dump(
        ordered_payload,
        Dumper=_ProjectConfigYamlDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


def render_tuning_config_yaml(tuning: TuningConfig) -> str:
    payload = tuning.model_dump(mode="json", exclude_none=True)
    for item in payload.get("params", []):
        if item.get("config_path"):
            item.pop("default", None)
    return yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


def normalized_project_config(project_root: str | Path, config: ProjectConfig) -> ProjectConfig:
    root = Path(project_root).expanduser().resolve()
    baseline_config = _normalize_baseline_config_path(root, config.launch.baseline_config)
    return config.model_copy(
        update={
            "launch": config.launch.model_copy(
                update={"baseline_config": baseline_config},
            )
        }
    )


def compile_project_spec(
    project_root: str | Path,
    config: ProjectConfig,
    *,
    security_mode: Optional[SecurityMode] = None,
) -> ProjectSpec:
    root = Path(project_root).expanduser().resolve()
    config = normalized_project_config(root, config)
    working_dir = _resolve_project_path(root, config.advanced.working_dir or ".")
    launcher = _render_launcher(root, config)
    baseline_config_path = str(_resolve_project_path(root, config.launch.baseline_config)) if config.launch.baseline_config else None
    output = _resolve_output_config(config.output, baseline_config_path, config.tuning.params)
    tunable_params = _resolve_config_tunable_defaults(config.tuning.params, baseline_config_path)
    return ProjectSpec(
        project_root=str(root),
        working_dir=str(working_dir),
        launcher_template=launcher,
        security_mode=security_mode or config.advanced.security_mode,
        data_paths=[str(_resolve_project_path(root, item.path)) for item in config.data],
        log_paths=config.advanced.log_paths,
        signal_sources=config.advanced.signal_sources,
        wandb_enabled=config.advanced.wandb_enabled,
        heartbeat_interval_sec=config.advanced.heartbeat_interval_sec,
        round_timeout_sec=config.run.timeout_minutes * 60 if config.run.timeout_minutes is not None else None,
        max_rounds=config.run.max_rounds,
        tunable_params=tunable_params,
        baseline_config_path=baseline_config_path,
        output=output,
        visuals=config.visuals,
        metric_specs=config.metrics.specs,
        metric_prompt=config.metrics.prompt,
        tuning_prompt=config.advanced.tuning_prompt,
    )


def detect_project(project_root: str | Path) -> ProjectDiscovery:
    root = Path(project_root).expanduser().resolve()
    environment, env_name = _detect_environment(root)
    entrypoints = _discover_files(
        root,
        ("train.py", "main.py", "run.py", "scripts/train.py", "scripts/run.py", "scripts/train*.py"),
        limit=8,
    )
    data_dirs = _discover_named_dirs(root, {"data", "dataset", "datasets"}, limit=8)
    config_files = [
        path
        for path in _discover_files(
            root,
            ("config.yaml", "config.yml", "configs/*.yaml", "configs/*.yml", "*.yaml", "*.yml"),
            limit=16,
        )
        if Path(path).name not in {"environment.yml", "environment.yaml"}
    ][:8]
    limit_flags = _detect_limit_flags(root / entrypoints[0]) if entrypoints else []
    return ProjectDiscovery(
        environment=environment,
        env_name=env_name,
        entrypoints=entrypoints,
        data_dirs=data_dirs,
        config_files=config_files,
        limit_flags=limit_flags,
    )


def default_project_config(
    project_root: str | Path,
    discovery: Optional[ProjectDiscovery] = None,
    *,
    baseline_config: Optional[str] = None,
) -> ProjectConfig:
    discovery = discovery or detect_project(project_root)
    command = ["python", discovery.entrypoints[0]] if discovery.entrypoints else ["python", "train.py"]
    normalized_baseline = _normalize_baseline_config_path(
        Path(project_root).expanduser().resolve(),
        baseline_config,
    )
    return ProjectConfig(
        data=[DataInput(path=path) for path in discovery.data_dirs],
        launch=LaunchConfig(
            environment=discovery.environment,
            env_name=discovery.env_name,
            command=command,
            baseline_config=normalized_baseline,
        ),
        output=OutputConfig(),
        run=RunConfig(fixed_args=discovery.limit_flags),
    )


def registration_payload(project_root: str | Path, config: ProjectConfig) -> dict[str, Any]:
    return {"project_root": str(Path(project_root).expanduser().resolve()), **config.model_dump(mode="json")}


def _render_launcher(project_root: Path, config: ProjectConfig) -> str:
    if config.advanced.shell_command:
        parts = [config.advanced.shell_command.strip()]
    else:
        command = list(config.launch.command)
        if config.launch.environment == "uv":
            command = ["uv", "run", *command]
        elif config.launch.environment == "venv":
            if command[0] in {"python", "python3"}:
                command[0] = str(project_root / ".venv" / "bin" / "python")
        elif config.launch.environment == "conda":
            command = ["conda", "run", "-n", str(config.launch.env_name), *command]
        parts = [_quote_command(command)]

    if config.launch.baseline_config:
        if _uses_generated_config(config):
            parts.append("--config {config_path}")
        else:
            parts.append(
                _render_arg(
                    CommandArg(
                        flag="--config",
                        value=str(_resolve_project_path(project_root, config.launch.baseline_config)),
                    )
                )
            )
    for item in config.launch.args:
        parts.append(_render_arg(item))
    for item in config.data:
        if item.flag:
            parts.append(_render_arg(CommandArg(flag=item.flag, value=str(_resolve_project_path(project_root, item.path)))))
    for item in config.run.fixed_args:
        parts.append(_render_arg(item))
    if not any("{extra_args}" in part for part in parts):
        parts.append("{extra_args}")
    return " ".join(part for part in parts if part).strip()


def _uses_generated_config(config: ProjectConfig) -> bool:
    return bool(config.launch.baseline_config)


def _resolve_config_tunable_defaults(
    params: list[TunableParam],
    baseline_config_path: Optional[str],
) -> list[TunableParam]:
    if not params:
        return []
    baseline: Optional[dict[str, Any]] = None
    resolved: list[TunableParam] = []
    for param in params:
        if not param.config_path:
            resolved.append(param)
            continue
        if baseline_config_path is None:
            raise ValueError(f"tuning param {param.name} uses config_path but launch.baseline_config is not set")
        if baseline is None:
            baseline = _load_yaml_mapping(Path(baseline_config_path))
        try:
            default = get_config_value(baseline, param.config_path)
        except ValueError as exc:
            raise ValueError(f"tuning param {param.name} config_path is not in launch.baseline_config: {param.config_path}") from exc
        payload = param.model_dump(mode="python")
        payload["default"] = default
        resolved.append(TunableParam.model_validate(payload))
    return resolved


def _resolve_output_config(
    output: Optional[OutputConfig],
    baseline_config_path: Optional[str],
    params: list[TunableParam],
) -> Optional[OutputConfig]:
    if output is None or not output.config_path:
        return None
    if baseline_config_path is None:
        raise ValueError("output.config_path requires launch.baseline_config")
    conflicts = [item.name for item in params if item.config_path == output.config_path]
    if conflicts:
        raise ValueError(
            "output.config_path must not also be listed in tuning.yaml params: "
            + ", ".join(conflicts)
        )
    baseline = _load_yaml_mapping(Path(baseline_config_path))
    try:
        get_config_value(baseline, output.config_path)
    except ValueError as exc:
        raise ValueError(f"output.config_path is not in launch.baseline_config: {output.config_path}") from exc
    return output


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


def _quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in command)


def _render_arg(item: CommandArg) -> str:
    if item.value is None or item.value is True:
        return item.flag
    if item.value is False:
        return ""
    return f"{item.flag} {shlex.quote(str(item.value))}"


def _param_key(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", value.lstrip("-")).strip("_").lower()


def _param_aliases(value: str) -> set[str]:
    key = _param_key(value)
    if not key:
        return set()
    parts = key.split("_")
    singular = "_".join(_singularize_param_part(part) for part in parts)
    return {key, singular}


def _singularize_param_part(value: str) -> str:
    aliases = {
        "iterations": "iter",
        "iteration": "iter",
        "iters": "iter",
        "epochs": "epoch",
    }
    if value in aliases:
        return aliases[value]
    if len(value) > 3 and value.endswith("s") and not value.endswith(("ss", "us", "is")):
        return value[:-1]
    return value


def _resolve_project_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _normalize_baseline_config_path(project_root: Path, raw_path: Optional[str]) -> Optional[str]:
    if raw_path is None:
        return None
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    candidate = candidate.resolve()
    try:
        relative = candidate.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"launch.baseline_config must stay within project_root: {candidate}") from exc
    if not candidate.is_file():
        raise ValueError(f"launch.baseline_config not found: {candidate}")
    return relative.as_posix()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _detect_environment(project_root: Path) -> tuple[LaunchEnvironment, Optional[str]]:
    if (project_root / "uv.lock").is_file():
        return "uv", None
    if (project_root / ".venv" / "bin" / "python").is_file():
        return "venv", None
    for name in ("environment.yml", "environment.yaml"):
        path = project_root / name
        if not path.is_file():
            continue
        match = re.search(r"(?m)^\s*name:\s*([A-Za-z0-9_.-]+)\s*$", path.read_text(encoding="utf-8", errors="ignore"))
        return "conda", match.group(1) if match else project_root.name
    return "system", None


def _discover_files(project_root: Path, patterns: tuple[str, ...], limit: int) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for path in sorted(project_root.glob(pattern)):
            if not path.is_file() or ".trainee" in path.parts:
                continue
            relative = str(path.relative_to(project_root))
            if relative in seen:
                continue
            seen.add(relative)
            found.append(relative)
            if len(found) >= limit:
                return found
    return found


def _discover_named_dirs(project_root: Path, names: set[str], limit: int) -> list[str]:
    found: list[str] = []
    for path in project_root.rglob("*"):
        if not path.is_dir() or path.name.lower() not in names:
            continue
        relative = path.relative_to(project_root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        found.append(str(relative))
        if len(found) >= limit:
            break
    return found


def _detect_limit_flags(entrypoint: Path) -> list[CommandArg]:
    if not entrypoint.is_file():
        return []
    text = entrypoint.read_text(encoding="utf-8", errors="ignore")
    flags = ("max-iter", "max-iters", "max-steps", "limit")
    detected: list[CommandArg] = []
    for flag in flags:
        pattern = re.compile(
            rf"add_argument\(\s*[\"']--{re.escape(flag)}[\"'][^)]*?default\s*=\s*(\d+(?:\.\d+)?)",
            re.DOTALL,
        )
        match = pattern.search(text)
        if match:
            raw = match.group(1)
            value: Any = float(raw) if "." in raw else int(raw)
            detected.append(CommandArg(flag=f"--{flag}", value=value))
    return detected
