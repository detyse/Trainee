from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

from pydantic import ValidationError

from trainee.models import ProjectSpec, TunableParam
from trainee.security import SANDBOX_DIRS
from trainee.settings import Settings, load_settings

DoctorStatus = Literal["ok", "warn", "fail"]


@dataclass(frozen=True)
class DoctorFinding:
    status: DoctorStatus
    message: str
    detail: str = ""
    fix: str = ""


@dataclass
class DoctorSection:
    name: str
    findings: list[DoctorFinding] = field(default_factory=list)

    def add(self, status: DoctorStatus, message: str, detail: str = "", fix: str = "") -> None:
        self.findings.append(DoctorFinding(status=status, message=message, detail=detail, fix=fix))


@dataclass
class DoctorReport:
    project_root: Path
    sections: list[DoctorSection]
    fixes: list[str]

    @property
    def has_failures(self) -> bool:
        return any(finding.status == "fail" for section in self.sections for finding in section.findings)


def run_doctor(project_root: Path) -> DoctorReport:
    project_root = project_root.expanduser().resolve()
    sections: list[DoctorSection] = []
    fixes: list[str] = []

    project_section, spec = _check_project(project_root)
    sections.append(project_section)
    _collect_fixes(project_section, fixes)

    environment_section = _check_environment(project_root, spec)
    sections.append(environment_section)
    _collect_fixes(environment_section, fixes)

    launcher_section = _check_launcher(project_root, spec)
    sections.append(launcher_section)
    _collect_fixes(launcher_section, fixes)

    sandbox_section = _check_sandbox(project_root, spec)
    sections.append(sandbox_section)
    _collect_fixes(sandbox_section, fixes)

    llm_section = _check_llm(project_root)
    sections.append(llm_section)
    _collect_fixes(llm_section, fixes)

    return DoctorReport(project_root=project_root, sections=sections, fixes=fixes)


def format_doctor_report(report: DoctorReport) -> str:
    lines = ["Trainee doctor", ""]
    for section in report.sections:
        lines.append(section.name)
        for finding in section.findings:
            lines.append(f"  [{finding.status}] {finding.message}")
            if finding.detail:
                lines.append(f"      {finding.detail}")
        lines.append("")

    lines.append("Result")
    if report.has_failures:
        lines.append("  not ready")
        if report.fixes:
            lines.extend(["", "Fix:"])
            lines.extend(f"  {fix}" for fix in report.fixes)
    else:
        lines.extend(["  ready to run:", "    trainee run"])
    return "\n".join(lines).rstrip() + "\n"


def _check_project(project_root: Path) -> tuple[DoctorSection, ProjectSpec | None]:
    section = DoctorSection("Project")
    spec: ProjectSpec | None = None
    section.add("ok" if project_root.is_dir() else "fail", f"root: {project_root}")
    if not project_root.is_dir():
        return section, None

    trainee_dir = project_root / ".trainee"
    project_json = trainee_dir / "project.json"
    for path, label in (
        (trainee_dir, ".trainee exists"),
        (project_json, ".trainee/project.json exists"),
        (trainee_dir / "runs", ".trainee/runs exists"),
        (trainee_dir / "logs", ".trainee/logs exists"),
    ):
        if path.exists():
            section.add("ok", label)
        else:
            section.add("fail", f"{label}: missing", fix="trainee init")

    if project_json.is_file():
        try:
            payload = json.loads(project_json.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("project.json must contain a JSON object")
            spec = ProjectSpec.model_validate(payload)
            section.add("ok", "project.json is valid")
        except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
            section.add("fail", f"project.json is invalid: {exc}", fix="trainee init --force")

    _check_git(project_root, section)
    return section, spec


def _check_git(project_root: Path, section: DoctorSection) -> None:
    repo = _run_command(["git", "-C", str(project_root), "rev-parse", "--is-inside-work-tree"], timeout=5)
    if repo is None or repo.returncode != 0 or repo.stdout.strip() != "true":
        section.add("warn", "git repo not detected")
        return
    section.add("ok", "git repo detected")

    status = _run_command(["git", "-C", str(project_root), "status", "--short"], timeout=5)
    if status is None or status.returncode != 0:
        section.add("warn", "could not read git status")
        return
    if status.stdout.strip():
        section.add("warn", "working tree has uncommitted changes")
    else:
        section.add("ok", "working tree clean")


def _check_environment(project_root: Path, spec: ProjectSpec | None) -> DoctorSection:
    section = DoctorSection("Environment")
    launcher = spec.launcher_template if spec else ""

    if _is_uv_project(project_root):
        section.add("ok", "uv project detected")
        uv = shutil.which("uv")
        if uv is None:
            section.add("fail", "uv not found", fix="install uv")
            return section
        section.add("ok", f"uv found: {uv}")
        result = _run_command([uv, "sync", "--help"], cwd=project_root, timeout=10)
        if result is not None and result.returncode == 0:
            section.add("ok", "uv sync available")
        else:
            section.add("fail", "uv sync is not available")
        return section

    venv = project_root / ".venv"
    if venv.exists():
        section.add("ok", ".venv detected")
        python = venv / "bin" / "python"
        if python.is_file():
            section.add("ok", f".venv python found: {python}")
        else:
            section.add("fail", ".venv/bin/python not found", fix="python -m venv .venv")
        return section

    conda_env = _detect_conda_env(project_root, launcher)
    if conda_env is not None or (project_root / "environment.yml").is_file() or (project_root / "environment.yaml").is_file():
        section.add("ok", "conda project detected")
        conda = shutil.which("conda")
        if conda is None:
            section.add("fail", "conda not found", fix="install conda or adjust launcher_template")
            return section
        section.add("ok", f"conda found: {conda}")
        if conda_env is None:
            section.add("warn", "conda env name not detected")
            return section
        if _conda_env_exists(conda, conda_env):
            section.add("ok", f"conda env found: {conda_env}")
        else:
            section.add("fail", f"conda env {conda_env} not found", fix=f"conda env create -n {conda_env}")
        return section

    python = shutil.which("python3") or shutil.which("python")
    if python is None:
        section.add("fail", "system python not found", fix="install python")
    else:
        section.add("ok", f"system python found: {python}")
    return section


def _check_launcher(project_root: Path, spec: ProjectSpec | None) -> DoctorSection:
    section = DoctorSection("Launcher")
    if spec is None:
        section.add("fail", "launcher unavailable because project config is invalid", fix="trainee init")
        return section

    launcher = spec.launcher_template.strip()
    if not launcher:
        section.add("fail", "launcher_template is empty", fix="edit .trainee/project.json")
        return section

    entrypoint = _detect_train_entrypoint(launcher)
    if entrypoint:
        section.add("ok", f"train entrypoint: {entrypoint}")
    else:
        section.add("warn", "train entrypoint not detected")

    for operator in _suspicious_shell_operators(launcher):
        section.add("warn", f"launcher contains suspicious shell operator: {operator}")

    output_targets = _launcher_output_targets(launcher)
    if any(target.name == "output_dir" for target in output_targets):
        unsafe = [target for target in output_targets if not _is_safe_trainee_path(target.value)]
        if unsafe:
            for target in unsafe:
                section.add("warn", f"{target.name} may write outside .trainee: {target.value}")
        else:
            section.add("ok", "output_dir points to .trainee/runs")
    else:
        section.add("warn", "launcher has no output_dir")
        for target in output_targets:
            if not _is_safe_trainee_path(target.value):
                section.add("warn", f"{target.name} may write outside .trainee: {target.value}")

    unbounded = _unbounded_tunable_params(spec.tunable_params)
    if unbounded:
        section.add("warn", "tunable params are not bounded: " + ", ".join(unbounded))
    else:
        section.add("ok", "tunable params are bounded")
    return section


def _check_sandbox(project_root: Path, spec: ProjectSpec | None) -> DoctorSection:
    section = DoctorSection("Sandbox")
    security_mode = spec.security_mode if spec is not None else "guarded"
    trainee_dir = project_root / ".trainee"

    if security_mode == "unsafe":
        section.add("warn", "security_mode is unsafe; project writes are not sandbox-restricted")
    else:
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            section.add("fail", "bwrap not found", detail="install: sudo apt install bubblewrap", fix="sudo apt install bubblewrap")
        else:
            section.add("ok", f"bwrap found: {bwrap}")

    if trainee_dir.is_dir() and os.access(trainee_dir, os.W_OK):
        section.add("ok", ".trainee is writable")
    elif trainee_dir.exists():
        section.add("fail", ".trainee is not writable", fix="fix .trainee permissions")
    else:
        section.add("fail", ".trainee is missing", fix="trainee init")

    if security_mode == "guarded":
        section.add("ok", "writable paths: .trainee")
        section.add("ok", "project will be read-only during training")
        redirected = ", ".join(sorted(SANDBOX_DIRS))
        section.add("ok", f"sandbox env redirects into .trainee: {redirected}")
    return section


def _check_llm(project_root: Path) -> DoctorSection:
    section = DoctorSection("LLM")
    try:
        settings = load_settings(repo_root=project_root, project_root=project_root)
    except (OSError, ValueError) as exc:
        section.add("warn", f"LLM config could not be read: {exc}")
        return section

    provider = settings.llm_provider
    if provider == "none":
        section.add("warn", "no LLM key found, will use heuristic mode")
        return section

    if _provider_has_key(settings, provider):
        section.add("ok", f"provider: {provider}")
    else:
        section.add("warn", f"provider {provider} has no API key configured")
    return section


def _collect_fixes(section: DoctorSection, fixes: list[str]) -> None:
    for finding in section.findings:
        if finding.fix and finding.fix not in fixes:
            fixes.append(finding.fix)


def _run_command(
    argv: list[str],
    *,
    cwd: Path | None = None,
    timeout: float,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _is_uv_project(project_root: Path) -> bool:
    if (project_root / "uv.lock").is_file():
        return True
    pyproject = project_root / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return True
    return "[tool.uv]" in text or pyproject.is_file()


def _detect_conda_env(project_root: Path, launcher: str) -> str | None:
    try:
        tokens = shlex.split(launcher)
    except ValueError:
        tokens = launcher.split()
    for index, token in enumerate(tokens):
        if token in {"-n", "--name"} and index > 0 and tokens[index - 1] == "run" and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith("--name=") and index > 0 and tokens[index - 1] == "run":
            return token.split("=", 1)[1]
    for path in (project_root / "environment.yml", project_root / "environment.yaml"):
        if not path.is_file():
            continue
        try:
            match = re.search(r"(?m)^\s*name:\s*([A-Za-z0-9_.-]+)\s*$", path.read_text(encoding="utf-8"))
        except OSError:
            match = None
        if match:
            return match.group(1)
    return None


def _conda_env_exists(conda: str, env_name: str) -> bool:
    result = _run_command([conda, "env", "list", "--json"], timeout=10)
    if result is None or result.returncode != 0:
        return False
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False
    envs = payload.get("envs", []) if isinstance(payload, dict) else []
    return any(Path(item).name == env_name for item in envs if isinstance(item, str))


def _detect_train_entrypoint(launcher: str) -> str:
    try:
        tokens = shlex.split(launcher)
    except ValueError:
        tokens = launcher.split()
    for token in tokens:
        if token.endswith((".py", ".sh")):
            return token
    return ""


def _suspicious_shell_operators(launcher: str) -> list[str]:
    operators = []
    for operator in (">>", "&&", "||", ";", "`", "$("):
        if operator in launcher:
            operators.append(operator)
    if re.search(r"(?<!>)>(?!>)", launcher):
        operators.append(">")
    if re.search(r"(?<!\|)\|(?!\|)", launcher):
        operators.append("|")
    return operators


@dataclass(frozen=True)
class _OutputTarget:
    name: str
    value: str


def _launcher_output_targets(launcher: str) -> list[_OutputTarget]:
    try:
        tokens = shlex.split(launcher)
    except ValueError:
        tokens = launcher.split()

    targets: list[_OutputTarget] = []
    names = {"output_dir", "save_dir", "log_dir", "checkpoint_dir", "wandb_dir"}
    for index, token in enumerate(tokens):
        if token.startswith("WANDB_DIR="):
            targets.append(_OutputTarget("WANDB_DIR", token.split("=", 1)[1]))
            continue
        normalized = token.lstrip("-").replace("-", "_")
        if "=" in normalized:
            name, value = normalized.split("=", 1)
            if name in names or name == "WANDB_DIR":
                targets.append(_OutputTarget(name, value))
            continue
        if normalized in names and index + 1 < len(tokens):
            targets.append(_OutputTarget(normalized, tokens[index + 1]))
    return targets


def _is_safe_trainee_path(value: str) -> bool:
    value = value.strip().strip("\"'")
    if not value:
        return False
    safe_prefixes = (
        ".trainee/",
        "./.trainee/",
        "{trainee_dir}",
        "{trainee_dir}/",
        "{session_dir}",
        "{session_dir}/",
        "{round_dir}",
        "{round_dir}/",
        "{config_path}",
        "{project_root}/.trainee/",
        "$TRAINEE_DIR/",
        "$TRAINEE_SESSION_DIR/",
        "$TRAINEE_ROUND_DIR/",
        "$TRAINEE_CONFIG_PATH",
    )
    return value == ".trainee" or value.startswith(safe_prefixes)


def _unbounded_tunable_params(params: Iterable[TunableParam]) -> list[str]:
    unbounded: list[str] = []
    for param in params:
        if param.type in {"int", "float"}:
            if param.min_value is None or param.max_value is None:
                unbounded.append(param.name)
        elif param.type == "str" and not param.choices:
            unbounded.append(param.name)
    return unbounded


def _provider_has_key(settings: Settings, provider: str) -> bool:
    if provider == "openai":
        return bool(settings.openai_api_key)
    if provider == "anthropic":
        return bool(settings.anthropic_api_key)
    if provider == "moonshot":
        return bool(settings.moonshot_api_key)
    return False
