from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from trainee.models import SecurityMode

SANDBOX_DIRS = {
    "HOME": "home",
    "XDG_CACHE_HOME": "cache",
    "HF_HOME": "hf_home",
    "TORCH_HOME": "torch",
    "MPLCONFIGDIR": "matplotlib",
    "WANDB_DIR": "wandb",
}


@dataclass(frozen=True)
class SecureCommand:
    argv: list[str]
    env: dict[str, str]
    cwd: Path | None


def build_secure_command(
    *,
    project_root: Path,
    working_dir: Path,
    command: str,
    security_mode: SecurityMode,
    base_env: Mapping[str, str] | None = None,
) -> SecureCommand:
    project_root = project_root.expanduser().resolve()
    working_dir = working_dir.expanduser().resolve()
    _ensure_within(working_dir, project_root, "working_dir")

    trainee_dir = project_root / ".trainee"
    env = _sandbox_env(trainee_dir, base_env)

    if security_mode == "unsafe":
        return SecureCommand(argv=["/bin/bash", "-lc", command], env=env, cwd=working_dir)

    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError(
            "guarded run requires bubblewrap (`bwrap`). Install bubblewrap or rerun with `--unsafe`."
        )

    argv = [
        bwrap,
        "--die-with-parent",
        "--ro-bind",
        "/",
        "/",
        "--tmpfs",
        "/tmp",
        *_project_mount_args(project_root),
        "--bind",
        str(trainee_dir),
        str(trainee_dir),
        "--proc",
        "/proc",
        "--dev-bind",
        "/dev",
        "/dev",
    ]
    for key in SANDBOX_DIRS:
        argv.extend(["--setenv", key, env[key]])
    argv.extend(["--chdir", str(working_dir), "/bin/bash", "-lc", command])
    return SecureCommand(argv=argv, env=env, cwd=None)


def project_trainee_dir(project_root: Path) -> Path:
    return project_root.expanduser().resolve() / ".trainee"


def _sandbox_env(trainee_dir: Path, base_env: Mapping[str, str] | None) -> dict[str, str]:
    trainee_dir.mkdir(parents=True, exist_ok=True)
    for relative_path in [*SANDBOX_DIRS.values(), "runs", "logs"]:
        (trainee_dir / relative_path).mkdir(parents=True, exist_ok=True)

    env = dict(base_env or os.environ)
    for key, relative_path in SANDBOX_DIRS.items():
        env[key] = str(trainee_dir / relative_path)
    return env


def _ensure_within(path: Path, root: Path, field_name: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field_name} must stay within project_root: {path}") from exc


def _project_mount_args(project_root: Path) -> list[str]:
    tmp_root = Path("/tmp")
    try:
        relative = project_root.relative_to(tmp_root)
    except ValueError:
        return []

    args: list[str] = []
    current = tmp_root
    for part in relative.parts:
        current = current / part
        args.extend(["--dir", str(current)])
    args.extend(["--ro-bind", str(project_root), str(project_root)])
    return args
