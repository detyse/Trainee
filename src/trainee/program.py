from __future__ import annotations

from pathlib import Path


DEFAULT_AGENT_PROGRAM = """# Trainee Agent Rules

You are a conservative training automation agent.

- Use the project context, configured metrics, and completed round history to decide the next action.
- Change only parameters listed in `tunable_params`.
- Prefer controlled, explainable parameter changes over broad random changes.
- Respect each metric's configured optimization goal.
- Explain the evidence behind every parameter change.
- Stop when there is no safe, useful next experiment or when the evidence is insufficient.
"""


def program_path(project_root: str | Path) -> Path:
    root = Path(project_root).expanduser().resolve()
    return root / ".trainee" / "program.md"


def write_program(project_root: str | Path, content: str) -> Path:
    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"project root does not exist or is not a directory: {root}")
    path = program_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = content.rstrip() + "\n" if content.strip() else ""
    path.write_text(normalized, encoding="utf-8")
    return path
