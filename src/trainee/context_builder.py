from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from trainee.models import ProjectContext, ProjectSpec


class ContextBuilder:
    def build(self, spec: ProjectSpec) -> ProjectContext:
        project_root = Path(spec.project_root).expanduser().resolve()
        working_dir = Path(spec.working_dir).expanduser().resolve()
        readme_text = self._read_readme(project_root)
        entrypoints = self._discover_entrypoints(project_root)
        configs = self._discover_configs(project_root)
        detected_data_dirs = self._discover_named_dirs(project_root, {"data", "dataset", "datasets"})
        detected_log_dirs = self._discover_named_dirs(project_root, {"logs", "outputs", "runs", "wandb"})
        warnings: List[str] = []

        if not readme_text:
            warnings.append("README not found in external project root; project summary is based on launcher and file heuristics.")
        if not spec.data_paths and not detected_data_dirs:
            warnings.append("No explicit data_paths configured and no common data directories were detected.")
        if not spec.metric_specs:
            warnings.append("No metric_specs configured; only built-in loss parsing will be available.")
        if spec.wandb_enabled and not detected_log_dirs:
            warnings.append("W&B is enabled, but no common log directories were detected under the project root.")

        summary_parts = []
        if readme_text:
            summary_parts.append(readme_text)
        summary_parts.append(f"Working directory: {working_dir}")
        summary_parts.append(f"Launcher template: {spec.launcher_template}")
        project_summary = "\n\n".join(part for part in summary_parts if part)

        entry_lines = [f"Configured launcher: {spec.launcher_template}"]
        if entrypoints:
            entry_lines.append("Detected training-oriented entrypoints: " + ", ".join(entrypoints))
        if configs:
            entry_lines.append("Nearby config files: " + ", ".join(configs))
        training_entrypoint_summary = "\n".join(entry_lines)

        data_lines = []
        if spec.data_paths:
            data_lines.append("Configured data paths: " + ", ".join(spec.data_paths))
        if detected_data_dirs:
            data_lines.append("Detected project data directories: " + ", ".join(detected_data_dirs))
        if spec.log_paths:
            data_lines.append("Configured legacy log paths: " + ", ".join(spec.log_paths))
        if spec.signal_log_paths():
            data_lines.append("Configured signal log paths: " + ", ".join(spec.signal_log_paths()))
        if spec.metric_log_paths():
            data_lines.append("Configured metric log paths: " + ", ".join(spec.metric_log_paths()))
        if detected_log_dirs:
            data_lines.append("Detected log directories: " + ", ".join(detected_log_dirs))
        data_summary = "\n".join(data_lines) or "No data or log directories were identified yet."

        parameter_lines = []
        if spec.tunable_params:
            rendered = []
            for item in spec.tunable_params:
                target = item.config_path or item.flag or "unset"
                target_kind = "config" if item.config_path else "cli"
                bits = [f"{item.name} ({target_kind}:{target}, {item.type})"]
                if item.default is not None:
                    bits.append(f"default={item.default}")
                if item.min_value is not None or item.max_value is not None:
                    bits.append(f"range=[{item.min_value}, {item.max_value}]")
                if item.choices:
                    bits.append(f"choices={item.choices}")
                rendered.append(", ".join(bits))
            parameter_lines.append("Tunable CLI parameters: " + " | ".join(rendered))
        else:
            parameter_lines.append("No tunable_params configured yet.")
        if spec.tuning_prompt:
            parameter_lines.append("Tuning prompt: " + spec.tuning_prompt.strip())
        parameter_summary = "\n".join(parameter_lines)

        result_lines = []
        if spec.metric_specs:
            result_lines.append(
                "Primary metrics: "
                + " | ".join(
                    f"{item.name} via {item.source} ({item.goal}, required={item.required})"
                    for item in spec.metric_specs
                )
            )
        else:
            result_lines.append("Built-in parser will look for loss/total_loss in logs.")
        if spec.metric_prompt:
            result_lines.append("Metric prompt: " + spec.metric_prompt.strip())
        result_lines.append("W&B tracking is enabled." if spec.wandb_enabled else "W&B tracking is disabled.")
        if spec.log_paths:
            result_lines.append("Legacy log paths: " + ", ".join(spec.log_paths))
        if spec.signal_sources:
            result_lines.append(
                "Signal sources: "
                + " | ".join(f"{item.type}: {', '.join(item.configured_paths()) or 'process output'}" for item in spec.signal_sources)
            )
        if spec.metric_log_paths():
            result_lines.append("Metric log files: " + ", ".join(spec.metric_log_paths()))
        result_reading_summary = "\n".join(result_lines)

        return ProjectContext(
            project_summary=project_summary,
            training_entrypoint_summary=training_entrypoint_summary,
            data_summary=data_summary,
            parameter_summary=parameter_summary,
            result_reading_summary=result_reading_summary,
            warnings=warnings,
        )

    def _read_readme(self, project_root: Path) -> str:
        for name in ("README.md", "README.rst", "README.txt", "readme.md"):
            path = project_root / name
            if path.exists() and path.is_file():
                return self._truncate(path.read_text(encoding="utf-8", errors="ignore").strip())
        return ""

    def _discover_entrypoints(self, project_root: Path) -> List[str]:
        patterns = ("train*.py", "main.py", "run*.py", "scripts/*.py")
        return self._collect_matches(project_root, patterns, limit=8, name_filter={"train", "main", "run"})

    def _discover_configs(self, project_root: Path) -> List[str]:
        patterns = ("*.yaml", "*.yml", "*.json", "*.toml")
        return self._collect_matches(project_root, patterns, limit=8, name_filter={"train", "config", "hparam", "sweep"})

    def _discover_named_dirs(self, project_root: Path, names: Iterable[str]) -> List[str]:
        results: List[str] = []
        allowed = {item.lower() for item in names}
        for path in project_root.rglob("*"):
            if not path.is_dir():
                continue
            if any(part.startswith(".") for part in path.parts if part != project_root.name):
                continue
            if path.name.lower() in allowed:
                results.append(str(path.relative_to(project_root)))
            if len(results) >= 8:
                break
        return results

    def _collect_matches(self, project_root: Path, patterns: Iterable[str], limit: int, name_filter: set[str]) -> List[str]:
        matches: List[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            for path in project_root.rglob(pattern):
                if not path.is_file():
                    continue
                relative = str(path.relative_to(project_root))
                lowered = relative.lower()
                if any(part.startswith(".") for part in path.parts if part != project_root.name):
                    continue
                if name_filter and not any(token in lowered for token in name_filter):
                    continue
                if relative in seen:
                    continue
                seen.add(relative)
                matches.append(relative)
                if len(matches) >= limit:
                    return matches
        return matches

    def _truncate(self, text: str, limit: int = 1200) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."
