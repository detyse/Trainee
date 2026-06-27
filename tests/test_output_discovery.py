from __future__ import annotations

import asyncio
import json
from pathlib import Path

from trainee.context_builder import ContextBuilder
from trainee.decision import ProviderCompletion
from trainee.output_discovery import OutputDiscoveryEngine
from trainee.project_config import LaunchConfig, ProjectConfig, compile_project_spec
from trainee.settings import Settings


def test_output_discovery_accepts_existing_agent_selected_key(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    (project / "configs" / "fit.yaml").write_text(
        "data:\n  root: data\noutput:\n  root: outputs\n",
        encoding="utf-8",
    )
    spec = compile_project_spec(
        project,
        ProjectConfig(
            launch=LaunchConfig(command=["python", "train.py"], baseline_config="configs/fit.yaml"),
        ),
    )
    context = ContextBuilder().build(spec)
    fake = _FakeOutputDecision("output.root")

    result = asyncio.run(OutputDiscoveryEngine(_settings(tmp_path, project), decision_engine=fake).suggest(spec, context))

    assert fake.user_payload["scalar_config_leaves"]["output.root"] == "outputs"
    assert result.output is not None
    assert result.output.config_path == "output.root"
    assert result.candidates[0].current_value == "outputs"


def test_output_discovery_rejects_agent_invented_key(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    (project / "configs" / "fit.yaml").write_text("output:\n  root: outputs\n", encoding="utf-8")
    spec = compile_project_spec(
        project,
        ProjectConfig(
            launch=LaunchConfig(command=["python", "train.py"], baseline_config="configs/fit.yaml"),
        ),
    )
    fake = _FakeOutputDecision("output.missing")

    result = asyncio.run(OutputDiscoveryEngine(_settings(tmp_path, project), decision_engine=fake).suggest(spec, ContextBuilder().build(spec)))

    assert result.output is None
    assert result.candidates == []
    assert "selected output config_path is not an existing scalar key." in result.warnings


class _FakeOutputDecision:
    def __init__(self, selected_path: str) -> None:
        self.selected_path = selected_path
        self.user_payload: dict[str, object] = {}

    async def _provider_complete(self, system_prompt: str, user_prompt: str) -> ProviderCompletion:
        assert "output directory field" in system_prompt
        self.user_payload = json.loads(user_prompt)
        content = {
            "selected": {
                "config_path": self.selected_path,
                "current_value": "outputs",
                "confidence": 0.95,
                "reason": "The key is under output and points at the output directory.",
                "evidence": ["baseline config"],
            },
            "candidates": [
                {
                    "config_path": self.selected_path,
                    "current_value": "outputs",
                    "confidence": 0.95,
                    "reason": "The key is under output and points at the output directory.",
                    "evidence": ["baseline config"],
                }
            ],
        }
        return ProviderCompletion(content=json.dumps(content), raw_response_body="{}", http_status=200)


def _settings(tmp_path: Path, project: Path) -> Settings:
    data_dir = tmp_path / "runtime"
    return Settings(
        repo_root=project,
        project_root=project,
        project_data_dir=data_dir,
        database_path=data_dir / "runtime.sqlite3",
        artifacts_dir=data_dir / "artifacts",
        template_dir=project / "templates",
        static_dir=project / "static",
        global_config_path=tmp_path / "home" / ".trainee" / "config.json",
        llm_provider="openai",
        llm_timeout_sec=5.0,
        openai_api_key="test-key",
        openai_base_url="https://api.openai.com/v1",
        openai_model="test-model",
        anthropic_api_key=None,
        anthropic_base_url="https://api.anthropic.com",
        anthropic_model="claude-3-5-haiku-latest",
        anthropic_version="2023-06-01",
        anthropic_max_tokens=1024,
        system_prompt="test",
    )
