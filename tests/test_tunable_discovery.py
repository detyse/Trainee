from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from trainee.context_builder import ContextBuilder
from trainee.llm import ProviderCompletion
from trainee.project_config import CommandArg, LaunchConfig, ProjectConfig, RunConfig, compile_project_spec
from trainee.settings import Settings
from trainee.tunable_discovery import (
    TunableDiscoveryEngine,
    TunableDiscoveryRequest,
    TunableParamSuggestion,
    apply_tunable_suggestions,
    suggest_tunable_params_heuristic,
)


def test_tunable_discovery_default_limit_is_32() -> None:
    assert TunableDiscoveryRequest().limit == 32


def test_heuristic_discovery_suggests_config_loss_weights(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    baseline = project / "configs" / "fit.yaml"
    baseline.write_text(
        """
data:
  max_frames: null
fit:
  term_weights:
    theta: 9.0
    stretch: 30.0
  render:
    opacity: 0.7
output:
  root: outputs
""".lstrip(),
        encoding="utf-8",
    )

    config = ProjectConfig(
        launch=LaunchConfig(command=["python", "train.py"], baseline_config="configs/fit.yaml"),
        run=RunConfig(max_rounds=1),
    )
    spec = compile_project_spec(project, config)
    context = ContextBuilder().build(spec)

    result = suggest_tunable_params_heuristic(spec, context)

    paths = [item.config_path for item in result.suggestions]
    assert "fit.term_weights.theta" in paths
    assert "fit.term_weights.stretch" in paths
    assert "fit.render.opacity" not in paths
    assert result.suggestions[0].to_tunable_param().config_path


def test_apply_tunable_suggestions_appends_reviewed_params() -> None:
    config = ProjectConfig(launch=LaunchConfig(command=["python", "train.py"]))
    suggestion = TunableParamSuggestion(
        name="theta_weight",
        config_path="fit.term_weights.theta",
        type="float",
        default=9.0,
        min_value=1.0,
        max_value=15.0,
        reason="test",
    )

    updated, applied = apply_tunable_suggestions(config, [suggestion])

    assert [item.name for item in applied] == ["theta_weight"]
    assert updated.tuning.params[0].config_path == "fit.term_weights.theta"


def test_apply_tunable_suggestions_skips_fixed_arg_exclusions() -> None:
    config = ProjectConfig(
        launch=LaunchConfig(command=["python", "train.py"]),
        run=RunConfig(fixed_args=[CommandArg(flag="--max-iter", value=1000)]),
    )
    suggestions = [
        TunableParamSuggestion(
            name="max_iter",
            config_path="fit.stages.global.max_iters",
            type="int",
            default=1000,
            min_value=100,
            max_value=5000,
            reason="excluded",
        ),
        TunableParamSuggestion(
            name="theta_weight",
            config_path="fit.term_weights.theta",
            type="float",
            default=9.0,
            min_value=1.0,
            max_value=15.0,
            reason="accepted",
        ),
    ]

    updated, applied = apply_tunable_suggestions(config, suggestions)

    assert [item.name for item in applied] == ["theta_weight"]
    assert [item.name for item in updated.tuning.params] == ["theta_weight"]


def test_heuristic_discovery_excludes_fixed_arg_config_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    (project / "configs" / "fit.yaml").write_text(
        """
fit:
  stages:
    global:
      max_iters: 1000
  term_weights:
    theta: 9.0
""".lstrip(),
        encoding="utf-8",
    )
    config = ProjectConfig(
        launch=LaunchConfig(command=["python", "train.py"], baseline_config="configs/fit.yaml"),
        run=RunConfig(fixed_args=[CommandArg(flag="--max-iter", value=1000)]),
    )
    spec = compile_project_spec(project, config)
    context = ContextBuilder().build(spec)

    result = suggest_tunable_params_heuristic(
        spec,
        context,
        exclusions={"--max-iter", "max_iter"},
    )

    paths = {item.config_path for item in result.suggestions}
    assert "fit.term_weights.theta" in paths
    assert "fit.stages.global.max_iters" not in paths


def test_llm_discovery_uses_project_prompt_and_returns_stage_candidates(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    (project / ".trainee").mkdir()
    shutil.copy(repo_root / "_examples" / "fit_20260622.yaml", project / "configs" / "fit.yaml")
    shutil.copy(repo_root / "_examples" / "context.md", project / ".trainee" / "context.md")

    config = ProjectConfig(
        launch=LaunchConfig(command=["python", "-m", "src.run_fitting"], baseline_config="configs/fit.yaml"),
        run=RunConfig(max_rounds=1),
    )
    spec = compile_project_spec(project, config)
    context = ContextBuilder().build(spec)
    fake_engine = _FakeDiscoveryDecisionEngine()

    result = _run_async(
        TunableDiscoveryEngine(
            _settings(tmp_path, repo_root, project),
            llm_client=fake_engine,  # type: ignore[arg-type]
        ).suggest(spec, context)
    )

    prompt_payload = fake_engine.user_payload
    prompt_documents = prompt_payload["prompt_documents"]
    assert any(item["path"] == ".trainee/context.md" for item in prompt_documents)
    assert "fit.stages.keypoint.max_iters" in prompt_payload["numeric_config_leaves"]
    assert "fit.stages.keypoint.render_keypoints" in prompt_payload["scalar_config_leaves"]
    assert "fit.stages.<stage>.max_iters" in prompt_documents[0]["text"]

    candidate_targets = {item.target for item in result.candidates}
    suggestion_paths = {item.config_path for item in result.suggestions}
    assert "fit.stages.keypoint.max_iters" in candidate_targets
    assert "fit.stages.keypoint.render_keypoints" in candidate_targets
    assert "fit.stages.keypoint.max_iters" in suggestion_paths
    assert "fit.stages.silhouette.term_weights.theta" in suggestion_paths


def test_tunable_discovery_api_suggests_then_applies(runtime_env) -> None:
    client = runtime_env["client"]
    project = runtime_env["external_project"]
    python = runtime_env["python"]

    (project / "configs").mkdir(exist_ok=True)
    (project / "configs" / "fit.yaml").write_text(
        """
fit:
  term_weights:
    theta: 9.0
    stretch: 30.0
""".lstrip(),
        encoding="utf-8",
    )
    registration = {
        "project_root": str(project),
        "version": 1,
        "data": [{"path": "data"}],
        "launch": {
            "environment": "system",
            "command": [python, "train.py"],
            "baseline_config": "configs/fit.yaml",
            "args": [],
        },
        "run": {"max_rounds": 1, "timeout_minutes": 1, "fixed_args": []},
        "tuning": {"params": []},
        "metrics": {"specs": [], "prompt": ""},
        "advanced": {
            "security_mode": "unsafe",
            "working_dir": ".",
            "heartbeat_interval_sec": 0.1,
            "signal_sources": [{"type": "stdout"}],
            "log_paths": [".trainee/runs/**/*.log"],
            "wandb_enabled": False,
        },
    }

    assert client.post("/api/project/register", json=registration).status_code == 200
    suggested = client.post("/api/project/tunables/suggest", json={"project_root": str(project), "limit": 2})

    assert suggested.status_code == 200
    candidates = suggested.json()["candidates"]
    suggestions = suggested.json()["suggestions"]
    assert candidates
    assert suggestions
    assert {item["target"] for item in candidates} <= {
        "fit.term_weights.theta",
        "fit.term_weights.stretch",
    }
    assert {item["config_path"] for item in suggestions} <= {
        "fit.term_weights.theta",
        "fit.term_weights.stretch",
    }

    applied = client.post(
        "/api/project/tunables/apply",
        json={"project_root": str(project), "suggestions": suggestions[:1]},
    )

    assert applied.status_code == 200
    project_saved = yaml.safe_load((project / ".trainee" / "project.yaml").read_text(encoding="utf-8"))
    saved = yaml.safe_load((project / ".trainee" / "tuning.yaml").read_text(encoding="utf-8"))
    assert "tuning" not in project_saved
    assert saved["params"][0]["config_path"] == suggestions[0]["config_path"]
    assert "default" not in saved["params"][0]


class _FakeDiscoveryDecisionEngine:
    def __init__(self) -> None:
        self.user_payload: dict[str, object] = {}

    async def complete_active(self, system_prompt: str, user_prompt: str) -> ProviderCompletion:
        assert "auditable" in system_prompt
        self.user_payload = json.loads(user_prompt)
        return ProviderCompletion(
            content=json.dumps(
                {
                    "candidates": [
                        {
                            "name": "keypoint_max_iters",
                            "target_kind": "config_path",
                            "target": "fit.stages.keypoint.max_iters",
                            "type": "int",
                            "applicability": "auto_applyable",
                            "risk": "medium",
                            "reason": "Stage iteration budget is explicitly called out in the project prompt.",
                            "evidence": [".trainee/context.md", "baseline stage config"],
                            "confidence": 0.9,
                        },
                        {
                            "name": "silhouette_theta_weight",
                            "target_kind": "config_path",
                            "target": "fit.stages.silhouette.term_weights.theta",
                            "type": "float",
                            "applicability": "auto_applyable",
                            "risk": "medium",
                            "reason": "Silhouette-stage theta regularization controls implausible articulation.",
                            "evidence": [".trainee/context.md", "baseline stage config"],
                            "confidence": 0.9,
                        },
                        {
                            "name": "keypoint_render_keypoints",
                            "target_kind": "config_path",
                            "target": "fit.stages.keypoint.render_keypoints",
                            "type": "bool",
                            "applicability": "needs_review",
                            "risk": "low",
                            "reason": "Useful for review, but not an auto numeric tuning parameter.",
                            "evidence": [".trainee/context.md"],
                            "confidence": 0.7,
                        },
                    ]
                }
            ),
            raw_response_body="{}",
            http_status=200,
        )


def _settings(tmp_path: Path, repo_root: Path, project_root: Path) -> Settings:
    data_dir = tmp_path / "runtime-data"
    return Settings(
        repo_root=repo_root,
        project_root=project_root,
        project_data_dir=data_dir,
        database_path=data_dir / "runtime.sqlite3",
        artifacts_dir=data_dir / "artifacts",
        template_dir=repo_root / "src" / "trainee" / "templates",
        static_dir=repo_root / "src" / "trainee" / "static",
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
    )


def _run_async(coro):
    import asyncio

    return asyncio.run(coro)
