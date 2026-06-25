from __future__ import annotations

from pathlib import Path

import yaml

from trainee.context_builder import ContextBuilder
from trainee.project_config import CommandArg, LaunchConfig, ProjectConfig, RunConfig, compile_project_spec
from trainee.tunable_discovery import (
    TunableParamSuggestion,
    apply_tunable_suggestions,
    suggest_tunable_params_heuristic,
)


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
            "stall_timeout_sec": 1.5,
            "signal_sources": [{"type": "stdout"}],
            "log_paths": [".trainee/runs/**/*.log"],
            "wandb_enabled": False,
        },
    }

    assert client.post("/api/project/register", json=registration).status_code == 200
    suggested = client.post("/api/project/tunables/suggest", json={"project_root": str(project), "limit": 2})

    assert suggested.status_code == 200
    suggestions = suggested.json()["suggestions"]
    assert suggestions
    assert {item["config_path"] for item in suggestions} <= {
        "fit.term_weights.theta",
        "fit.term_weights.stretch",
    }

    applied = client.post(
        "/api/project/tunables/apply",
        json={"project_root": str(project), "suggestions": suggestions[:1]},
    )

    assert applied.status_code == 200
    saved = yaml.safe_load((project / ".trainee" / "project.yaml").read_text(encoding="utf-8"))
    assert saved["tuning"]["params"][0]["config_path"] == suggestions[0]["config_path"]
