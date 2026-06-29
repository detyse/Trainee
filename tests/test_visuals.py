from __future__ import annotations

import os
from pathlib import Path

from trainee.executor import TrainingExecutor
from trainee.models import ProjectSpec, RoundRecord, RunSession, VisualAnalysisResult
from trainee.project_config import LaunchConfig, ProjectConfig, compile_project_spec, save_project_config
from trainee.storage import Storage
from trainee.visuals import VisualAnalyzer


def test_visuals_config_compiles_and_round_trips_to_project_yaml(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = ProjectConfig(
        launch=LaunchConfig(command=["python", "train.py"]),
        visuals={
            "enabled": True,
            "patterns": ["{round_output_dir}/**/*.png"],
            "max_images_per_round": 2,
            "selection": "newest",
            "prompt": "Read loss and validation plots.",
        },
    )

    spec = compile_project_spec(project, config)
    save_project_config(project, config)
    reloaded = compile_project_spec(project, ProjectConfig.model_validate(config.model_dump(mode="json")))

    assert spec.visuals.enabled is True
    assert spec.visuals.patterns == ["{round_output_dir}/**/*.png"]
    assert reloaded.visuals.max_images_per_round == 2
    assert "visuals:" in (project / ".trainee" / "project.yaml").read_text(encoding="utf-8")


def test_visual_analyzer_selects_newest_matching_images(tmp_path: Path) -> None:
    spec = ProjectSpec(
        project_root=str(tmp_path),
        working_dir=str(tmp_path),
        launcher_template="python train.py",
        visuals={
            "enabled": True,
            "patterns": ["{round_dir}/outputs/*.png"],
            "max_images_per_round": 2,
            "selection": "newest",
        },
    )
    executor = TrainingExecutor()
    workspace = executor.round_workspace(spec, session_id=1, round_index=1)
    output_dir = workspace.round_dir / "outputs"
    output_dir.mkdir(parents=True)
    paths = [
        output_dir / "old.png",
        output_dir / "middle.png",
        output_dir / "new.png",
    ]
    for index, path in enumerate(paths, start=1):
        path.write_bytes(b"image")
        os.utime(path, (index, index))

    selected = VisualAnalyzer(executor).select_images(spec, session_id=1, round_index=1)

    assert [path.name for path in selected] == ["new.png", "middle.png"]


def test_visual_analyzer_builds_structured_observations(tmp_path: Path) -> None:
    spec = ProjectSpec(
        project_root=str(tmp_path),
        working_dir=str(tmp_path),
        launcher_template="python train.py",
        visuals={
            "enabled": True,
            "patterns": ["{round_dir}/outputs/*.png"],
            "max_images_per_round": 1,
            "selection": "newest",
            "prompt": "Analyze loss plots.",
        },
    )
    executor = TrainingExecutor()
    image_path = executor.round_workspace(spec, session_id=2, round_index=3).round_dir / "outputs" / "loss.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake-png")
    calls: list[dict[str, object]] = []

    class FakeDecisionEngine:
        async def analyze_image(self, prompt, image):
            calls.append({"prompt": prompt, "image": image})
            return {
                "content": (
                    '{"likely_meaning":"validation loss plateau",'
                    '"visible_signals":["flat validation curve"],'
                    '"concerns":["possible plateau"],'
                    '"decision_relevant_observations":["validation is no longer improving"],'
                    '"confidence":0.8}'
                )
            }

    result = _run(
        VisualAnalyzer(executor).analyze_round(
            spec=spec,
            session_id=2,
            round_index=3,
            decision_engine=FakeDecisionEngine(),  # type: ignore[arg-type]
            reserve_image_analysis=lambda: {"session_id": 2, "used": 1, "limit": 3},
        )
    )

    assert result is not None
    assert result.status == "completed"
    assert result.image_paths == [".trainee/runs/session-0002/round-0003/outputs/loss.png"]
    assert result.plots[0].likely_meaning == "validation loss plateau"
    assert result.decision_relevant_observations == ["validation is no longer improving"]
    assert result.image_analysis_usage == [{"session_id": 2, "used": 1, "limit": 3}]
    assert "ordinary ML training/evaluation plot" in str(calls[0]["prompt"])


def test_storage_round_trips_visual_observations(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "runtime.sqlite3")
    try:
        session = storage.create_session(RunSession(status="running"))
        record = storage.create_round(
            RoundRecord(
                session_id=session.id,
                round_index=1,
                resolved_command="train",
                status="completed",
                visual_observations=VisualAnalysisResult(
                    status="completed",
                    image_paths=["plots/loss.png"],
                    overall_visual_summary="validation loss plateau",
                ),
            )
        )

        loaded = storage.get_round(record.id)

        assert loaded is not None
        assert loaded.visual_observations is not None
        assert loaded.visual_observations.overall_visual_summary == "validation loss plateau"
    finally:
        storage.close()


def _run(coro):
    import asyncio

    return asyncio.run(coro)
