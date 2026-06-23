from __future__ import annotations

from trainee.context_builder import ContextBuilder
from trainee.models import MetricSpec, ProjectSpec, TunableParam


def test_context_builder_extracts_project_shape(runtime_env):
    external_project = runtime_env["external_project"]
    spec = ProjectSpec(
        project_root=str(external_project),
        working_dir=str(external_project),
        launcher_template="python {project_root}/train.py --log-file {project_root}/logs/external.log {extra_args}",
        data_paths=[str(external_project / "data")],
        log_paths=[str(external_project / "logs" / "*.log")],
        tunable_params=[
            TunableParam(name="lr", flag="--lr", type="float", default=0.1, min_value=0.01, max_value=1.0),
            TunableParam(name="epochs", flag="--epochs", type="int", default=3, min_value=1, max_value=10),
        ],
        metric_specs=[
            MetricSpec(
                name="total_loss",
                source="log_regex",
                key_or_pattern=r"total_loss=(?P<value>-?\d+(?:\.\d+)?)",
                goal="min",
                required=True,
            )
        ],
        metric_prompt="Look at total_loss.",
        tuning_prompt="Prefer smaller lr if loss gets worse.",
    )

    context = ContextBuilder().build(spec)

    assert "Fake Trainer" in context.project_summary
    assert "train.py" in context.training_entrypoint_summary
    assert "Configured data paths" in context.data_summary
    assert "lr (--lr, float)" in context.parameter_summary
    assert "total_loss" in context.result_reading_summary
