from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from trainee.decision import DecisionResult
from trainee.ledger import LedgerExporter
from trainee.models import AgentDecision, MetricSpec, ProjectContext, ProjectSpec, RoundRecord, RunSession, TunableParam
from trainee.prompt_assembler import PromptAssembler
from trainee.prompt_documents import PromptDocumentLoader
from trainee.reporter import ReportGenerator
from trainee.research_state import ResearchStateBuilder
from trainee.storage import Storage


def test_prompt_documents_ignore_legacy_program_md(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / ".trainee").mkdir(parents=True)
    (project / "context.md").write_text("root context\n", encoding="utf-8")
    (project / ".trainee" / "program.md").write_text("agent rules\n", encoding="utf-8")

    documents = PromptDocumentLoader().load(project)

    assert [(item.path, item.kind, item.priority) for item in documents] == [
        ("context.md", "project_context", 100),
    ]
    assert documents[0].digest == hashlib.sha256(b"root context\n").hexdigest()


def test_prompt_assembler_keeps_static_before_dynamic(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    context = ProjectContext(project_summary="test")
    state = ResearchStateBuilder().build(spec, [])
    assembler = PromptAssembler()

    first = assembler.assemble(spec, context, state, {"lr": 0.2}, [], "system prompt")
    second = assembler.assemble(spec, context, state, {"lr": 0.1}, [], "system prompt")

    assert first.user_prompt.startswith("<STATIC_CONTEXT>\n")
    assert "\n</STATIC_CONTEXT>\n\n<DYNAMIC_ROUND_STATE>\n" in first.user_prompt
    assert first.user_prompt.split("\n</STATIC_CONTEXT>", 1)[0] == second.user_prompt.split(
        "\n</STATIC_CONTEXT>", 1
    )[0]
    assert first.static_context_json == second.static_context_json
    assert first.dynamic_state_json["current_params"] == {"lr": 0.2}
    assert second.dynamic_state_json["current_params"] == {"lr": 0.1}


def test_research_state_tracks_baseline_and_best(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    rounds = _rounds()

    state = ResearchStateBuilder().build(spec, rounds)

    assert state.baseline_round is not None
    assert state.baseline_round.round_index == 1
    assert state.best_so_far_round is not None
    assert state.best_so_far_round.round_index == 2
    assert state.latest_round is not None
    assert state.latest_round.round_index == 3
    assert [item.round_index for item in state.recent_rounds] == [1, 2, 3]
    assert len(state.tried_change_signatures) == 2
    assert state.rejected_change_signatures == [state.latest_round.change_signature]
    assert state.recent_rounds[1].hypothesis == "Lower lr should improve loss."
    assert state.recent_rounds[1].change_summary == "Set lr from 0.2 to 0.1."


def test_agent_decision_accepts_hypothesis_fields() -> None:
    decision = AgentDecision(
        action="continue",
        next_params={"lr": 0.1},
        reason="test",
        hypothesis="smaller lr",
        change_summary="lr 0.2 -> 0.1",
        latest_round_judgement="improved",
        compare_to_baseline="-0.2",
        compare_to_best="new best",
        expected_effect="lower loss",
        avoid_repeating=["signature"],
        confidence=0.8,
    )

    assert decision.hypothesis == "smaller lr"
    assert decision.avoid_repeating == ["signature"]
    with pytest.raises(ValidationError):
        AgentDecision(action="stop", reason="bad confidence", confidence=1.1)


def test_runtime_decision_receives_research_state(runtime_env, wait_for) -> None:
    client = runtime_env["client"]
    runtime = client.app.state.runtime
    external_project = runtime_env["external_project"]
    python = runtime_env["python"]
    captured: dict[str, object] = {}

    async def fake_decide_with_prompt(**kwargs):
        captured.update(kwargs)
        return DecisionResult(
            decision=AgentDecision(
                action="stop",
                next_params=kwargs["current_params"],
                reason="captured",
            )
        )

    runtime.decision_engine.decide_with_prompt = fake_decide_with_prompt
    payload = {
        "project_root": str(external_project),
        "working_dir": str(external_project),
        "launcher_template": f"{python} {{project_root}}/train.py --log-file {{project_root}}/logs/state.log {{extra_args}}",
        "security_mode": "unsafe",
        "heartbeat_interval_sec": 0.1,
        "max_rounds": 2,
        "tunable_params": [
            {"name": "lr", "flag": "--lr", "type": "float", "default": 0.2, "min_value": 0.05, "max_value": 0.4},
        ],
        "metric_specs": [
            {
                "name": "total_loss",
                "source": "log_regex",
                "key_or_pattern": r"total_loss=(?P<value>-?\d+(?:\.\d+)?)",
                "goal": "min",
                "required": True,
            }
        ],
    }
    registration = {
        "project_root": payload["project_root"],
        "version": 1,
        "data": [],
        "launch": {
            "environment": "system",
            "command": ["python", "train.py"],
            "args": [],
        },
        "run": {
            "max_rounds": 2,
            "timeout_minutes": None,
            "fixed_args": [],
        },
        "tuning": {"params": payload["tunable_params"]},
        "metrics": {"specs": payload["metric_specs"], "prompt": ""},
        "advanced": {
            "security_mode": "unsafe",
            "working_dir": payload["working_dir"],
            "heartbeat_interval_sec": 0.1,
            "signal_sources": [],
            "log_paths": [],
            "shell_command": str(payload["launcher_template"]).replace("{extra_args}", "").strip(),
        },
    }
    assert client.post("/api/project/register", json=registration).status_code == 200
    assert client.post("/api/loop/start").status_code == 200
    wait_for(lambda: client.get("/api/loop").json()["status"] == "stopped")

    state = captured["research_state"]
    assert state.baseline_round.round_index == 1
    assert state.latest_round.primary_metric_value is not None
    assert captured["current_params"] == {"lr": 0.2}
    assert isinstance(captured["prompt_documents"], list)


def test_final_round_records_decision_and_rejected_signature(runtime_env, wait_for) -> None:
    client = runtime_env["client"]
    runtime = client.app.state.runtime
    external_project = runtime_env["external_project"]
    python = runtime_env["python"]
    decisions = [
        AgentDecision(
            action="continue",
            next_params={"lr": 0.1},
            reason="test lower lr",
            hypothesis="Lower lr may help.",
            change_summary="Set lr from 0.2 to 0.1.",
            latest_round_judgement="baseline",
        ),
        AgentDecision(
            action="continue",
            next_params={"lr": 0.05},
            reason="the final experiment was worse",
            latest_round_judgement="worse",
        ),
    ]

    async def fake_decide_with_prompt(**kwargs):
        return DecisionResult(decision=decisions.pop(0))

    runtime.decision_engine.decide_with_prompt = fake_decide_with_prompt
    registration = {
        "project_root": str(external_project),
        "version": 1,
        "data": [],
        "launch": {
            "environment": "system",
            "command": ["python", "train.py"],
            "args": [],
        },
        "run": {
            "max_rounds": 2,
            "timeout_minutes": None,
            "fixed_args": [],
        },
        "tuning": {
            "params": [
                {
                    "name": "lr",
                    "flag": "--lr",
                    "type": "float",
                    "default": 0.2,
                    "min_value": 0.05,
                    "max_value": 0.4,
                }
            ]
        },
        "metrics": {
            "specs": [
                {
                    "name": "total_loss",
                    "source": "log_regex",
                    "key_or_pattern": r"total_loss=(?P<value>-?\d+(?:\.\d+)?)",
                    "goal": "min",
                    "required": True,
                }
            ],
            "prompt": "",
        },
        "advanced": {
            "security_mode": "unsafe",
            "working_dir": str(external_project),
            "heartbeat_interval_sec": 0.1,
            "signal_sources": [],
            "log_paths": [],
            "shell_command": (
                f"{python} {{project_root}}/train.py "
                "--log-file {project_root}/logs/final-decision.log"
            ),
        },
    }

    assert client.post("/api/project/register", json=registration).status_code == 200
    assert client.post("/api/loop/start").status_code == 200
    wait_for(lambda: client.get("/api/loop").json()["status"] == "stopped")

    runs_payload = client.get("/api/runs").json()
    rounds_by_index = {item["round_index"]: item for item in runs_payload["rounds"]}
    assert set(rounds_by_index) == {1, 2}
    assert rounds_by_index[2]["agent_decision"]["latest_round_judgement"] == "worse"
    assert runs_payload["sessions"][0]["stop_reason"] == "Reached max_rounds."

    session_id = runs_payload["sessions"][0]["id"]
    state_path = (
        runtime_env["data_dir"]
        / "artifacts"
        / f"session-{session_id:04d}"
        / "research_state.json"
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["latest_round"]["latest_round_judgement"] == "worse"
    assert state["latest_round"]["change_signature"] in state["rejected_change_signatures"]


def test_ledger_exports_each_round(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    output_dir = tmp_path / "artifacts"

    paths = LedgerExporter().export(7, spec, _rounds(), output_dir)

    assert set(paths) == {"csv", "jsonl", "research_state"}
    jsonl_rows = [
        json.loads(line)
        for line in paths["jsonl"].read_text(encoding="utf-8").splitlines()
    ]
    assert len(jsonl_rows) == 3
    assert jsonl_rows[1]["hypothesis"] == "Lower lr should improve loss."
    assert jsonl_rows[1]["delta_vs_baseline"] == pytest.approx(-0.2)
    assert jsonl_rows[2]["delta_vs_best"] == pytest.approx(0.4)
    assert jsonl_rows[2]["best_so_far"] == pytest.approx(0.8)
    assert isinstance(jsonl_rows[1]["param_diff"], dict)

    with paths["csv"].open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 3
    assert json.loads(csv_rows[1]["param_diff"]) == {"lr": {"from": 0.2, "to": 0.1}}
    research_state = json.loads(paths["research_state"].read_text(encoding="utf-8"))
    assert research_state["best_so_far_round"]["round_index"] == 2


def test_resume_includes_ancestor_rounds(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "runtime.sqlite3")
    try:
        first = storage.create_session(RunSession())
        second = storage.create_session(RunSession(resumed_from=first.id))
        third = storage.create_session(RunSession(resumed_from=second.id))
        for session, round_index in ((first, 1), (second, 2), (third, 3)):
            storage.create_round(
                RoundRecord(
                    session_id=session.id,
                    round_index=round_index,
                    resolved_command="train",
                    param_values={"lr": 0.2},
                    status="completed",
                    metrics={"total_loss": 1.0 / round_index},
                )
            )

        assert [item.round_index for item in storage.list_research_rounds(third.id)] == [1, 2, 3]
        assert [
            item.round_index
            for item in storage.list_research_rounds(third.id, include_resumed_ancestors=False)
        ] == [3]
    finally:
        storage.close()


def test_resumed_report_uses_ancestor_best_round(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "runtime.sqlite3")
    try:
        spec = _spec(tmp_path)
        first = storage.create_session(
            RunSession(project_spec=spec, project_context=ProjectContext())
        )
        second = storage.create_session(
            RunSession(
                resumed_from=first.id,
                project_spec=spec,
                project_context=ProjectContext(),
            )
        )
        storage.create_round(
            RoundRecord(
                session_id=first.id,
                round_index=1,
                resolved_command="train",
                param_values={"lr": 0.2},
                status="completed",
                metrics={"total_loss": 0.5},
            )
        )
        storage.create_round(
            RoundRecord(
                session_id=second.id,
                round_index=2,
                resolved_command="train",
                param_values={"lr": 0.1},
                status="completed",
                metrics={"total_loss": 0.8},
            )
        )

        report = ReportGenerator(storage).generate_session_report(second.id)

        assert "Total Rounds**: 2" in report
        assert "Best observed round: **1**" in report
        assert "total_loss=0.5" in report
    finally:
        storage.close()


def _spec(project_root: Path) -> ProjectSpec:
    return ProjectSpec(
        project_root=str(project_root),
        working_dir=str(project_root),
        launcher_template="python train.py {extra_args}",
        tunable_params=[
            TunableParam(name="lr", flag="--lr", type="float", default=0.2, min_value=0.05, max_value=0.4),
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
    )


def _rounds() -> list[RoundRecord]:
    return [
        RoundRecord(
            id=1,
            session_id=1,
            round_index=1,
            resolved_command="train --lr 0.2",
            param_values={"lr": 0.2},
            status="completed",
            metrics={"total_loss": 1.0},
            agent_decision=AgentDecision(
                action="continue",
                next_params={"lr": 0.1},
                reason="try lower lr",
                hypothesis="Lower lr should improve loss.",
                change_summary="Set lr from 0.2 to 0.1.",
                latest_round_judgement="baseline",
            ),
        ),
        RoundRecord(
            id=2,
            session_id=1,
            round_index=2,
            resolved_command="train --lr 0.1",
            param_values={"lr": 0.1},
            status="completed",
            metrics={"total_loss": 0.8},
            agent_decision=AgentDecision(
                action="continue",
                next_params={"lr": 0.05},
                reason="try another decrease",
                hypothesis="A smaller lr may improve stability.",
                change_summary="Set lr from 0.1 to 0.05.",
                latest_round_judgement="improved",
            ),
        ),
        RoundRecord(
            id=3,
            session_id=1,
            round_index=3,
            resolved_command="train --lr 0.05",
            param_values={"lr": 0.05},
            status="completed",
            metrics={"total_loss": 1.2},
            agent_decision=AgentDecision(
                action="stop",
                next_params={"lr": 0.05},
                reason="worse",
                latest_round_judgement="worse",
            ),
        ),
    ]
