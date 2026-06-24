from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trainee.app import build_app
from trainee.settings import Settings


def _write_fake_project(project_root: Path) -> None:
    (project_root / "logs").mkdir(parents=True, exist_ok=True)
    (project_root / "data").mkdir(parents=True, exist_ok=True)
    (project_root / "README.md").write_text(
        "# Fake Trainer\n\nThis fixture simulates an external training project with a simple CLI training entrypoint.\n",
        encoding="utf-8",
    )
    (project_root / "train.py").write_text(
        """
import argparse
import json
import math
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--log-file", required=True)
parser.add_argument("--sleep", type=float, default=0.15)
parser.add_argument("--stall", action="store_true")
args = parser.parse_args()

log_file = Path(args.log_file)
log_file.parent.mkdir(parents=True, exist_ok=True)
wandb_dir = log_file.parent / "wandb" / "run-test" / "files"
wandb_dir.mkdir(parents=True, exist_ok=True)
wandb_url = "https://wandb.ai/fake/project/runs/test-run"

for step in range(args.epochs):
    if args.stall and step == 0:
        time.sleep(1.0)
    loss = round(1.0 / (step + 1) + args.lr, 4)
    line = f"step={step} total_loss={loss} loss={loss}\\n"
    print(line, end="", flush=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(line)
    time.sleep(args.sleep)

print(wandb_url, flush=True)
summary = {"total_loss": loss, "loss": loss}
(wandb_dir / "wandb-summary.json").write_text(json.dumps(summary), encoding="utf-8")
""".strip()
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture()
def runtime_env(tmp_path: Path) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    external_project = tmp_path / "external-project"
    _write_fake_project(external_project)

    data_dir = tmp_path / "runtime-data"
    settings = Settings(
        repo_root=repo_root,
        project_root=None,
        data_dir=data_dir,
        database_path=data_dir / "runtime.sqlite3",
        artifacts_dir=data_dir / "artifacts",
        template_dir=repo_root / "src" / "trainee" / "templates",
        static_dir=repo_root / "src" / "trainee" / "static",
        config_path=data_dir / "config.json",
        llm_provider="none",
        llm_timeout_sec=5.0,
        openai_api_key=None,
        openai_base_url="https://api.openai.com/v1",
        openai_model="gpt-4o-mini",
        anthropic_api_key=None,
        anthropic_base_url="https://api.anthropic.com",
        anthropic_model="claude-3-5-haiku-latest",
        anthropic_version="2023-06-01",
        anthropic_max_tokens=1024,
    )
    app = build_app(settings)
    client = TestClient(app)
    with client:
        yield {
            "client": client,
            "external_project": external_project,
            "python": sys.executable,
            "data_dir": data_dir,
        }


@pytest.fixture()
def wait_for():
    def _wait_for(condition, timeout: float = 8.0, interval: float = 0.1) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if condition():
                return
            time.sleep(interval)
        raise AssertionError("condition was not satisfied before timeout")

    return _wait_for
