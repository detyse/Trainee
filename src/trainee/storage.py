from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from trainee.models import AgentDecision, LoopSnapshot, ProjectContext, ProjectSpec, PromptPreset, PromptPreview, RoundRecord, RunSession, utc_now


class Storage:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(str(database_path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._init_db()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def ping(self) -> bool:
        with self._lock:
            self._connection.execute("SELECT 1").fetchone()
        return True

    def _init_db(self) -> None:
        with self._lock:
            cursor = self._connection.cursor()
            cursor.executescript(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    stop_reason TEXT,
                    requested_stop INTEGER NOT NULL DEFAULT 0,
                    current_round INTEGER NOT NULL DEFAULT 0,
                    resumed_from INTEGER,
                    project_spec_json TEXT,
                    project_context_json TEXT
                );

                CREATE TABLE IF NOT EXISTS rounds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    round_index INTEGER NOT NULL,
                    resolved_command TEXT NOT NULL,
                    param_values_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    exit_code INTEGER,
                    log_paths_json TEXT NOT NULL,
                    wandb_run_url TEXT,
                    metrics_json TEXT NOT NULL,
                    agent_decision_json TEXT,
                    prompt_preview_json TEXT,
                    error TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );
                """
            )
            self._ensure_column("rounds", "prompt_preview_json", "TEXT")
            self._ensure_column("sessions", "resumed_from", "INTEGER")
            self._ensure_column("sessions", "project_spec_json", "TEXT")
            self._ensure_column("sessions", "project_context_json", "TEXT")
            self._connection.commit()

    def _ensure_column(self, table: str, column: str, column_type: str) -> None:
        columns = {row["name"] for row in self._connection.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            self._connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    def _set_setting(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value)),
            )
            self._connection.commit()

    def _get_setting(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._connection.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        if not row:
            return None
        return json.loads(row["value"])

    def save_project_spec(self, spec: ProjectSpec) -> None:
        self._set_setting("project_spec", spec.model_dump(mode="json"))

    def get_project_spec(self) -> Optional[ProjectSpec]:
        payload = self._get_setting("project_spec")
        return ProjectSpec.model_validate(payload) if payload else None

    def save_project_context(self, context: ProjectContext) -> None:
        self._set_setting("project_context", context.model_dump(mode="json"))

    def get_project_context(self) -> Optional[ProjectContext]:
        payload = self._get_setting("project_context")
        return ProjectContext.model_validate(payload) if payload else None

    def list_prompt_presets(self, project_root: Optional[str] = None) -> List[PromptPreset]:
        payload = self._get_setting("prompt_presets") or {"items": []}
        raw_items = payload.get("items", []) if isinstance(payload, dict) else payload
        presets = [PromptPreset.model_validate(item) for item in raw_items]
        if project_root is not None:
            presets = [item for item in presets if item.project_root == project_root]
        return sorted(presets, key=lambda item: (item.project_root, item.name.lower(), item.created_at))

    def get_prompt_preset(self, preset_id: str) -> Optional[PromptPreset]:
        for item in self.list_prompt_presets():
            if item.id == preset_id:
                return item
        return None

    def upsert_prompt_preset(self, preset: PromptPreset) -> PromptPreset:
        presets = self.list_prompt_presets()
        now = utc_now()
        match_index: Optional[int] = None
        for index, item in enumerate(presets):
            if item.id == preset.id or (item.project_root == preset.project_root and item.name == preset.name):
                match_index = index
                break

        if match_index is None:
            preset = preset.model_copy(update={"updated_at": now})
            presets.append(preset)
        else:
            existing = presets[match_index]
            preset = preset.model_copy(update={"id": existing.id, "created_at": existing.created_at, "updated_at": now})
            presets[match_index] = preset

        self._set_setting("prompt_presets", {"items": [item.model_dump(mode="json") for item in presets]})
        return preset

    def save_loop_snapshot(self, snapshot: LoopSnapshot) -> None:
        self._set_setting("loop_snapshot", snapshot.model_dump(mode="json"))

    def get_loop_snapshot(self) -> LoopSnapshot:
        payload = self._get_setting("loop_snapshot")
        return LoopSnapshot.model_validate(payload) if payload else LoopSnapshot()

    def create_session(self, session: RunSession) -> RunSession:
        with self._lock:
            cursor = self._connection.execute(
                """
                INSERT INTO sessions(
                    status, started_at, ended_at, stop_reason, requested_stop, current_round,
                    resumed_from, project_spec_json, project_context_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.status,
                    session.started_at,
                    session.ended_at,
                    session.stop_reason,
                    int(session.requested_stop),
                    session.current_round,
                    session.resumed_from,
                    json.dumps(session.project_spec.model_dump(mode="json")) if session.project_spec else None,
                    json.dumps(session.project_context.model_dump(mode="json")) if session.project_context else None,
                ),
            )
            self._connection.commit()
        session.id = int(cursor.lastrowid)
        return session

    def update_session(self, session: RunSession) -> RunSession:
        if session.id is None:
            raise ValueError("session.id is required for updates")
        with self._lock:
            self._connection.execute(
                """
                UPDATE sessions
                SET status = ?, ended_at = ?, stop_reason = ?, requested_stop = ?, current_round = ?,
                    resumed_from = ?, project_spec_json = ?, project_context_json = ?
                WHERE id = ?
                """,
                (
                    session.status,
                    session.ended_at,
                    session.stop_reason,
                    int(session.requested_stop),
                    session.current_round,
                    session.resumed_from,
                    json.dumps(session.project_spec.model_dump(mode="json")) if session.project_spec else None,
                    json.dumps(session.project_context.model_dump(mode="json")) if session.project_context else None,
                    session.id,
                ),
            )
            self._connection.commit()
        return session

    def get_session(self, session_id: int) -> Optional[RunSession]:
        with self._lock:
            row = self._connection.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return self._session_from_row(row) if row else None

    def get_latest_session(self) -> Optional[RunSession]:
        with self._lock:
            row = self._connection.execute("SELECT * FROM sessions ORDER BY id DESC LIMIT 1").fetchone()
        return self._session_from_row(row) if row else None

    def list_sessions(self) -> List[RunSession]:
        with self._lock:
            rows = self._connection.execute("SELECT * FROM sessions ORDER BY id DESC").fetchall()
        return [self._session_from_row(row) for row in rows]

    def create_round(self, record: RoundRecord) -> RoundRecord:
        if record.session_id is None:
            raise ValueError("round.session_id is required")
        with self._lock:
            cursor = self._connection.execute(
                """
                INSERT INTO rounds(
                    session_id, round_index, resolved_command, param_values_json, status, start_time,
                    end_time, exit_code, log_paths_json, wandb_run_url, metrics_json, agent_decision_json,
                    prompt_preview_json, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.session_id,
                    record.round_index,
                    record.resolved_command,
                    json.dumps(record.param_values),
                    record.status,
                    record.start_time,
                    record.end_time,
                    record.exit_code,
                    json.dumps(record.log_paths),
                    record.wandb_run_url,
                    json.dumps(record.metrics),
                    json.dumps(record.agent_decision.model_dump(mode="json")) if record.agent_decision else None,
                    json.dumps(record.prompt_preview.model_dump(mode="json")) if record.prompt_preview else None,
                    record.error,
                ),
            )
            self._connection.commit()
        record.id = int(cursor.lastrowid)
        return record

    def update_round(self, record: RoundRecord) -> RoundRecord:
        if record.id is None:
            raise ValueError("round.id is required for updates")
        with self._lock:
            self._connection.execute(
                """
                UPDATE rounds
                SET resolved_command = ?, param_values_json = ?, status = ?, start_time = ?, end_time = ?,
                    exit_code = ?, log_paths_json = ?, wandb_run_url = ?, metrics_json = ?, agent_decision_json = ?,
                    prompt_preview_json = ?, error = ?
                WHERE id = ?
                """,
                (
                    record.resolved_command,
                    json.dumps(record.param_values),
                    record.status,
                    record.start_time,
                    record.end_time,
                    record.exit_code,
                    json.dumps(record.log_paths),
                    record.wandb_run_url,
                    json.dumps(record.metrics),
                    json.dumps(record.agent_decision.model_dump(mode="json")) if record.agent_decision else None,
                    json.dumps(record.prompt_preview.model_dump(mode="json")) if record.prompt_preview else None,
                    record.error,
                    record.id,
                ),
            )
            self._connection.commit()
        return record

    def list_rounds(self, session_id: Optional[int] = None) -> List[RoundRecord]:
        query = "SELECT * FROM rounds"
        params: tuple[Any, ...] = ()
        if session_id is not None:
            query += " WHERE session_id = ?"
            params = (session_id,)
        query += " ORDER BY id DESC"
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return [self._round_from_row(row) for row in rows]

    def get_round(self, round_id: int) -> Optional[RoundRecord]:
        with self._lock:
            row = self._connection.execute("SELECT * FROM rounds WHERE id = ?", (round_id,)).fetchone()
        return self._round_from_row(row) if row else None

    def get_latest_round(self) -> Optional[RoundRecord]:
        with self._lock:
            row = self._connection.execute("SELECT * FROM rounds ORDER BY id DESC LIMIT 1").fetchone()
        return self._round_from_row(row) if row else None

    def get_last_completed_round(self, session_id: int) -> Optional[RoundRecord]:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM rounds
                WHERE session_id = ? AND status = 'completed'
                ORDER BY round_index DESC, id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        return self._round_from_row(row) if row else None

    def _session_from_row(self, row: sqlite3.Row) -> RunSession:
        project_spec = json.loads(row["project_spec_json"]) if row["project_spec_json"] else None
        project_context = json.loads(row["project_context_json"]) if row["project_context_json"] else None
        return RunSession(
            id=row["id"],
            status=row["status"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            stop_reason=row["stop_reason"],
            requested_stop=bool(row["requested_stop"]),
            current_round=row["current_round"],
            resumed_from=row["resumed_from"],
            project_spec=ProjectSpec.model_validate(project_spec) if project_spec else None,
            project_context=ProjectContext.model_validate(project_context) if project_context else None,
        )

    def _round_from_row(self, row: sqlite3.Row) -> RoundRecord:
        decision = json.loads(row["agent_decision_json"]) if row["agent_decision_json"] else None
        prompt_preview = json.loads(row["prompt_preview_json"]) if row["prompt_preview_json"] else None
        return RoundRecord(
            id=row["id"],
            session_id=row["session_id"],
            round_index=row["round_index"],
            resolved_command=row["resolved_command"],
            param_values=json.loads(row["param_values_json"]),
            status=row["status"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            exit_code=row["exit_code"],
            log_paths=json.loads(row["log_paths_json"]),
            wandb_run_url=row["wandb_run_url"],
            metrics=json.loads(row["metrics_json"]),
            agent_decision=AgentDecision.model_validate(decision) if decision else None,
            prompt_preview=PromptPreview.model_validate(prompt_preview) if prompt_preview else None,
            error=row["error"],
        )
