
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import psycopg
except Exception:  # pragma: no cover
    psycopg = None

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SQLITE_PATH = DATA_DIR / "trazocad_local.db"


class Persistence:
    def __init__(self) -> None:
        self.database_url = os.getenv("DATABASE_URL", "").strip()
        self.provider = "memoria"
        self.enabled = False
        self.detail = "Sin persistencia externa. Solo memoria y archivos locales."
        self._sqlite_path = SQLITE_PATH
        self._init_provider()

    def _init_provider(self) -> None:
        if self.database_url and self.database_url.startswith(("postgres://", "postgresql://")) and psycopg is not None:
            self.provider = "postgres"
            self.enabled = True
            self.detail = "Persistencia en PostgreSQL/Neon habilitada."
            self._init_postgres()
            return
        self.provider = "sqlite"
        self.enabled = True
        self.detail = f"Persistencia local SQLite habilitada en {self._sqlite_path.name}."
        self._init_sqlite()

    @contextmanager
    def connect(self):
        if self.provider == "postgres":
            conn = psycopg.connect(self.database_url)
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()
        else:
            conn = sqlite3.connect(self._sqlite_path)
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _init_sqlite(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_runs (
                    job_id TEXT PRIMARY KEY,
                    state TEXT,
                    stage TEXT,
                    progress INTEGER,
                    message TEXT,
                    file_name TEXT,
                    result_json TEXT,
                    error_text TEXT,
                    created_at REAL,
                    updated_at REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS revision_states (
                    job_id TEXT PRIMARY KEY,
                    payload_json TEXT,
                    updated_at REAL
                )
                """
            )

    def _init_postgres(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_runs (
                    job_id TEXT PRIMARY KEY,
                    state TEXT,
                    stage TEXT,
                    progress INTEGER,
                    message TEXT,
                    file_name TEXT,
                    result_json TEXT,
                    error_text TEXT,
                    created_at DOUBLE PRECISION,
                    updated_at DOUBLE PRECISION
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS revision_states (
                    job_id TEXT PRIMARY KEY,
                    payload_json TEXT,
                    updated_at DOUBLE PRECISION
                )
                """
            )

    def save_job(self, data: dict[str, Any]) -> None:
        now = float(data.get("updated_at") or time.time())
        created = float(data.get("created_at") or now)
        payload = json.dumps(data.get("result"), ensure_ascii=False) if data.get("result") is not None else None
        params = (
            data.get("job_id"),
            data.get("state"),
            data.get("stage"),
            int(data.get("progress") or 0),
            data.get("message"),
            data.get("file_name"),
            payload,
            data.get("error"),
            created,
            now,
        )
        with self.connect() as conn:
            if self.provider == "postgres":
                conn.execute(
                    """
                    INSERT INTO job_runs (job_id, state, stage, progress, message, file_name, result_json, error_text, created_at, updated_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (job_id) DO UPDATE SET
                        state=EXCLUDED.state,
                        stage=EXCLUDED.stage,
                        progress=EXCLUDED.progress,
                        message=EXCLUDED.message,
                        file_name=EXCLUDED.file_name,
                        result_json=EXCLUDED.result_json,
                        error_text=EXCLUDED.error_text,
                        updated_at=EXCLUDED.updated_at
                    """,
                    params,
                )
            else:
                conn.execute(
                    """
                    INSERT INTO job_runs (job_id, state, stage, progress, message, file_name, result_json, error_text, created_at, updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(job_id) DO UPDATE SET
                        state=excluded.state,
                        stage=excluded.stage,
                        progress=excluded.progress,
                        message=excluded.message,
                        file_name=excluded.file_name,
                        result_json=excluded.result_json,
                        error_text=excluded.error_text,
                        updated_at=excluded.updated_at
                    """,
                    params,
                )

    def load_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            if self.provider == "postgres":
                row = conn.execute(
                    "SELECT job_id,state,stage,progress,message,file_name,result_json,error_text,created_at,updated_at FROM job_runs WHERE job_id=%s",
                    (job_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT job_id,state,stage,progress,message,file_name,result_json,error_text,created_at,updated_at FROM job_runs WHERE job_id=?",
                    (job_id,),
                ).fetchone()
        if not row:
            return None
        result = json.loads(row[6]) if row[6] else None
        return {
            "job_id": row[0], "state": row[1], "stage": row[2], "progress": row[3], "message": row[4],
            "file_name": row[5], "result": result, "error": row[7], "created_at": row[8], "updated_at": row[9]
        }

    def save_revision(self, job_id: str, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False)
        now = time.time()
        with self.connect() as conn:
            if self.provider == "postgres":
                conn.execute(
                    """
                    INSERT INTO revision_states (job_id, payload_json, updated_at)
                    VALUES (%s,%s,%s)
                    ON CONFLICT (job_id) DO UPDATE SET payload_json=EXCLUDED.payload_json, updated_at=EXCLUDED.updated_at
                    """,
                    (job_id, raw, now),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO revision_states (job_id, payload_json, updated_at)
                    VALUES (?,?,?)
                    ON CONFLICT(job_id) DO UPDATE SET payload_json=excluded.payload_json, updated_at=excluded.updated_at
                    """,
                    (job_id, raw, now),
                )

    def load_revision(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            if self.provider == "postgres":
                row = conn.execute("SELECT payload_json FROM revision_states WHERE job_id=%s", (job_id,)).fetchone()
            else:
                row = conn.execute("SELECT payload_json FROM revision_states WHERE job_id=?", (job_id,)).fetchone()
        if not row or not row[0]:
            return None
        return json.loads(row[0])

    def stats(self) -> dict[str, Any]:
        try:
            with self.connect() as conn:
                if self.provider == "postgres":
                    jobs = conn.execute("SELECT COUNT(*) FROM job_runs").fetchone()[0]
                    revs = conn.execute("SELECT COUNT(*) FROM revision_states").fetchone()[0]
                else:
                    jobs = conn.execute("SELECT COUNT(*) FROM job_runs").fetchone()[0]
                    revs = conn.execute("SELECT COUNT(*) FROM revision_states").fetchone()[0]
            return {
                "habilitada": self.enabled,
                "proveedor": self.provider,
                "detalle": self.detail,
                "tareas_registradas": int(jobs),
                "revisiones_guardadas": int(revs),
            }
        except Exception as exc:
            return {
                "habilitada": False,
                "proveedor": self.provider,
                "detalle": f"Persistencia no disponible: {exc}",
                "tareas_registradas": 0,
                "revisiones_guardadas": 0,
            }

persistence = Persistence()
