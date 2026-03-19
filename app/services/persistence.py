from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

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
        self.provider = "sqlite"
        self.enabled = True
        self.detail = "Persistencia local SQLite pendiente de inicialización."
        self._sqlite_path = SQLITE_PATH
        self._runtime_warning: str | None = None
        self._initialized = False
        self._postgres_ready = False

    def _normalize_database_url(self, url: str) -> str:
        url = url.strip()
        if not url.startswith(("postgres://", "postgresql://")):
            return url
        if "sslmode=" not in url:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}sslmode=require"
        return url

    @contextmanager
    def _connect_sqlite(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._sqlite_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @contextmanager
    def _connect_postgres(self) -> Iterator[Any]:
        if psycopg is None:
            raise RuntimeError("psycopg no está disponible.")
        last_exc = None
        for _ in range(2):
            try:
                conn = psycopg.connect(self.database_url, connect_timeout=4)
                break
            except Exception as exc:  # pragma: no cover - depende del entorno
                last_exc = exc
                time.sleep(0.25)
        else:
            raise RuntimeError(f"PostgreSQL/Neon no respondió: {last_exc}")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._init_sqlite()
        self.provider = "sqlite"
        self.enabled = True
        self.detail = f"Persistencia local SQLite habilitada en {self._sqlite_path.name}."
        db_url = self.database_url.strip()
        if db_url and db_url.startswith(("postgres://", "postgresql://")) and psycopg is not None:
            self.database_url = self._normalize_database_url(db_url)
            try:
                self._init_postgres()
                self.provider = "postgres"
                self._postgres_ready = True
                self.detail = "Persistencia principal en PostgreSQL/Neon con espejo local SQLite para recuperación de estado."
                self._runtime_warning = None
            except Exception as exc:
                self.provider = "sqlite"
                self._postgres_ready = False
                self._runtime_warning = f"PostgreSQL/Neon no quedó listo al inicializar: {exc}"
                self.detail = f"Persistencia local SQLite habilitada en {self._sqlite_path.name}. PostgreSQL/Neon quedó en modo diferido."
        self._initialized = True

    @contextmanager
    def connect(self, target: str = "auto"):
        self._ensure_initialized()
        selected = self.provider if target == "auto" else target
        if selected == "postgres":
            with self._connect_postgres() as conn:
                yield conn
        else:
            with self._connect_sqlite() as conn:
                yield conn

    def _ensure_sqlite_columns(self) -> None:
        with self._connect_sqlite() as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(job_runs)").fetchall()}
            if "meta_json" not in columns:
                conn.execute("ALTER TABLE job_runs ADD COLUMN meta_json TEXT")

    def _ensure_postgres_columns(self) -> None:
        with self._connect_postgres() as conn:
            conn.execute("ALTER TABLE job_runs ADD COLUMN IF NOT EXISTS meta_json TEXT")

    def _init_sqlite(self) -> None:
        with self._connect_sqlite() as conn:
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
                    updated_at REAL,
                    meta_json TEXT
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
        self._ensure_sqlite_columns()

    def _init_postgres(self) -> None:
        with self._connect_postgres() as conn:
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
                    updated_at DOUBLE PRECISION,
                    meta_json TEXT
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
        self._ensure_postgres_columns()

    def _params(self, data: dict[str, Any]) -> tuple[Any, ...]:
        now = float(data.get("updated_at") or time.time())
        created = float(data.get("created_at") or now)
        payload = json.dumps(data.get("result"), ensure_ascii=False) if data.get("result") is not None else None
        meta = json.dumps(data.get("meta"), ensure_ascii=False) if data.get("meta") is not None else None
        return (
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
            meta,
        )

    def _save_job_sqlite(self, data: dict[str, Any]) -> None:
        params = self._params(data)
        with self._connect_sqlite() as conn:
            conn.execute(
                """
                INSERT INTO job_runs (job_id, state, stage, progress, message, file_name, result_json, error_text, created_at, updated_at, meta_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(job_id) DO UPDATE SET
                    state=excluded.state,
                    stage=excluded.stage,
                    progress=excluded.progress,
                    message=excluded.message,
                    file_name=excluded.file_name,
                    result_json=excluded.result_json,
                    error_text=excluded.error_text,
                    updated_at=excluded.updated_at,
                    meta_json=excluded.meta_json
                """,
                params,
            )

    def _save_job_postgres(self, data: dict[str, Any]) -> None:
        params = self._params(data)
        with self._connect_postgres() as conn:
            conn.execute(
                """
                INSERT INTO job_runs (job_id, state, stage, progress, message, file_name, result_json, error_text, created_at, updated_at, meta_json)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (job_id) DO UPDATE SET
                    state=EXCLUDED.state,
                    stage=EXCLUDED.stage,
                    progress=EXCLUDED.progress,
                    message=EXCLUDED.message,
                    file_name=EXCLUDED.file_name,
                    result_json=EXCLUDED.result_json,
                    error_text=EXCLUDED.error_text,
                    updated_at=EXCLUDED.updated_at,
                    meta_json=EXCLUDED.meta_json
                """,
                params,
            )

    def save_job(self, data: dict[str, Any]) -> None:
        self._ensure_initialized()
        self._save_job_sqlite(data)
        if self._postgres_ready:
            try:
                self._save_job_postgres(data)
                self._runtime_warning = None
            except Exception as exc:  # pragma: no cover - depende del entorno
                self._runtime_warning = f"Persistencia principal degradada temporalmente: {exc}"

    def _row_to_job(self, row: Any) -> dict[str, Any] | None:
        if not row:
            return None
        result = json.loads(row[6]) if row[6] else None
        meta = json.loads(row[10]) if len(row) > 10 and row[10] else None
        return {
            "job_id": row[0],
            "state": row[1],
            "stage": row[2],
            "progress": row[3],
            "message": row[4],
            "file_name": row[5],
            "result": result,
            "error": row[7],
            "created_at": row[8],
            "updated_at": row[9],
            "meta": meta,
        }

    def load_job(self, job_id: str) -> dict[str, Any] | None:
        self._ensure_initialized()
        row = None
        if self._postgres_ready:
            try:
                with self._connect_postgres() as conn:
                    row = conn.execute(
                        "SELECT job_id,state,stage,progress,message,file_name,result_json,error_text,created_at,updated_at,meta_json FROM job_runs WHERE job_id=%s",
                        (job_id,),
                    ).fetchone()
                self._runtime_warning = None
            except Exception as exc:  # pragma: no cover - depende del entorno
                self._runtime_warning = f"Persistencia principal degradada temporalmente: {exc}"
        if row is None:
            with self._connect_sqlite() as conn:
                row = conn.execute(
                    "SELECT job_id,state,stage,progress,message,file_name,result_json,error_text,created_at,updated_at,meta_json FROM job_runs WHERE job_id=?",
                    (job_id,),
                ).fetchone()
        return self._row_to_job(row)

    def save_revision(self, job_id: str, payload: dict[str, Any]) -> None:
        self._ensure_initialized()
        raw = json.dumps(payload, ensure_ascii=False)
        now = time.time()
        with self._connect_sqlite() as conn:
            conn.execute(
                """
                INSERT INTO revision_states (job_id, payload_json, updated_at)
                VALUES (?,?,?)
                ON CONFLICT(job_id) DO UPDATE SET payload_json=excluded.payload_json, updated_at=excluded.updated_at
                """,
                (job_id, raw, now),
            )
        if self._postgres_ready:
            try:
                with self._connect_postgres() as conn:
                    conn.execute(
                        """
                        INSERT INTO revision_states (job_id, payload_json, updated_at)
                        VALUES (%s,%s,%s)
                        ON CONFLICT (job_id) DO UPDATE SET payload_json=EXCLUDED.payload_json, updated_at=EXCLUDED.updated_at
                        """,
                        (job_id, raw, now),
                    )
            except Exception as exc:  # pragma: no cover
                self._runtime_warning = f"Persistencia principal degradada temporalmente: {exc}"

    def load_revision(self, job_id: str) -> dict[str, Any] | None:
        self._ensure_initialized()
        row = None
        if self._postgres_ready:
            try:
                with self._connect_postgres() as conn:
                    row = conn.execute("SELECT payload_json FROM revision_states WHERE job_id=%s", (job_id,)).fetchone()
            except Exception as exc:  # pragma: no cover
                self._runtime_warning = f"Persistencia principal degradada temporalmente: {exc}"
        if row is None:
            with self._connect_sqlite() as conn:
                row = conn.execute("SELECT payload_json FROM revision_states WHERE job_id=?", (job_id,)).fetchone()
        if not row or not row[0]:
            return None
        return json.loads(row[0])

    def stats(self) -> dict[str, Any]:
        self._ensure_initialized()
        try:
            with self._connect_sqlite() as sqlite_conn:
                sqlite_jobs = sqlite_conn.execute("SELECT COUNT(*) FROM job_runs").fetchone()[0]
            postgres_jobs = None
            if self._postgres_ready:
                try:
                    with self._connect_postgres() as conn:
                        postgres_jobs = conn.execute("SELECT COUNT(*) FROM job_runs").fetchone()[0]
                except Exception as exc:  # pragma: no cover
                    self._runtime_warning = f"Persistencia principal degradada temporalmente: {exc}"
            return {
                "habilitada": self.enabled,
                "proveedor": self.provider,
                "detalle": self.detail,
                "advertencia_runtime": self._runtime_warning,
                "tareas_registradas": int(postgres_jobs if postgres_jobs is not None else sqlite_jobs),
                "tareas_espejo_sqlite": int(sqlite_jobs),
            }
        except Exception as exc:
            return {
                "habilitada": False,
                "proveedor": self.provider,
                "detalle": f"Persistencia no disponible: {exc}",
                "advertencia_runtime": self._runtime_warning,
                "tareas_registradas": 0,
                "tareas_espejo_sqlite": 0,
            }


persistence = Persistence()
