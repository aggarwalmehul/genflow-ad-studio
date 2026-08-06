import os
import logging

from google.cloud.sql.connector import Connector, IPTypes
import pg8000.dbapi

logger = logging.getLogger(__name__)

# ---- Config from environment (injected by Cloud Run / exported locally) ----
INSTANCE_CONNECTION_NAME = os.getenv("INSTANCE_CONNECTION_NAME", "")
DB_USER = os.getenv("DB_USER", "genflow")
DB_PASS = os.getenv("DB_PASS", "")
DB_NAME = os.getenv("DB_NAME", "genflow")
# Use PRIVATE IP inside the VPC if DB_PRIVATE_IP is set; PUBLIC otherwise.
DB_IP_TYPE = IPTypes.PRIVATE if os.getenv("DB_PRIVATE_IP") else IPTypes.PUBLIC

# One shared connector for the process lifetime.
_connector = Connector()

# ---- Schema ported from the SQLite version (TEXT/JSON kept as-is) ----
_SCHEMA_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        user_email TEXT DEFAULT 'unknown',
        status TEXT NOT NULL DEFAULT 'pending',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        request_json TEXT NOT NULL,
        progress_json TEXT,
        script_json TEXT,
        avatar_variants_json TEXT,
        selected_avatar TEXT,
        storyboard_results_json TEXT,
        video_results_json TEXT,
        final_video_path TEXT,
        error TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS reviews (
        job_id TEXT PRIMARY KEY,
        review_status TEXT NOT NULL DEFAULT 'pending',
        reviewed_at TEXT,
        notes TEXT,
        FOREIGN KEY (job_id) REFERENCES jobs(job_id)
    )""",
    """CREATE TABLE IF NOT EXISTS pipeline_logs (
        id BIGSERIAL PRIMARY KEY,
        job_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        level TEXT NOT NULL DEFAULT 'info',
        message TEXT NOT NULL,
        metadata_json TEXT,
        FOREIGN KEY (job_id) REFERENCES jobs(job_id)
    )""",
]


class DictRow(dict):
    """Row supporting name access (r['col']), positional (r[0]) and dict(r),
    mirroring the sqlite3.Row semantics the app relied on."""
    __slots__ = ("_values",)

    def __init__(self, columns, values):
        super().__init__(zip(columns, values))
        object.__setattr__(self, "_values", list(values))

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._values[key]
        return super().__getitem__(key)


class _Cursor:
    def __init__(self, raw):
        self._c = raw

    def _cols(self):
        return [d[0] for d in self._c.description] if self._c.description else []

    def fetchall(self):
        cols = self._cols()
        return [DictRow(cols, r) for r in self._c.fetchall()]

    def fetchone(self):
        row = self._c.fetchone()
        if row is None:
            return None
        return DictRow(self._cols(), row)

    @property
    def rowcount(self):
        return self._c.rowcount


class _Conn:
    """Wraps a pg8000 connection to mimic the app's sqlite3 usage:
      - .execute(sql, params) using qmark ('?') placeholders
      - context manager: commit on success, rollback on error, always close
    """
    def __init__(self, raw):
        self._raw = raw

    def execute(self, sql, params=()):
        cur = self._raw.cursor()
        cur.execute(sql, params)
        return _Cursor(cur)

    def executescript(self, script):
        cur = self._raw.cursor()
        for stmt in (s.strip() for s in script.split(";")):
            if not stmt or stmt.upper().startswith("PRAGMA"):
                continue
            cur.execute(stmt)
        return _Cursor(cur)

    def commit(self):
        self._raw.commit()

    def rollback(self):
        self._raw.rollback()

    def close(self):
        self._raw.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self._raw.commit()
            else:
                self._raw.rollback()
        finally:
            self._raw.close()


class Database:
    # db_path kept only for backwards-compat with the old signature; ignored.
    def __init__(self, db_path=None):
        if not INSTANCE_CONNECTION_NAME:
            raise RuntimeError(
                "INSTANCE_CONNECTION_NAME is not set - cannot connect to Cloud SQL"
            )
        # qmark => existing '?' placeholders across the app keep working unchanged
        pg8000.dbapi.paramstyle = "qmark"
        self._init_db()

    def connect(self) -> _Conn:
        raw = _connector.connect(
            INSTANCE_CONNECTION_NAME,
            "pg8000",
            user=DB_USER,
            password=DB_PASS,
            db=DB_NAME,
            ip_type=DB_IP_TYPE,
        )
        return _Conn(raw)

    def _init_db(self):
        with self.connect() as conn:
            for stmt in _SCHEMA_STATEMENTS:
                conn.execute(stmt)
        try:
            with self.connect() as conn:
                conn.execute(
                    "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS "
                    "user_email TEXT DEFAULT 'unknown'"
                )
        except Exception as e:
            logger.warning("user_email migration: %s", e)
        logger.info(
            "Cloud SQL database initialized (%s / %s)",
            INSTANCE_CONNECTION_NAME, DB_NAME,
        )
