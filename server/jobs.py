"""Job queue in SQLite: one row per uploaded recording.

status: queued -> running -> done | failed
Files live under storage/:
    uploads/<job>/video.<ext>      the recording (deleted after RETENTION_DAYS)
    results/<job>/match<N>/        pipeline output + report.html per match
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORAGE = ROOT / "storage"
DB = STORAGE / "jobs.sqlite"
RETENTION_DAYS = 30          # recordings older than this are deleted; reports are kept

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    created REAL NOT NULL,
    filename TEXT NOT NULL,
    video TEXT NOT NULL,
    player TEXT,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    error TEXT,
    result TEXT,
    finished REAL
);
"""


@contextmanager
def connect(db: Path | None = None):
    db = db or DB
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db, timeout=30, isolation_level=None)   # autocommit
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    try:
        yield con
    finally:
        con.close()


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def create(job_id: str, filename: str, video: Path, player: str | None, db: Path | None = None) -> None:
    with connect(db) as con:
        con.execute("INSERT INTO jobs (id, created, filename, video, player, status, message) "
                    "VALUES (?, ?, ?, ?, ?, 'queued', 'Waiting to start')",
                    (job_id, time.time(), filename, str(video), player))


def claim_next(db: Path | None = None) -> dict | None:
    """Atomically take the oldest queued job."""
    with connect(db) as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute("SELECT * FROM jobs WHERE status = 'queued' ORDER BY created LIMIT 1").fetchone()
        if row is None:
            con.execute("COMMIT")
            return None
        con.execute("UPDATE jobs SET status = 'running', message = 'Starting' WHERE id = ?", (row["id"],))
        con.execute("COMMIT")
        return dict(row)


def update(job_id: str, db: Path | None = None, **fields) -> None:
    if "result" in fields and not isinstance(fields["result"], str):
        fields["result"] = json.dumps(fields["result"], ensure_ascii=False)
    cols = ", ".join(f"{k} = ?" for k in fields)
    with connect(db) as con:
        con.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))


def get(job_id: str, db: Path | None = None) -> dict | None:
    with connect(db) as con:
        row = con.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _decode(row) if row else None


def all_jobs(db: Path | None = None) -> list[dict]:
    with connect(db) as con:
        rows = con.execute("SELECT * FROM jobs ORDER BY created DESC").fetchall()
    return [_decode(r) for r in rows]


def requeue_stale(db: Path | None = None) -> int:
    """Jobs left 'running' by a worker that stopped (crash, Ctrl+C) start over."""
    with connect(db) as con:
        return con.execute("UPDATE jobs SET status = 'queued', progress = 0, message = 'Restarting' "
                           "WHERE status = 'running'").rowcount


def delete(job_id: str, storage: Path | None = None, db: Path | None = None) -> None:
    storage = storage or STORAGE
    for sub in ("uploads", "results"):
        shutil.rmtree(storage / sub / job_id, ignore_errors=True)
    with connect(db) as con:
        con.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


def expire_videos(days: float = RETENTION_DAYS, storage: Path | None = None, db: Path | None = None) -> int:
    """Delete recordings older than `days`; their reports stay (the map replay works without video)."""
    storage = storage or STORAGE
    cutoff = time.time() - days * 86400
    n = 0
    for job in all_jobs(db):
        if job["created"] < cutoff and job["status"] in ("done", "failed"):
            folder = storage / "uploads" / job["id"]
            if folder.exists():
                shutil.rmtree(folder, ignore_errors=True)
                n += 1
    return n


def _decode(row) -> dict:
    d = dict(row)
    d["result"] = json.loads(d["result"]) if d.get("result") else None
    return d
