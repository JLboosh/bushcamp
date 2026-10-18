"""Web app: upload a recording, watch it being analysed, open the reports.

    GET  /                    upload page + your analysed matches
    POST /api/uploads         multipart: file (video), player (your in-game name)
    GET  /api/jobs            all jobs, newest first
    GET  /api/jobs/{id}       one job (status, progress, message, matches)
    DELETE /api/jobs/{id}     delete a job, its recording and its reports
    GET  /files/...           reports and recordings (video served in byte ranges)

Run with `python -m server` (starts this and the worker).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server import jobs

VIDEO_TYPES = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}
MAX_BYTES = 4 * 1024 ** 3
STATIC = Path(__file__).resolve().parent / "static"
PLAYER_FILE = jobs.ROOT / "data" / "player.json"


def create_app(storage: Path = jobs.STORAGE, db: Path | None = None) -> FastAPI:
    app = FastAPI(title="Brawl positioning analyzer")
    (storage / "uploads").mkdir(parents=True, exist_ok=True)
    (storage / "results").mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/config")
    def config():
        name = json.loads(PLAYER_FILE.read_text()).get("name") if PLAYER_FILE.exists() else None
        return {"player": name, "retention_days": jobs.RETENTION_DAYS}

    @app.post("/api/uploads")
    async def upload(file: UploadFile = File(...), player: str = Form("")):
        ext = Path(file.filename or "").suffix.lower()
        if ext not in VIDEO_TYPES:
            raise HTTPException(400, f"Please upload a video ({', '.join(sorted(VIDEO_TYPES))}).")
        job_id = jobs.new_id()
        folder = storage / "uploads" / job_id
        folder.mkdir(parents=True)
        dest = folder / f"video{ext}"
        size = 0
        with dest.open("wb") as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_BYTES:
                    out.close()
                    jobs.delete(job_id, storage, db)
                    raise HTTPException(413, "That file is too large (over 4 GB).")
                out.write(chunk)
        name = re.sub(r"[^\w.\- ]", "_", file.filename or "recording")[:120]
        jobs.create(job_id, name, dest, player.strip() or None, db)
        return {"id": job_id}

    @app.get("/api/jobs")
    def list_jobs():
        return [_public(j, storage) for j in jobs.all_jobs(db)]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        j = jobs.get(job_id, db)
        if j is None:
            raise HTTPException(404, "No such job.")
        return _public(j, storage)

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        j = jobs.get(job_id, db)
        if j is None:
            raise HTTPException(404, "No such job.")
        if j["status"] == "running":
            raise HTTPException(409, "This recording is being analysed right now; try again when it's done.")
        jobs.delete(job_id, storage, db)
        return {"deleted": job_id}

    app.mount("/files", StaticFiles(directory=storage), name="files")
    return app


def _public(j: dict, storage: Path) -> dict:
    """What the page needs; never other players' names (the summaries already use brawlers only)."""
    return dict(id=j["id"], created=j["created"], filename=j["filename"], status=j["status"],
                progress=j["progress"], message=j["message"], error=j["error"],
                video_kept=(storage / "uploads" / j["id"]).exists(),
                matches=[dict(title=m["title"], map=m["map_name"], brawler=m["brawler"], footage=m["footage"],
                              summary=m["summary"], report=m["report"], me_found=m["me_found"])
                         for m in (j["result"] or [])])


app = create_app()
