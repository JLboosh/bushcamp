"""Background worker: takes queued jobs one at a time and runs the pipeline.

    python -m server.worker
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path

from server import jobs


def process(job: dict, storage: Path = jobs.STORAGE) -> None:
    from pipeline.run import UnsupportedFootage, analyze_video

    last = {"t": 0.0}

    def progress(frac: float, message: str) -> None:
        now = time.time()
        if now - last["t"] >= 1.0 or frac >= 1.0:          # don't hammer the database
            jobs.update(job["id"], progress=round(frac, 3), message=message)
            last["t"] = now

    try:
        results = analyze_video(Path(job["video"]), storage / "results" / job["id"], player=job["player"],
                                on_progress=progress)
        rel = [dict(r, report=f"/files/results/{job['id']}/{Path(r['folder']).name}/report.html",
                    folder=Path(r["folder"]).name) for r in results]
        jobs.update(job["id"], status="done", progress=1.0, message="Done", result=rel, finished=time.time())
    except UnsupportedFootage as e:
        jobs.update(job["id"], status="failed", message="Can't analyse this recording", error=str(e),
                    finished=time.time())
    except Exception as e:  # noqa: BLE001 - report any pipeline failure on the job
        traceback.print_exc()
        jobs.update(job["id"], status="failed", message="Something went wrong",
                    error=f"{type(e).__name__}: {e}", finished=time.time())


def main(poll_s: float = 2.0) -> None:
    n = jobs.requeue_stale()
    if n:
        print(f"worker: restarting {n} interrupted job(s)")
    removed = jobs.expire_videos()
    if removed:
        print(f"worker: deleted {removed} recording(s) older than {jobs.RETENTION_DAYS} days")
    print("worker: waiting for jobs")
    while True:
        job = jobs.claim_next()
        if job is None:
            time.sleep(poll_s)
            continue
        print(f"worker: {job['id']} ({job['filename']})")
        process(job)
        print(f"worker: {job['id']} {jobs.get(job['id'])['status']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
