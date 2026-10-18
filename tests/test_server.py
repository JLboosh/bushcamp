from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import jobs
from server.app import create_app


@pytest.fixture
def client(tmp_path):
    db = tmp_path / "jobs.sqlite"
    return TestClient(create_app(storage=tmp_path, db=db)), tmp_path, db


def test_upload_creates_queued_job(client):
    c, storage, db = client
    r = c.post("/api/uploads", files={"file": ("match.mp4", b"fake video bytes", "video/mp4")},
               data={"player": "LOUIHBOOSHES"})
    assert r.status_code == 200
    job_id = r.json()["id"]
    j = c.get(f"/api/jobs/{job_id}").json()
    assert j["status"] == "queued" and j["filename"] == "match.mp4" and j["matches"] == []
    assert (storage / "uploads" / job_id / "video.mp4").read_bytes() == b"fake video bytes"
    assert jobs.get(job_id, db)["player"] == "LOUIHBOOSHES"


def test_rejects_non_video(client):
    c, _, _ = client
    r = c.post("/api/uploads", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_queue_order_and_stale_jobs(tmp_path):
    db = tmp_path / "j.sqlite"
    jobs.create("a", "a.mp4", Path("a.mp4"), None, db)
    jobs.create("b", "b.mp4", Path("b.mp4"), None, db)
    first = jobs.claim_next(db)
    assert first["id"] == "a" and jobs.get("a", db)["status"] == "running"
    assert jobs.requeue_stale(db) == 1                       # worker died mid-job
    assert jobs.claim_next(db)["id"] == "a"


def test_result_exposes_reports_and_delete(client):
    c, storage, db = client
    (storage / "uploads" / "x").mkdir(parents=True)
    jobs.create("x", "m.mp4", storage / "uploads" / "x" / "video.mp4", None, db)
    jobs.update("x", db, status="done", progress=1.0, result=[dict(
        title="Opening Move: Pierce", map_name="Opening Move", brawler="Pierce", footage="replay",
        summary=["You overextended once."], report="/files/results/x/match1/report.html", me_found=True)])
    m = c.get("/api/jobs").json()[0]["matches"][0]
    assert m["report"].endswith("report.html") and m["brawler"] == "Pierce"
    assert c.delete("/api/jobs/x").status_code == 200
    assert c.get("/api/jobs").json() == [] and not (storage / "uploads" / "x").exists()


def test_video_served_in_byte_ranges(client):
    c, storage, _ = client
    (storage / "uploads" / "v").mkdir(parents=True)
    (storage / "uploads" / "v" / "video.mp4").write_bytes(bytes(range(256)) * 4)
    r = c.get("/files/uploads/v/video.mp4", headers={"Range": "bytes=10-19"})
    assert r.status_code == 206 and r.content == bytes(range(10, 20))
