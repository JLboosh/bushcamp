import csv

import cv2
import numpy as np
import pytest

from pipeline.sampling import dhash, extract_training_frames, is_blank, iter_frames, video_info

FPS = 30
W, H = 640, 360


def _texture(seed=0, w=2000, h=H):
    """Smooth random blobs: stands in for map scenery the camera pans over."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    return cv2.GaussianBlur(cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC), (0, 0), 3)


def _pan(world, t):
    x = int(t * 200)  # 200 px/s
    return world[:, x:x + W].copy()


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """10 s synthetic replay recording.

    0-3 s  static scene
    3-4 s  black (fade / loading)
    4-8 s  camera pans across the map
    8-10 s rewind: the 4-6 s section plays again
    """
    path = tmp_path_factory.mktemp("vid") / "match1.mp4"
    world = _texture()
    still = _texture(seed=1, w=W)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(10 * FPS):
        t = i / FPS
        if t < 3:
            frame = still
        elif t < 4:
            frame = np.zeros((H, W, 3), np.uint8)
        elif t < 8:
            frame = _pan(world, t - 4)
        else:
            frame = _pan(world, t - 8)
        writer.write(frame)
    writer.release()
    return path


def test_video_info(video):
    info = video_info(video)
    assert (info["width"], info["height"]) == (W, H)
    assert info["frame_count"] == 10 * FPS
    assert info["duration"] == pytest.approx(10, abs=0.1)


def test_iter_frames_rate(video):
    frames = list(iter_frames(video, fps=10))
    assert len(frames) == 100
    times = np.array([f.video_time for f in frames])
    assert np.allclose(np.diff(times), 0.1, atol=0.02)
    assert [f.index for f in frames[:4]] == [0, 3, 6, 9]


def test_iter_frames_slower_than_source(video):
    frames = list(iter_frames(video, fps=1))
    assert [round(f.video_time) for f in frames] == list(range(10))


def test_blank_and_hash():
    black = np.zeros((H, W, 3), np.uint8)
    assert is_blank(black)
    assert not is_blank(_texture(w=W))
    a = _texture(w=W)
    assert np.count_nonzero(dhash(a) != dhash(a)) == 0
    assert np.count_nonzero(dhash(a) != dhash(_texture(seed=5, w=W))) > 60


def test_extract_training_frames(video, tmp_path):
    res = extract_training_frames(video, tmp_path, interval=0.5)
    with res.manifest.open() as f:
        rows = list(csv.DictReader(f))
    by_time = {float(r["video_time"]): r for r in rows}

    # static section: first frame kept, the rest are duplicates
    assert by_time[0.0]["kept"] == "1"
    assert all(by_time[t]["reason"] == "duplicate" for t in (0.5, 1.0, 1.5, 2.0, 2.5))
    # black section
    assert by_time[3.0]["reason"] == by_time[3.5]["reason"] == "blank"
    # panning: every sample is new
    assert all(by_time[t]["kept"] == "1" for t in np.arange(4.0, 8.0, 0.5))
    # rewind repeats 4-6 s: duplicates of frames already kept
    assert all(by_time[t]["reason"] == "duplicate" for t in np.arange(8.0, 10.0, 0.5))

    assert len(res.kept) == 1 + 8
    assert all(p.exists() and p.parent.name == "match1" for p in res.kept)
    assert res.kept[0].name == "match1_t0000000.jpg"
    assert res.contact_sheet.exists()


def test_shared_match_id_and_resize(video, tmp_path):
    res = extract_training_frames(video, tmp_path, match_id="replay42", interval=2.0,
                                  max_side=320, contact_sheet=False)
    assert all(p.parent.name == "replay42" for p in res.kept)
    assert cv2.imread(str(res.kept[0])).shape[:2] == (180, 320)
    assert res.contact_sheet is None


def test_dedupe_disabled(video, tmp_path):
    res = extract_training_frames(video, tmp_path, interval=0.5, min_hash_distance=0,
                                  contact_sheet=False)
    assert len(res.kept) == 20 - 2  # everything except the 2 blank samples
