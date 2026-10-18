"""Frame extraction from replay recordings.

Two uses:
- Analysis: every frame at a fixed rate (e.g. 10 fps) via `iter_frames()`.
- Training data: sparse frames (e.g. one per second) with blank and
  near-duplicate frames dropped, via `extract_training_frames()` or the CLI.

All times here are *video* time. Match time comes later from the on-screen
timer (pipeline/timer.py); a rewound or fast-forwarded replay makes the two differ.

CLI:
    python -m pipeline.sampling VIDEO [VIDEO ...] --out data/frames --interval 1.0
"""
from __future__ import annotations

import argparse
import contextlib
import contextvars
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


# A processing window and a progress hook that every pass over the video
# (iter_frames) follows, so a long recording can be analysed one match at a
# time without cutting the file, and a server can show progress.
_WINDOW: contextvars.ContextVar[tuple[float, float] | None] = contextvars.ContextVar("window", default=None)
_PROGRESS: contextvars.ContextVar = contextvars.ContextVar("progress", default=None)


@contextlib.contextmanager
def window(start: float | None, end: float | None):
    """Within this block, iter_frames only yields frames with start <= video time <= end."""
    token = _WINDOW.set(None if start is None and end is None else (start or 0.0, end if end is not None else float("inf")))
    try:
        yield
    finally:
        _WINDOW.reset(token)


@contextlib.contextmanager
def progress_hook(callback):
    """callback(fraction) is called about once a second of video during every pass."""
    token = _PROGRESS.set(callback)
    try:
        yield
    finally:
        _PROGRESS.reset(token)


@dataclass
class Frame:
    index: int          # frame index in the source video
    video_time: float   # seconds from the start of the video
    image: np.ndarray   # BGR


def video_info(path: str | Path) -> dict:
    cap = _open(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {
        "fps": fps,
        "frame_count": n,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration": n / fps if fps else None,
    }
    cap.release()
    return info


def iter_frames(path: str | Path, fps: float) -> Iterator[Frame]:
    """Yield frames at roughly `fps` frames per second of video time.

    Reads sequentially and picks frames by their timestamps rather than by
    seeking: seeking is unreliable for some codecs, and phone screen
    recordings are often variable frame rate, so frame_index / fps would drift.
    """
    cap = _open(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) / src_fps) if src_fps else 0.0
    step = 1.0 / fps
    win = _WINDOW.get()
    report = _PROGRESS.get()
    start, end = win if win else (0.0, float("inf"))
    span = max(1e-6, min(end, duration or end) - start)
    next_t = start
    next_report = start
    index = -1
    if start > 1.0:                   # jump close to the window, then read on sequentially
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, (start - 1.0) * 1000))
        index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    try:
        while cap.grab():
            index += 1
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t <= 0 and index > 0:  # some backends don't report timestamps
                t = index / src_fps
            if t > end:
                break
            if report and t >= next_report:
                report(min(1.0, max(0.0, (t - start) / span)))
                next_report = t + 1.0
            # Small tolerance so 30 fps footage sampled at 10 fps hits every 3rd frame.
            if t + 1e-3 < next_t:
                continue
            ok, image = cap.retrieve()
            if not ok:
                continue
            yield Frame(index, t, image)
            # Advance from the schedule, not from t, so sampling doesn't drift;
            # skip ahead if the video jumped (e.g. a gap in a VFR recording).
            next_t += step
            if next_t <= t:
                next_t = t + step
    finally:
        cap.release()


# --- frame filters ---------------------------------------------------------

def dhash(image: np.ndarray, size: int = 16) -> np.ndarray:
    """Difference hash: size*size bits, robust to compression noise and brightness."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).ravel()


def is_blank(image: np.ndarray, max_mean: float = 15.0, max_std: float = 6.0) -> bool:
    """Near-black or near-uniform frames (fades, loading screens, recording start/end)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
    return small.mean() < max_mean or small.std() < max_std


# --- training-data extraction ----------------------------------------------

@dataclass
class ExtractResult:
    kept: list[Path]
    manifest: Path
    contact_sheet: Path | None


def extract_training_frames(
    video: str | Path,
    out_dir: str | Path,
    match_id: str | None = None,
    interval: float = 1.0,
    min_hash_distance: int = 12,
    max_side: int | None = None,
    jpeg_quality: int = 95,
    contact_sheet: bool = True,
) -> ExtractResult:
    """Save one frame every `interval` seconds, skipping blank and near-duplicate frames.

    A frame is a near-duplicate if its dHash is within `min_hash_distance` bits
    (out of 256) of *any* frame already kept from this video. Comparing against
    all kept frames, not just the last one, also drops repeats caused by
    rewinding the replay.

    Output goes to out_dir/<match_id>/; match_id defaults to the video name.
    Pass the same match_id for several recordings of one replay, so later
    train/val/test splits keep a match's frames together.
    """
    video = Path(video)
    match_id = match_id or video.stem
    dest = Path(out_dir) / match_id
    dest.mkdir(parents=True, exist_ok=True)

    kept: list[Path] = []
    kept_hashes: list[np.ndarray] = []
    rows = []
    for frame in iter_frames(video, fps=1.0 / interval):
        row = {"file": "", "frame_index": frame.index,
               "video_time": round(frame.video_time, 3), "kept": 0,
               "reason": "", "hash_distance": ""}
        if is_blank(frame.image):
            row["reason"] = "blank"
            rows.append(row)
            continue
        h = dhash(frame.image)
        if kept_hashes:
            dist = int(np.count_nonzero(np.stack(kept_hashes) != h, axis=1).min())
            row["hash_distance"] = dist
            if dist < min_hash_distance:
                row["reason"] = "duplicate"
                rows.append(row)
                continue

        image = _limit_size(frame.image, max_side)
        name = f"{video.stem}_t{round(frame.video_time * 1000):07d}.jpg"
        path = dest / name
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        kept.append(path)
        kept_hashes.append(h)
        row.update(file=name, kept=1)
        rows.append(row)

    manifest = dest / f"{video.stem}_manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["file"])
        writer.writeheader()
        writer.writerows(rows)

    sheet = None
    if contact_sheet and kept:
        sheet = dest / f"{video.stem}_contact_sheet.jpg"
        cv2.imwrite(str(sheet), make_contact_sheet(kept))
    return ExtractResult(kept, manifest, sheet)


def make_contact_sheet(paths: list[Path], max_tiles: int = 48, cols: int = 8,
                       tile_width: int = 240) -> np.ndarray:
    """Grid of thumbnails (evenly spread over `paths`) labeled with file names, for checking by eye."""
    if len(paths) > max_tiles:
        picks = np.linspace(0, len(paths) - 1, max_tiles).round().astype(int)
        paths = [paths[i] for i in picks]
    tiles = []
    for p in paths:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        tile = cv2.resize(img, (tile_width, round(h * tile_width / w)), interpolation=cv2.INTER_AREA)
        label = p.stem.rsplit("_t", 1)[-1]
        label = f"{int(label) / 1000:.1f}s" if label.isdigit() else p.stem
        cv2.putText(tile, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(tile, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(tile)
    # Tiles can differ in height if a match mixes aspect ratios; pad to the tallest.
    th = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


# --- helpers -----------------------------------------------------------------

def _open(path: str | Path) -> cv2.VideoCapture:
    if not Path(path).exists():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"OpenCV could not open video: {path}")
    return cap


def _limit_size(image: np.ndarray, max_side: int | None) -> np.ndarray:
    if not max_side:
        return image
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Extract de-duplicated frames from replay recordings for labeling.")
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/frames"))
    ap.add_argument("--match", help="match id shared by all given videos (default: each video's file name)")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between sampled frames")
    ap.add_argument("--min-hash-distance", type=int, default=12,
                    help="drop frames within this many dHash bits (of 256) of a kept frame; 0 disables")
    ap.add_argument("--max-side", type=int, help="downscale so the longer side is at most this many pixels")
    ap.add_argument("--no-contact-sheet", action="store_true")
    args = ap.parse_args(argv)

    for video in args.videos:
        info = video_info(video)
        print(f"{video.name}: {info['width']}x{info['height']} @ {info['fps']:.1f} fps, "
              f"{info['duration'] or 0:.1f}s")
        res = extract_training_frames(
            video, args.out, match_id=args.match, interval=args.interval,
            min_hash_distance=args.min_hash_distance, max_side=args.max_side,
            contact_sheet=not args.no_contact_sheet,
        )
        with res.manifest.open() as f:
            reasons = [r["reason"] for r in csv.DictReader(f)]
        print(f"  kept {len(res.kept)}, skipped {reasons.count('duplicate')} duplicate, "
              f"{reasons.count('blank')} blank -> {res.manifest.parent}")
        if res.contact_sheet:
            print(f"  contact sheet: {res.contact_sheet}")


if __name__ == "__main__":
    main()
