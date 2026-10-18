"""Brawler detection with the trained YOLO model, plus tracking and per-box details.

For every sampled frame: boxes (ally/enemy) with a tracker id (ByteTrack),
and for each box the health bar found inside it (pipeline/color_bootstrap):
- HP fraction: how full the bar is
- name tag: the clearest few crops per tracker id are kept for reading the
  name (pipeline/track.py), plus a small fingerprint of the name's text
  pixels for boxes whose name can't be read. A player's name looks the same
  all match, which links tracker ids that break whenever someone leaves the
  screen. (Bar width turned out not to identify the recording player: some
  bars carry a badge that shifts them.)

    python -m pipeline.detect data/raw/match.mp4 --weights runs/detect/brawlers_v1/weights/best.pt
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from pipeline.color_bootstrap import NAME_TEXT, find_bars
from pipeline.sampling import iter_frames

DEFAULT_WEIGHTS = Path("runs/detect/brawlers_v1/weights/best.pt")
CLASSES = ["ally", "enemy"]
NAME_SIZE = (48, 8)          # fingerprint size (w, h)
CROP_SIZE = (210, 18)        # stored name-tag crops (w, h), for OCR
# Clearest name crops kept per tracker id, team and 5 s stretch: spread over
# time because a tracker id can switch to another brawler and gets split later.
CROPS_PER_CHUNK = 4
CHUNK_FRAMES = 50


@dataclass
class Detection:
    frame: int               # index among the sampled frames
    video_time: float
    track_id: int            # ByteTrack id, -1 if the tracker didn't assign one
    team: str                # ally | enemy
    conf: float
    x0: float
    y0: float
    x1: float
    y1: float
    hp: float | None         # health-bar fill fraction (None if no bar found in the box)

    @property
    def feet(self) -> tuple[float, float]:
        """Bottom-centre of the box: where the brawler stands."""
        return (self.x0 + self.x1) / 2, self.y1


def box_details(image: np.ndarray, box: tuple[float, float, float, float], team: str):
    """(HP fraction, name fingerprint, name crop, text share) for one box; Nones if no bar found."""
    h_img, w_img = image.shape[:2]
    s = h_img / 720
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    # The bar sits in the top part of the box; widen the crop for long names.
    cx0, cx1 = max(0, x0 - int(40 * s)), min(w_img, x1 + int(40 * s))
    cy0, cy1 = max(0, y0 - int(10 * s)), min(h_img, y0 + int((y1 - y0) * 0.6))
    if cx1 - cx0 < 20 or cy1 - cy0 < 20:
        return None, None, None, 0.0
    bars = [b for b in find_bars(image[cy0:cy1, cx0:cx1], scale=s, skip_ui=False) if b.cls == team]
    if not bars:
        return None, None, None, 0.0
    bar = min(bars, key=lambda b: abs(cx0 + b.x + b.w / 2 - (x0 + x1) / 2))
    bx, by = cx0 + bar.x, cy0 + bar.y
    # Name tag: the text band above the HP digits, centred on the bar; names
    # can be much wider than the bar (e.g. LOUIHBOOSHES).
    mid = bx + bar.w / 2
    nx0, nx1 = int(mid - 105 * s), int(mid + 105 * s)
    ny0, ny1 = int(by - 2.9 * bar.h), int(by - 1.15 * bar.h)
    if nx0 < 0 or nx1 > w_img or ny0 < 0 or ny1 <= ny0:
        return bar.fill, None, None, 0.0
    crop = image[ny0:ny1, nx0:nx1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    text = ((hsv[..., 1] <= NAME_TEXT["s_max"]) & (hsv[..., 2] >= NAME_TEXT["v_min"])).astype(np.float32)
    share = float(text.mean())
    if share < 0.03:
        return bar.fill, None, None, share
    fp = cv2.resize(text[:, text.shape[1] // 4: 3 * text.shape[1] // 4], NAME_SIZE, interpolation=cv2.INTER_AREA)
    # Trim to this player's name: the run of text columns over the bar's centre
    # (gaps up to a letter space), so a neighbour's tag beside it is left out.
    a, b = _text_run(text.mean(axis=0) > 0.05, centre=crop.shape[1] // 2, max_gap=int(9 * s))
    if b - a < 6 * s:
        return bar.fill, fp, None, share
    a, b = max(0, a - int(4 * s)), min(crop.shape[1], b + int(4 * s))
    name = crop[:, a:b]
    # Pad to the stored size without stretching (OCR reads undistorted text best).
    h = CROP_SIZE[1]
    w = min(CROP_SIZE[0], max(1, round(name.shape[1] * h / name.shape[0])))
    name = cv2.resize(name, (w, h), interpolation=cv2.INTER_AREA)
    canvas = np.empty((h, CROP_SIZE[0], 3), np.uint8)
    canvas[:] = np.median(crop.reshape(-1, 3), axis=0)
    x = (CROP_SIZE[0] - w) // 2
    canvas[:, x:x + w] = name
    return bar.fill, fp, canvas, share


def _text_run(cols: np.ndarray, centre: int, max_gap: int) -> tuple[int, int]:
    """Start/end (end exclusive) of the text run nearest the centre, joining gaps up to max_gap."""
    on = np.flatnonzero(cols)
    if not len(on):
        return 0, 0
    runs, start, prev = [], on[0], on[0]
    for c in on[1:]:
        if c - prev > max_gap:
            runs.append((start, prev + 1))
            start = c
        prev = c
    runs.append((start, prev + 1))
    return min(runs, key=lambda r: 0 if r[0] <= centre < r[1] else min(abs(r[0] - centre), abs(r[1] - centre)))


def detect_video(video: Path, weights: Path = DEFAULT_WEIGHTS, fps: float = 10.0, conf: float = 0.25,
                 imgsz: int = 960) -> tuple[list[Detection], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Detections for every sampled frame, plus name fingerprints and name-tag crops,
    both keyed by detection index."""
    from ultralytics import YOLO
    import torch
    device = "0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(weights))
    dets: list[Detection] = []
    names: dict[int, np.ndarray] = {}
    best_crops: dict[tuple, list[tuple[float, int, np.ndarray]]] = {}   # (id, team, chunk) -> [(text share, det, crop)]
    for k, f in enumerate(iter_frames(video, fps=fps)):
        r = model.track(f.image, persist=True, tracker="bytetrack.yaml", conf=conf, imgsz=imgsz,
                        verbose=False, device=device)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [-1] * len(r.boxes)
        for box, c, p, tid in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist(), ids):
            team = CLASSES[c]
            hp, fp, crop, share = box_details(f.image, box, team)
            i = len(dets)
            if fp is not None:
                names[i] = fp
            if crop is not None and tid >= 0:
                kept = best_crops.setdefault((tid, team, k // CHUNK_FRAMES), [])
                kept.append((share, i, crop))
                kept.sort(key=lambda c: -c[0])
                del kept[CROPS_PER_CHUNK:]
            dets.append(Detection(k, round(f.video_time, 4), tid, team, round(p, 3), *(round(v, 1) for v in box),
                                  None if hp is None else round(hp, 2)))
    crops = {i: crop for kept in best_crops.values() for _, i, crop in kept}
    return dets, names, crops


def save(out: Path, dets: list[Detection], names: dict[int, np.ndarray], crops: dict[int, np.ndarray]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "detections.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(dets[0])) if dets else ["frame"])
        w.writeheader()
        w.writerows(asdict(d) for d in dets)
    np.savez_compressed(out / "names.npz",
                        idx=np.array(list(names), int),
                        fp=np.array(list(names.values()), np.float32).reshape(-1, NAME_SIZE[1], NAME_SIZE[0]),
                        crop_idx=np.array(list(crops), int),
                        crop=np.array(list(crops.values()), np.uint8).reshape(-1, CROP_SIZE[1], CROP_SIZE[0], 3))


def load(out: Path) -> tuple[list[Detection], dict[int, np.ndarray], dict[int, np.ndarray]]:
    def num(v):
        return None if v in ("", "None") else float(v)
    dets = []
    with (out / "detections.csv").open() as f:
        for r in csv.DictReader(f):
            dets.append(Detection(int(r["frame"]), float(r["video_time"]), int(r["track_id"]), r["team"],
                                  float(r["conf"]), float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"]),
                                  num(r["hp"])))
    z = np.load(out / "names.npz")
    names = {int(i): fp for i, fp in zip(z["idx"], z["fp"])}
    crops = {int(i): c for i, c in zip(z["crop_idx"], z["crop"])}
    return dets, names, crops


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Detect and track brawlers in a replay recording.")
    ap.add_argument("video", type=Path)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=Path("runs/match"))
    args = ap.parse_args(argv)
    dets, names, crops = detect_video(args.video, args.weights, args.fps)
    save(args.out / args.video.stem, dets, names, crops)
    ids = {d.track_id for d in dets}
    print(f"{len(dets)} detections, {len(ids)} tracker ids, {len(names)} name fingerprints, "
          f"{len(crops)} name crops -> {args.out / args.video.stem}")


if __name__ == "__main__":
    main()
