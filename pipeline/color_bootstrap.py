"""Rough brawler pre-labels from health-bar colour, for correcting in Roboflow/CVAT.

Not the real detector (that's the YOLO model trained on corrected labels).
This only has to be right often enough that fixing its boxes is faster than
drawing them from scratch.

How a health bar is found (pixel sizes are at 720p and scale with frame height):
1. Fill: a cyan (ally) or red (enemy) blob 7-12 px tall. Its left end is the
   bar's left end, because the fill drains from the right.
2. Bar span: bars are a near-fixed width (~58 px; the recording player's
   own bar is ~63), so assume that width and check the span looks like a bar:
   fill, the drained part (light greyish purple) or digits. This keeps low-HP
   bars and rejects barrels, spawn pads and effects, which aren't bar-shaped.
3. HP digits: white digits sit on top of every bar.
4. Name tag: text above the digits. Spawned objects (turrets, pets) have a
   health bar but no name, so this keeps them out.

Then the box is grown from the bar to cover the name tag above and the brawler below.

Colours measured on Opening Move replay footage (2026-10). In replays the
recording player and their teammates share the cyan bar, so classes are
ally/enemy; "me" is picked out later by name tag.

CLI:
    python -m pipeline.color_bootstrap data/frames --viz 24
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["ally", "enemy"]

# OpenCV HSV: H 0-179, S and V 0-255.
FILL = {
    "ally": dict(h=(84, 96), s_min=150, v_min=200),
    "enemy": dict(h=(165, 5), s_min=150, v_min=140),  # wraps around 180
}
EMPTY = dict(h=(100, 140), s_max=115, v=(140, 225))   # drained part of a bar
WHITE = dict(s_max=60, v_min=215)                     # HP digits
NAME_TEXT = dict(s_max=150, v_min=215)                # name tags (cyan, pink or white)
MIN_NAME_TEXT = 0.03

# Geometry at 720p, multiplied by frame_height / 720.
BAR_HEIGHT = (7, 12)     # real bars measured 8-11 px
BAR_WIDTH = 58           # other players; the recording player's own bar is ~63
MAX_FILL_WIDTH = 70
MIN_BARISH = 0.7         # share of the assumed bar span that must look bar-like
BOX_ABOVE_BAR = 30       # name tag
BOX_BELOW_BAR = 95       # body down to the feet
BOX_MIN_WIDTH = 84
BOX_WIDTH_PAD = 24

# UI that looks bar-like. Round indicator, top centre: turns cyan or red as
# rounds are won. Exit button, bottom-left: its outline catches the red threshold.
ROUND_INDICATOR = dict(max_y=50, half_width=25)
EXIT_BUTTON = dict(max_x=200, min_y=630)


@dataclass
class Bar:
    cls: str
    x: int        # left edge of the bar
    y: int        # top of the fill
    w: int        # full bar width (assumed, see BAR_WIDTH)
    h: int
    fill: float   # fraction of the bar that is coloured (≈ HP fraction)


@dataclass
class Box:
    cls: str
    x0: int
    y0: int
    x1: int
    y1: int
    bar: Bar

    def yolo(self, img_w: int, img_h: int) -> str:
        cx = (self.x0 + self.x1) / 2 / img_w
        cy = (self.y0 + self.y1) / 2 / img_h
        w = (self.x1 - self.x0) / img_w
        h = (self.y1 - self.y0) / img_h
        # Boxes touching the image edge can overshoot it by a rounding error
        # (Roboflow then reports "trimmed annotations"); shrink them to fit.
        w = min(w, 2 * cx, 2 * (1 - cx)) - 1e-6
        h = min(h, 2 * cy, 2 * (1 - cy)) - 1e-6
        return f"{CLASSES.index(self.cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _hue_in(h: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return (h >= lo) & (h <= hi) if lo <= hi else (h >= lo) | (h <= hi)


def find_bars(image: np.ndarray, scale: float | None = None, skip_ui: bool = True) -> list[Bar]:
    """Health bars in a frame. For a crop of a frame, pass the frame's `scale`
    (height / 720) and skip_ui=False (UI zones are in full-frame coordinates)."""
    s = scale or image.shape[0] / 720
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = (hsv[..., i] for i in range(3))
    empty = _hue_in(H, *EMPTY["h"]) & (S <= EMPTY["s_max"]) & (V >= EMPTY["v"][0]) & (V <= EMPTY["v"][1])
    white = (S <= WHITE["s_max"]) & (V >= WHITE["v_min"])
    name_text = (S <= NAME_TEXT["s_max"]) & (V >= NAME_TEXT["v_min"])

    bars = []
    for cls, spec in FILL.items():
        fill = _hue_in(H, *spec["h"]) & (S >= spec["s_min"]) & (V >= spec["v_min"])
        # Join fill pixels split by a digit outline crossing the bar.
        fill_u8 = cv2.morphologyEx(fill.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fill_u8)
        for i, (x, y, w, h, area) in enumerate(stats[1:], start=1):
            if not (BAR_HEIGHT[0] * s <= h <= BAR_HEIGHT[1] * s) or w > MAX_FILL_WIDTH * s:
                continue
            if skip_ui and y < ROUND_INDICATOR["max_y"] * s and \
                    abs(x - image.shape[1] / 2) < ROUND_INDICATOR["half_width"] * s:
                continue
            if skip_ui and x < EXIT_BUTTON["max_x"] * s and y > EXIT_BUTTON["min_y"] * s:
                continue
            # The digits cover the top of the fill, so test solidity on its bottom
            # rows: a bar is filled across its whole width there; effects aren't.
            bottom = labels[y + h - max(2, h // 3):y + h, x:x + w] == i
            if bottom.any(axis=0).mean() < 0.8:
                continue
            # Bars have a near-fixed width, so assume it from the left edge
            # rather than measuring it (digits printed on the bar make the
            # drained part hard to trace). Then check that span looks like a
            # bar. A column counts if a quarter of its rows are fill or drained
            # (only the drained part's top half is light), or if it has digit
            # pixels anywhere from just above the bar down.
            cap_w = max(w, int(BAR_WIDTH * s))
            cols = slice(x, x + cap_w)
            bar_cols = (fill | empty)[y:y + h, cols].mean(axis=0) >= 0.25
            digit_cols = white[max(0, y - h):y + h, cols].any(axis=0)
            if (bar_cols | digit_cols).mean() < MIN_BARISH:
                continue
            # HP digits: white pixels on or just above the bar.
            dy0 = max(0, int(y - 1.2 * h))
            digits = white[dy0:y + h, x:x + cap_w]
            if digits.mean() < 0.06:
                continue
            # Name tag above the digits: brawlers have one, spawned objects
            # (turrets, pets, etc.) have a health bar but no name.
            name = name_text[max(0, y - int(2.8 * h)):max(0, y - int(1.3 * h)),
                             max(0, x - int(12 * s)):x + cap_w + int(12 * s)]
            if name.size == 0 or name.mean() < MIN_NAME_TEXT:
                continue
            bars.append(Bar(cls, int(x), int(y), cap_w, int(h), round(float(w / cap_w), 2)))
    return _drop_overlapping(bars)


def _drop_overlapping(bars: list[Bar]) -> list[Bar]:
    """One bar can produce several fill blobs (e.g. split by digits); keep the widest."""
    bars = sorted(bars, key=lambda b: -b.w)
    kept: list[Bar] = []
    for b in bars:
        if not any(abs(b.y - k.y) < k.h and k.x - 2 <= b.x <= k.x + k.w for k in kept):
            kept.append(b)
    return kept


def bars_to_boxes(bars: list[Bar], img_w: int, img_h: int) -> list[Box]:
    s = img_h / 720
    boxes = []
    for b in bars:
        cx = b.x + b.w / 2
        half = max(b.w + BOX_WIDTH_PAD * s, BOX_MIN_WIDTH * s) / 2
        x0, x1 = max(0, cx - half), min(img_w, cx + half)
        y0, y1 = max(0, b.y - BOX_ABOVE_BAR * s), min(img_h, b.y + BOX_BELOW_BAR * s)
        boxes.append(Box(b.cls, int(x0), int(y0), int(x1), int(y1), b))
    return boxes


def detect(image: np.ndarray) -> list[Box]:
    return bars_to_boxes(find_bars(image), image.shape[1], image.shape[0])


COLORS = {"ally": (255, 220, 0), "enemy": (60, 60, 255)}


def draw(image: np.ndarray, boxes: list[Box]) -> np.ndarray:
    out = image.copy()
    for b in boxes:
        c = COLORS[b.cls]
        cv2.rectangle(out, (b.x0, b.y0), (b.x1, b.y1), c, 2)
        bar = b.bar
        cv2.rectangle(out, (bar.x, bar.y), (bar.x + bar.w, bar.y + bar.h), (255, 255, 255), 1)
        cv2.putText(out, f"{b.cls} {bar.fill:.0%}", (b.x0, max(12, b.y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return out


def prelabel_dir(frames_dir: Path, labels_dir: Path | None = None, viz: int = 0,
                 viz_dir: Path | None = None) -> dict:
    """Write a YOLO .txt next to every frame (or into labels_dir), plus classes.txt.

    Frames are searched recursively, so data/frames/<match>/... works directly.
    """
    images = sorted(p for p in frames_dir.rglob("*.jpg") if not p.stem.endswith("contact_sheet"))
    stats = {"images": 0, "ally": 0, "enemy": 0, "empty": 0}
    picks = set(np.linspace(0, len(images) - 1, viz).round().astype(int)) if viz and images else set()
    for i, path in enumerate(images):
        img = cv2.imread(str(path))
        boxes = detect(img)
        out = (labels_dir / path.relative_to(frames_dir)).with_suffix(".txt") if labels_dir \
            else path.with_suffix(".txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(b.yolo(img.shape[1], img.shape[0]) + "\n" for b in boxes))
        (out.parent / "classes.txt").write_text("\n".join(CLASSES) + "\n")
        stats["images"] += 1
        stats["empty"] += not boxes
        for b in boxes:
            stats[b.cls] += 1
        if i in picks:
            vd = viz_dir or frames_dir.parent / "prelabel_viz"
            vd.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vd / path.name), draw(img, boxes))
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Pre-label brawlers from health-bar colour (YOLO format).")
    ap.add_argument("frames", type=Path, help="folder of frames (searched recursively)")
    ap.add_argument("--labels", type=Path, help="write labels here instead of next to the frames")
    ap.add_argument("--viz", type=int, default=0, help="save this many annotated sample frames")
    ap.add_argument("--viz-dir", type=Path)
    args = ap.parse_args(argv)
    stats = prelabel_dir(args.frames, args.labels, args.viz, args.viz_dir)
    print(f"{stats['images']} images: {stats['ally']} ally, {stats['enemy']} enemy boxes; "
          f"{stats['empty']} images with no boxes")


if __name__ == "__main__":
    main()
