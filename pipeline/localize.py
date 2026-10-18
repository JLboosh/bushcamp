"""Map localization: where the camera is on the map, for every frame.

The replay camera is tilted but never rotates or zooms; it only slides over
the map. So the mapping from screen pixels to map tiles is

    tiles = T(camera position) · P

where P is a fixed perspective (calibrated once per screen resolution) and T
a 2D shift that changes per frame. To find the shift, a frame's bush and
water pixels are projected top-down with P and slid over the map's tile
layout (pipeline/maps.py) to where they fit best. The game's look and the
diagram differ, but where the bushes and water are does not.

Map orientation: a recording may show the map as drawn or turned 180°;
both are tried and the better one is kept for the whole recording. Knockout
maps are (nearly) symmetric under that turn, so the two fit almost equally
well and positioning is the same either way; which side is which team must
come from where the teams spawn (tracking), not from this choice.

CLI:
    python -m pipeline.localize data/raw/match.mp4 --map streets_with_no_name
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from pipeline.maps import MapRef, load
from pipeline.sampling import iter_frames

PPT = 8                       # pixels per tile in the top-down matching images
PAD = 12                      # tiles of empty margin around the layout (camera sees past the map edge)
# Bushes are tall: the game draws them up the screen from the tile they stand
# on, by an amount that depends on the map's art (map.json "bush_raise", in
# tiles; measured against something flat: Streets' water, Opening Move's floor
# squares). Correcting for it makes the fit line up the ground, where feet are.

# In-game colours (OpenCV HSV), measured on both MVP maps.
GAME_BUSH = dict(h=(89, 97), s_min=170, v=(110, 215))
GAME_WATER = dict(h=(84, 92), s=(120, 190), v_min=180)   # Streets with No Name's pools

# Screen regions that are UI, not map (fractions of width/height; mirrors the
# layout of the replay viewer seen so far).
UI_BOXES = [
    (0.00, 0.00, 0.22, 0.11),   # own team portraits, top left
    (0.78, 0.00, 1.00, 0.11),   # enemy portraits, top right
    (0.46, 0.00, 0.57, 0.11),   # round indicator
    (0.00, 0.74, 0.22, 1.00),   # skip buttons + Exit, bottom left
    (0.80, 0.74, 1.00, 1.00),   # skip buttons + follow button, bottom right
    (0.30, 0.83, 0.74, 0.92),   # progress bar
]


@dataclass
class Camera:
    """Fixed screen→ground perspective for one screen resolution.

    At screen row v, a pixel spans kx/d tiles across and kv/d tiles down,
    with d = 1 + p·(v - v0): rows lower on screen are nearer the camera.
    """
    width: int
    height: int
    kx: float
    kv: float
    p: float

    @property
    def u0(self) -> float:
        return self.width / 2

    @property
    def v0(self) -> float:
        return self.height / 2

    def matrix(self, x: float = 0.0, y: float = 0.0) -> np.ndarray:
        """3x3 homography: screen (u, v) → map tiles, with the screen centre at tile (x, y)."""
        m0 = np.array([[self.kx, 0, -self.kx * self.u0],
                       [0, self.kv, -self.kv * self.v0],
                       [0, self.p, 1 - self.p * self.v0]], float)
        return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], float) @ m0

    def to_map(self, uv: np.ndarray, x: float, y: float) -> np.ndarray:
        pts = np.asarray(uv, float).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.matrix(x, y)).reshape(-1, 2)

    def to_screen(self, xy: np.ndarray, x: float, y: float) -> np.ndarray:
        pts = np.asarray(xy, float).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, np.linalg.inv(self.matrix(x, y))).reshape(-1, 2)


def default_camera(width: int, height: int) -> Camera:
    """Starting guess, measured on 1562x720 footage (a tile ≈ 65 px wide, 41 px tall)."""
    s = height / 720
    return Camera(width, height, kx=1 / (65 * s), kv=1 / (41 * s), p=0.0)


# --- per-frame features --------------------------------------------------------

def ui_mask(height: int, width: int) -> np.ndarray:
    m = np.ones((height, width), np.uint8)
    for x0, y0, x1, y1 in UI_BOXES:
        m[int(y0 * height):int(y1 * height), int(x0 * width):int(x1 * width)] = 0
    return m


def feature_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    """Bush and water pixels (uint8 0/1) on screen, UI removed."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    b, w = GAME_BUSH, GAME_WATER
    bush = (h >= b["h"][0]) & (h <= b["h"][1]) & (s >= b["s_min"]) & (v >= b["v"][0]) & (v <= b["v"][1])
    water = (h >= w["h"][0]) & (h <= w["h"][1]) & (s >= w["s"][0]) & (s <= w["s"][1]) & (v >= w["v_min"])
    keep = ui_mask(*image.shape[:2])
    out = {}
    for name, m in (("bush", bush), ("water", water)):
        m = m.astype(np.uint8) & keep
        # Drop specks (health bars, effects): real bushes and pools are big.
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        out[name] = m
    out["valid"] = keep
    return out


# --- matching ------------------------------------------------------------------

@dataclass
class LayoutImages:
    """Top-down layout channels at PPT px/tile with PAD tiles of margin, per orientation."""
    channels: dict[str, np.ndarray]
    rotated: bool


def layout_images(ref: MapRef, rotated: bool = False) -> LayoutImages:
    ch = {"bush": ref.mask("B", PPT, PAD).astype(np.float32),
          "water": ref.mask("~", PPT, PAD).astype(np.float32)}
    if rotated:
        ch = {k: np.ascontiguousarray(v[::-1, ::-1]) for k, v in ch.items()}
    # Raise bushes up the screen (after any rotation: the raise is on screen, not on the map).
    shift = int(round(ref.bush_raise * PPT))
    bush = np.zeros_like(ch["bush"])
    bush[:-shift] = ch["bush"][shift:]
    ch["bush"] = bush
    return LayoutImages(ch, rotated)


def top_down(masks: dict[str, np.ndarray], cam: Camera) -> tuple[dict[str, np.ndarray], tuple[float, float]]:
    """Project screen masks onto the ground (camera centre at tile 0,0).

    Returns the projected channels and the tile coordinates of their top-left pixel.
    """
    corners = cam.to_map(np.array([[0, 0], [cam.width, 0], [0, cam.height], [cam.width, cam.height]]), 0, 0)
    x0, y0 = np.floor(corners.min(axis=0))
    x1, y1 = np.ceil(corners.max(axis=0))
    size = (int((x1 - x0) * PPT), int((y1 - y0) * PPT))
    shift = np.array([[PPT, 0, -x0 * PPT], [0, PPT, -y0 * PPT], [0, 0, 1]], float)
    m = shift @ cam.matrix(0, 0)
    out = {k: cv2.warpPerspective(v.astype(np.float32), m, size, flags=cv2.INTER_LINEAR)
           for k, v in masks.items()}
    return out, (float(x0), float(y0))


@dataclass
class Fix:
    x: float          # map tile under the screen centre (in the map's drawn orientation)
    y: float
    score: float      # 0..1 fit quality
    rotated: bool


def match(masks: dict[str, np.ndarray], cam: Camera, layout: LayoutImages,
          near: tuple[float, float] | None = None, radius: float = 3.0) -> Fix | None:
    """Best camera position for one frame; search only within `radius` tiles of `near` if given."""
    td, (ox, oy) = top_down(masks, cam)
    valid = (td["valid"] > 0.5).astype(np.float32)
    if td["bush"].sum() + td["water"].sum() < 4 * PPT * PPT:   # under ~4 tiles of features: no fix
        return None
    # Score = overlap ratio (IoU) of features, pooled over bush and water, inside
    # the part of the ground the screen covers: shared area / combined area.
    inter, union = None, None
    for name in ("bush", "water"):
        tpl = td[name] * valid
        img = layout.channels[name]
        if tpl.sum() == 0 and img.sum() == 0:
            continue
        if tpl.shape[0] >= img.shape[0] or tpl.shape[1] >= img.shape[1]:
            raise ValueError("screen covers more than the padded layout; raise PAD")
        both = cv2.matchTemplate(img, tpl, cv2.TM_CCORR)
        lay_area = cv2.matchTemplate(img, valid, cv2.TM_CCORR)
        u = lay_area + tpl.sum() - both
        inter = both if inter is None else inter + both
        union = u if union is None else union + u
    if inter is None:
        return None
    total = inter / np.maximum(union, 1.0)
    if near is not None:
        # Restrict to positions near the previous fix.
        nx = (near[0] + PAD + ox) * PPT
        ny = (near[1] + PAD + oy) * PPT
        r = radius * PPT
        yy, xx = np.mgrid[:total.shape[0], :total.shape[1]]
        total = np.where((np.abs(xx - nx) <= r) & (np.abs(yy - ny) <= r), total, -np.inf)
    by, bx = np.unravel_index(np.argmax(total), total.shape)
    best = total[by, bx]
    # Sub-pixel peak (parabola through the neighbours).
    def refine(a, b, c):
        den = a - 2 * b + c
        return 0.0 if not np.isfinite(den) or den == 0 else 0.5 * (a - c) / den
    dx = refine(total[by, bx - 1], best, total[by, bx + 1]) if 0 < bx < total.shape[1] - 1 else 0.0
    dy = refine(total[by - 1, bx], best, total[by + 1, bx]) if 0 < by < total.shape[0] - 1 else 0.0
    x = (bx + dx) / PPT - PAD - ox
    y = (by + dy) / PPT - PAD - oy
    return Fix(float(x), float(y), float(max(0.0, best)), layout.rotated)


# --- calibration ----------------------------------------------------------------

def calibrate(frames: list[np.ndarray], ref: MapRef, cam: Camera | None = None) -> tuple[Camera, bool, float]:
    """Fit the camera perspective (kx, kv, p) and map orientation to a set of frames.

    Maximises the mean match score over the frames. Returns (camera, rotated, score).
    """
    from scipy.optimize import minimize

    h, w = frames[0].shape[:2]
    cam = cam or default_camera(w, h)
    masks = [feature_masks(f) for f in frames]
    best = None
    for rotated in (False, True):
        layout = layout_images(ref, rotated)

        def cost(params):
            kx, kv, p = params
            if kx <= 0 or kv <= 0:
                return 1e9
            c = Camera(w, h, kx, kv, p)
            scores = []
            for m in masks:
                try:
                    fix = match(m, c, layout)
                except ValueError:
                    return 1e9
                scores.append(fix.score if fix else 0.0)
            return -float(np.mean(scores))

        x0 = np.array([cam.kx, cam.kv, cam.p])
        res = minimize(cost, x0, method="Nelder-Mead",
                       options=dict(initial_simplex=[x0, x0 * [1.15, 1, 1] + [0, 0, 0],
                                                     x0 * [1, 1.15, 1], x0 + [0, 0, 4e-4]],
                                    xatol=1e-6, fatol=1e-4, maxiter=150))
        cand = (Camera(w, h, *res.x), rotated, -res.fun)
        if best is None or cand[2] > best[2]:
            best = cand
    return best


# --- whole video -----------------------------------------------------------------

@dataclass
class FramePose:
    video_time: float
    x: float | None            # map tile under the screen centre (smoothed)
    y: float | None
    raw_x: float | None        # unsmoothed fit
    raw_y: float | None
    score: float
    status: str                # ok | low_score | outlier | no_features


def localize_video(video: Path, ref: MapRef, cam: Camera, rotated: bool, fps: float = 10.0,
                   min_score: float = 0.25) -> list[FramePose]:
    """Camera position per frame.

    Each frame is searched near the previous good fix (the camera can pan fast
    when the viewer swipes, ~2 tiles per frame at 10 fps, so the window is
    generous), with a full-map search when that fit is poor. Then the track is
    cleaned: a fix that disagrees with the frames on both sides, while those
    agree with each other, is an outlier. Real pans and replay skips keep
    agreeing with the frames that follow, so they survive.
    """
    layout = layout_images(ref, rotated)
    fixes: list[tuple[float, Fix | None]] = []
    last: Fix | None = None
    for f in iter_frames(video, fps=fps):
        masks = feature_masks(f.image)
        fix = match(masks, cam, layout, near=(last.x, last.y), radius=4.0) if last else None
        if fix is None or fix.score < min_score:
            full = match(masks, cam, layout)
            if full and (fix is None or full.score > fix.score):
                fix = full
        if fix and fix.score >= min_score:
            last = fix
        fixes.append((f.video_time, fix))
    return clean_track(fixes, min_score)


def clean_track(fixes: list[tuple[float, "Fix | None"]], min_score: float = 0.25,
                side: int = 3, tol: float = 1.0) -> list[FramePose]:
    """Flag outliers and smooth (see localize_video)."""
    n = len(fixes)
    xy = np.full((n, 2), np.nan)
    good = np.zeros(n, bool)
    for i, (_, fx) in enumerate(fixes):
        if fx is not None:
            xy[i] = (fx.x, fx.y)
            good[i] = fx.score >= min_score

    def side_median(i, step):
        idx, j = [], i + step
        while 0 <= j < n and len(idx) < side:
            if good[j]:
                idx.append(j)
            j += step
        return np.median(xy[idx], axis=0) if len(idx) >= 2 else None

    outlier = np.zeros(n, bool)
    for i in np.flatnonzero(good):
        before, after = side_median(i, -1), side_median(i, +1)
        far = lambda m: m is not None and np.hypot(*(xy[i] - m)) > tol
        if far(before) and far(after) and np.hypot(*(before - after)) <= tol:
            outlier[i] = True
        elif far(before) and after is None or far(after) and before is None:
            outlier[i] = True    # at either end, only one side to compare with
    use = good & ~outlier

    out = []
    for i, (t, fx) in enumerate(fixes):
        if fx is None:
            out.append(FramePose(t, None, None, None, None, 0.0, "no_features"))
            continue
        status = "outlier" if outlier[i] else "ok" if good[i] else "low_score"
        sx = sy = None
        if use[i]:
            # Median of this fix and its usable neighbours (one each side): removes
            # single-frame wobble without lagging behind pans.
            nb = [j for j in (i - 1, i, i + 1) if 0 <= j < n and use[j]]
            sx, sy = (round(float(v), 3) for v in np.median(xy[nb], axis=0))
        out.append(FramePose(t, sx, sy, round(fx.x, 3), round(fx.y, 3), round(fx.score, 3), status))
    return out


# --- debug output ----------------------------------------------------------------

LAYOUT_COLORS = {"B": (60, 230, 60), "~": (255, 200, 0), "W": (60, 60, 255), "O": (40, 40, 40),
                 "b": (255, 120, 0), "r": (0, 0, 255)}


def overlay(image: np.ndarray, ref: MapRef, cam: Camera, x: float, y: float, rotated: bool) -> np.ndarray:
    """Draw the map's tile grid and layout outlines onto the frame, for checking alignment by eye."""
    out = image.copy()
    def proj(pts):
        pts = np.asarray(pts, float)
        if rotated:   # layout drawn as-is; the screen sees it turned 180°
            pts = np.stack([ref.cols - pts[:, 0], ref.rows - pts[:, 1]], axis=1)
        return cam.to_screen(pts, x, y)
    for c in range(ref.cols + 1):
        a, b = proj([[c, 0], [c, ref.rows]])
        cv2.line(out, tuple(map(int, a)), tuple(map(int, b)), (255, 255, 255), 1, cv2.LINE_AA)
    for r in range(ref.rows + 1):
        a, b = proj([[0, r], [ref.cols, r]])
        cv2.line(out, tuple(map(int, a)), tuple(map(int, b)), (255, 255, 255), 1, cv2.LINE_AA)
    for r, row in enumerate(ref.layout):
        for c, ch in enumerate(row):
            if ch in LAYOUT_COLORS:
                quad = proj([[c, r], [c + 1, r], [c + 1, r + 1], [c, r + 1]]).astype(np.int32)
                cv2.polylines(out, [quad], True, LAYOUT_COLORS[ch], 2, cv2.LINE_AA)
    cv2.putText(out, f"camera at tile ({x:.1f}, {y:.1f})", (10, image.shape[0] - 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def ground_check(video: Path, ref: MapRef, cam: Camera, rotated: bool, poses: list[FramePose],
                 fps: float = 1.0) -> dict | None:
    """Accuracy against something flat that the fit didn't use (map.json "ground_check").

    "checker": the floor's squares are one tile each; project the frame
      top-down at the fitted position and measure where the square edges fall
      relative to the predicted tile grid.
    "water":  fit again using water only (flat) and compare with the full fit.
    Returns error statistics in tiles, or None if the map has no such check.
    """
    kind = ref.ground_check
    if kind not in ("checker", "water"):
        return None
    by_time = {round(p.video_time, 1): p for p in poses if p.status == "ok"}
    errs = []
    water_only = None
    if kind == "water":
        full = layout_images(ref, rotated)
        water_only = LayoutImages({"bush": full.channels["bush"] * 0, "water": full.channels["water"]}, rotated)
    for f in iter_frames(video, fps=fps):
        p = by_time.get(round(f.video_time, 1))
        if p is None:
            continue
        if kind == "checker":
            e = _checker_error(f.image, cam, p.x, p.y)
        else:
            m = feature_masks(f.image)
            if m["water"].sum() < 3000:
                continue
            w = match({"bush": m["bush"] * 0, "water": m["water"], "valid": m["valid"]}, cam, water_only,
                      near=(p.x, p.y), radius=2.0)
            e = (w.x - p.x, w.y - p.y) if w and w.score > 0.4 else None
        if e is not None:
            errs.append(e)
    if not errs:
        return None
    e = np.array(errs)
    dist = np.hypot(e[:, 0], e[:, 1])
    return dict(method=kind, frames=len(e), median_x=round(float(np.median(e[:, 0])), 3),
                median_y=round(float(np.median(e[:, 1])), 3), median_dist=round(float(np.median(dist)), 3),
                p90_dist=round(float(np.percentile(dist, 90)), 3))


def _checker_error(img: np.ndarray, cam: Camera, x: float, y: float, ppt: int = 16) -> tuple[float, float] | None:
    """Offset (tiles) of the floor's square edges from the predicted grid, from their periodicity."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    keep = ((ui_mask(*img.shape[:2]) > 0) & (feature_masks(img)["bush"] == 0)).astype(np.float32)
    x0, y0 = np.floor(x) - 8, np.floor(y) - 5
    size = (16 * ppt, 10 * ppt)
    m = np.array([[ppt, 0, -x0 * ppt], [0, ppt, -y0 * ppt], [0, 0, 1]], float) @ cam.matrix(x, y)
    td = cv2.warpPerspective(gray, m, size)
    valid = cv2.warpPerspective(keep, m, size) > 0.99
    out = []
    for axis, (dx, dy) in ((0, (1, 0)), (1, (0, 1))):
        g = np.abs(cv2.Sobel(td, cv2.CV_32F, dx, dy)) * valid
        prof = g.sum(axis=axis)
        spec = np.fft.rfft(prof - prof.mean())
        k = len(prof) // ppt                     # one cycle per tile
        if np.abs(spec[k]) < 0.1 * np.abs(spec[1:]).max():
            return None                          # no clear grid in view (bushes, effects)
        out.append((np.angle(spec[k]) / (2 * np.pi) + 0.5) % 1 - 0.5)
    return out[0], out[1]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Camera position on the map for every frame.")
    ap.add_argument("video", type=Path)
    ap.add_argument("--map", required=True, help="map id (folder in maps/)")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=Path("runs/localize"))
    ap.add_argument("--calib-frames", type=int, default=16)
    ap.add_argument("--check", action="store_true", help="measure accuracy against flat ground features")
    args = ap.parse_args(argv)

    ref = load(args.map)
    out = args.out / args.video.stem
    out.mkdir(parents=True, exist_ok=True)
    sample = [f.image for f in iter_frames(args.video, fps=0.5)]
    picks = [sample[i] for i in np.linspace(len(sample) * 0.1, len(sample) * 0.9, args.calib_frames).astype(int)]
    cam, rotated, score = calibrate(picks, ref)
    print(f"camera: kx={cam.kx:.5f} kv={cam.kv:.5f} p={cam.p:.6f}  rotated={rotated}  calib score={score:.3f}")
    poses = localize_video(args.video, ref, cam, rotated, args.fps)
    with (out / "poses.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(poses[0])))
        w.writeheader()
        w.writerows(asdict(p) for p in poses)
    (out / "camera.json").write_text(json.dumps(dict(asdict(cam), rotated=rotated, calib_score=score), indent=2))
    counts = {s: sum(p.status == s for p in poses) for s in ("ok", "low_score", "outlier", "no_features")}
    print(f"frames: {counts}")
    if args.check:
        acc = ground_check(args.video, ref, cam, rotated, poses)
        if acc:
            (out / "accuracy.json").write_text(json.dumps(acc, indent=2))
            print(f"accuracy ({acc['method']}, {acc['frames']} frames): median error {acc['median_dist']:.2f} tiles, "
                  f"90% within {acc['p90_dist']:.2f}; bias x {acc['median_x']:+.2f}, y {acc['median_y']:+.2f}")
    # Overlays on a spread of frames.
    ok = [p for p in poses if p.status == "ok"]
    times = {round(p.video_time, 1): p for p in ok[:: max(1, len(ok) // 8)]}
    for fr in iter_frames(args.video, fps=args.fps):
        p = times.get(round(fr.video_time, 1))
        if p:
            cv2.imwrite(str(out / f"overlay_{fr.video_time:06.1f}.jpg"), overlay(fr.image, ref, cam, p.x, p.y, rotated))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
