"""Match clock: match time for every frame, from the replay progress bar.

Knockout replays have no on-screen match timer, so the clock comes from the
yellow replay progress bar at the bottom of the screen. (Modes with a digit
timer, e.g. Brawl Ball, can add a second reader later; see the brief.)

Steps:
1. locate_bar: find the bar's left end and rows. The bar is the yellow strip
   whose right end moves over time (the skip buttons are yellow too, but static).
2. read_bar: filled pixels on each frame, or None when the controls are hidden
   or fading.
3. build_clock: convert pixels to seconds and split the recording into
   segments of normal playback, separated by skips and rewinds.
   - Fill rate (px per second of replay) is estimated per video from
     normal-speed stretches, so total replay length and resolution never matter.
   - Readings are only ~0.2 s apart in precision (1 px), so within a segment
     match time is fitted as video_time + offset, which is sub-pixel accurate.
   - A reading that disagrees with both neighbours, while they agree with each
     other, is a glitch (half-faded bar) and is dropped.
   - When the bar is hidden, the clock carries forward from the segment, so
     frames get a source flag saying how much to trust them.

"Match time" is replay time in seconds from the start of the replay (which
includes the pre-match intro).

CLI:
    python -m pipeline.timer data/raw/match.mp4 --fps 10
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pipeline.sampling import iter_frames

# Yellow fill of the progress bar (OpenCV HSV), measured on both MVP maps.
BAR_YELLOW = dict(h=(18, 35), s_min=130, v_min=190)
SEARCH_BOTTOM = 0.3         # the bar sits in the bottom 30% of the frame
BAR_LEFT_RANGE = (0.2, 0.5) # ...and its left end in this part of the width (measured: 0.33)

# Clock building.
RATE_PAIR_GAP = (4.0, 10.0) # seconds between readings for the first fill-rate guess
MIN_RATE_PAIRS = 3
MIN_SEGMENT_READINGS = 3  # fewer readings than this between jumps = glitch
TOLERANCE_S = 0.5           # disagreement (s) above which two readings aren't continuous playback
SMOOTH_S = 0.3              # half-width (s) of the running median over readings
STUTTER_MAX_S = 2.0         # jumps smaller than this are the game freezing and catching up


# --- reading the bar -----------------------------------------------------------

@dataclass
class BarLayout:
    x0: int   # first column of the fill (left end of the bar)
    y0: int   # rows of the yellow band
    y1: int


def _yellow(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lo, hi = BAR_YELLOW["h"]
    return (h >= lo) & (h <= hi) & (s >= BAR_YELLOW["s_min"]) & (v >= BAR_YELLOW["v_min"])


def _runs(row: np.ndarray, min_len: int = 3) -> list[tuple[int, int]]:
    """(start, end) of each run of True at least min_len long; end is exclusive."""
    padded = np.concatenate([[False], row, [False]]).astype(np.int8)
    d = np.diff(padded)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_len]


def _kendall_tau(y: np.ndarray) -> float:
    """Rank correlation of y with its own order: +1 if y only ever increases."""
    if len(y) < 2:
        return 0.0
    d = np.sign(y[None, :] - y[:, None])[np.triu_indices(len(y), 1)]
    return float(d.sum() / len(d))


def locate_bar(images) -> BarLayout | None:
    """Find the progress bar from frames in time order (any iterable of images).

    Collects yellow runs in the bottom of each frame and groups them by left
    edge. The bar is the group whose right end grows over time and that starts
    between the button groups (the yellow skip buttons sit at the screen
    edges and only change while fading in or out).
    """
    by_left: dict[int, list[tuple[int, int, int]]] = {}   # left x -> [(frame, row, right x)]
    masks, width, top = [], 0, 0
    for k, img in enumerate(images):
        top, width = int(img.shape[0] * (1 - SEARCH_BOTTOM)), img.shape[1]
        mask = _yellow(img[top:])
        masks.append(np.packbits(mask, axis=1))           # kept small to measure the band later
        for r in range(mask.shape[0]):
            for s, e in _runs(mask[r]):
                by_left.setdefault(s, []).append((k, r + top, e))

    def group(left):
        # Merge ±1 px: the bar's rounded cap shifts the left edge slightly between rows.
        return [x for d in (-1, 0, 1) for x in by_left.get(left + d, [])]

    best, best_score = None, 0.0
    for left in by_left:
        if not BAR_LEFT_RANGE[0] * width <= left <= BAR_LEFT_RANGE[1] * width:
            continue
        per_frame: dict[int, list[int]] = {}
        for k, _, e in group(left):
            per_frame.setdefault(k, []).append(e)
        if len(per_frame) < 3:
            continue
        rights = np.array([np.median(per_frame[k]) for k in sorted(per_frame)])
        tau = _kendall_tau(rights)
        if tau < 0.5 or np.ptp(rights) < 3:
            continue
        score = tau * len(per_frame)
        if score > best_score:
            best, best_score = left, score
    if best is None:
        return None
    g = group(best)
    lefts = np.array([l for l in (best - 1, best, best + 1) for _ in by_left.get(l, [])])
    x0 = int(np.median(lefts))

    # Band rows: the vertical extent of yellow just inside the fill (past the
    # rounded cap), measured on frames where the fill is long enough.
    tops, bottoms = [], []
    for k in {k for k, _, e in g if e - x0 >= 12}:
        mask = np.unpackbits(masks[k], axis=1, count=width).astype(bool)
        col = mask[:, x0 + 5:x0 + 10].mean(axis=1) >= 0.6
        rows = [r for r in range(len(col)) if col[r]]
        mid = int(np.median([r - top for kk, r, _ in g if kk == k]))
        runs = [(s, e) for s, e in _runs(col, 1) if s <= mid < e]
        if runs:
            tops.append(runs[0][0] + top)
            bottoms.append(runs[0][1] - 1 + top)
    if not tops:                                          # fill never long enough: fall back on rows
        rows = np.array([r for _, r, _ in g])
        tops, bottoms = [rows.min()], [rows.max()]
    return BarLayout(x0=x0, y0=int(np.median(tops)), y1=int(np.median(bottoms)))


def read_bar(image: np.ndarray, layout: BarLayout) -> float | None:
    """Filled pixels from the bar's left end, or None if the bar isn't (fully) visible."""
    height = layout.y1 - layout.y0 + 1
    m = max(1, height // 5)                       # skip the rounded top/bottom edges
    band = _yellow(image[layout.y0 + m:layout.y1 - m + 1, layout.x0:])
    cols = band.mean(axis=0) >= 0.7
    if not cols[:4].any():
        return None
    start = int(np.argmax(cols[:4]))              # the rounded cap can start a few pixels late
    rest = cols[start:]
    n = start + (int(np.argmin(rest)) if not rest.all() else rest.size)   # contiguous run from x0
    # The bar is a band of fixed height: clear of it above and below must not be
    # yellow (rejects big yellow effects and the fade-in/out animation).
    for y in (layout.y0 - height // 2 - 3, layout.y1 + height // 2 + 3):
        if 0 <= y < image.shape[0] and _yellow(image[y:y + 1, layout.x0:layout.x0 + n]).mean() > 0.5:
            return None
    return float(n)


# --- building the clock ----------------------------------------------------------

@dataclass
class Segment:
    """A stretch of continuous playback (no skips or rewinds inside).

    Match time is followed reading by reading rather than fitted as one line,
    because replays stutter: the game can freeze for a second and then catch up.
    """
    start: float               # video time of the first and last bar reading in it
    end: float
    knots_t: list[float]       # reading times and their smoothed match times
    knots_m: list[float]

    @property
    def readings(self) -> int:
        return len(self.knots_t)

    def match_time(self, video_time: float) -> float:
        """Linear between readings (bridges hidden stretches); normal speed beyond the ends."""
        if video_time < self.start:
            return self.knots_m[0] - (self.start - video_time)
        if video_time > self.end:
            return self.knots_m[-1] + (video_time - self.end)
        return float(np.interp(video_time, self.knots_t, self.knots_m))


@dataclass
class Event:
    type: str              # skip_forward | rewind | stutter
    video_from: float      # it happened somewhere between these video times
    video_to: float
    match_from: float
    match_to: float

    @property
    def jump(self) -> float:
        """Seconds gained (+) or lost (-) compared with playing straight through."""
        return (self.match_to - self.match_from) - (self.video_to - self.video_from)


@dataclass
class Clock:
    rate: float | None                     # bar pixels per second of replay
    segments: list[Segment] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    glitches: list[float] = field(default_factory=list)   # video times of dropped readings


def estimate_rate(vt: np.ndarray, px: np.ndarray) -> float | None:
    """First guess at the fill speed (px/s): median over pairs of readings several
    seconds apart (short gaps are biased by whole-pixel rounding; skips are outliers)."""
    ok = ~np.isnan(px)
    t, p = vt[ok], px[ok]
    rates = []
    j = 0
    for i in range(len(t)):
        while j < len(t) and t[j] - t[i] < RATE_PAIR_GAP[0]:
            j += 1
        if j < len(t) and t[j] - t[i] <= RATE_PAIR_GAP[1] and p[j] > p[i]:
            rates.append((p[j] - p[i]) / (t[j] - t[i]))
    if len(rates) < MIN_RATE_PAIRS:
        return None
    return float(np.median(rates))


def refine_rate(vt: np.ndarray, px: np.ndarray, clock: Clock) -> float | None:
    """Fill speed from long continuous segments: a 20 s baseline pins it to a
    fraction of a percent, where short pairs only get within ~5%.

    Uses the median slope between readings at least 5 s apart, so a freeze
    inside a segment or a stray reading doesn't bend it; glitches are excluded.
    """
    usable = ~np.isnan(px) & ~np.isin(vt, clock.glitches)
    slopes = []
    for s in clock.segments:
        sel = usable & (vt >= s.start) & (vt <= s.end)
        t, p = vt[sel], px[sel]
        j = np.searchsorted(t, t + 5.0)
        pairs = j < len(t)
        if pairs.sum() >= 10:
            slopes += list((p[j[pairs]] - p[pairs]) / (t[j[pairs]] - t[pairs]))
    if not slopes:
        return None
    rate = float(np.median(slopes))
    return rate if abs(rate / clock.rate - 1) < 0.15 else None


def build_clock(vt: np.ndarray, px: np.ndarray, rate: float | None = None) -> Clock:
    """Segments and events from bar readings (px is NaN where the bar wasn't readable).

    Pass `rate` to fix the fill speed; otherwise it's estimated, then refined.
    """
    vt, px = np.asarray(vt, float), np.asarray(px, float)
    if rate:
        return _build(vt, px, rate)
    guess = estimate_rate(vt, px)
    if guess is None:
        return Clock(None)
    clock = _build(vt, px, guess)
    refined = refine_rate(vt, px, clock)
    return _build(vt, px, refined) if refined else clock


def _build(vt: np.ndarray, px: np.ndarray, rate: float) -> Clock:
    clock = Clock(rate)
    tol = max(TOLERANCE_S, 2.5 / rate)            # at least 2.5 px of rounding
    m = px / rate

    def continuous(i, j):
        return abs(m[j] - (m[i] + vt[j] - vt[i])) <= tol

    # Drop glitches (e.g. a half-faded bar): a reading inconsistent with both
    # neighbours while they agree with each other.
    idx = np.flatnonzero(~np.isnan(px))
    keep = []
    for k, i in enumerate(idx):
        if 0 < k < len(idx) - 1:
            p, n = idx[k - 1], idx[k + 1]
            if not continuous(p, i) and not continuous(i, n) and continuous(p, n):
                clock.glitches.append(float(vt[i]))
                continue
        keep.append(i)

    # Split into segments of continuous playback. A "segment" of only a couple
    # of readings is a glitch too (a real skip is followed by steady readings);
    # drop those and split again until nothing changes.
    while True:
        groups: list[list[int]] = []
        for i in keep:
            if groups and continuous(groups[-1][-1], i):
                groups[-1].append(i)
            else:
                groups.append([i])
        short = [g for g in groups if len(g) < MIN_SEGMENT_READINGS]
        if not short or len(short) == len(groups):
            break
        drop = {i for g in short for i in g}
        clock.glitches += [float(vt[i]) for i in sorted(drop)]
        keep = [i for i in keep if i not in drop]
    clock.glitches.sort()

    for g in groups:
        tt, mm = vt[g], m[g]
        # Smooth away whole-pixel rounding with a short running median of
        # (match - video); short enough to keep a 1 s freeze-and-catch-up.
        off = mm - tt
        smooth = [float(np.median(off[np.abs(tt - x) <= SMOOTH_S])) for x in tt]
        clock.segments.append(Segment(float(tt[0]), float(tt[-1]), [float(x) for x in tt],
                                      [float(x + o) for x, o in zip(tt, smooth)]))
    # Neighbouring segments that join up (normal speed across the boundary) were
    # split by one rough reading, not by a jump: merge them.
    merged = clock.segments[:1]
    for s in clock.segments[1:]:
        a = merged[-1]
        if abs(a.match_time(s.start) - s.knots_m[0]) <= tol:
            merged[-1] = Segment(a.start, s.end, a.knots_t + s.knots_t, a.knots_m + s.knots_m)
        else:
            merged.append(s)
    clock.segments = merged
    for a, b in zip(clock.segments, clock.segments[1:]):
        ev = Event("", a.end, b.start, round(a.knots_m[-1], 3), round(b.knots_m[0], 3))
        ev.type = ("stutter" if abs(ev.jump) < STUTTER_MAX_S
                   else "skip_forward" if ev.jump > 0 else "rewind")
        clock.events.append(ev)
    return clock


@dataclass
class FrameTime:
    video_time: float
    match_time: float | None
    source: str        # bar | inferred | extrapolated | uncertain | none
    segment: int       # index into clock.segments, -1 if none
    bar_px: float | None
    repeat: bool       # this match time was already seen earlier (after a rewind)


def assign_times(vt: np.ndarray, px: np.ndarray, clock: Clock) -> list[FrameTime]:
    """Match time for every frame.

    source:
      bar          read from the bar on this frame
      inferred     bar hidden, but readings on both sides are continuous playback
                   (same segment, or only a stutter between them)
      extrapolated before the first or after the last reading of the recording
      uncertain    in a gap that contains a skip or rewind (its exact moment is unknown)
      none         no clock could be built
    """
    out = []
    segs = clock.segments
    glitch = set(clock.glitches)
    seen_until = -np.inf
    for t, p in zip(vt, px):
        if not segs:
            out.append(FrameTime(float(t), None, "none", -1, None, False))
            continue
        k = next((i for i, s in enumerate(segs) if s.start <= t <= s.end), None)
        if k is not None:
            readable = not np.isnan(p) and float(t) not in glitch
            source = "bar" if readable else "inferred"
        elif t < segs[0].start:
            k, source = 0, "extrapolated"
        elif t > segs[-1].end:
            k, source = len(segs) - 1, "extrapolated"
        else:   # between two segments: something happened somewhere in here
            k = max(i for i, s in enumerate(segs) if s.end < t)
            a, b = segs[k], segs[k + 1]
            if clock.events[k].type == "stutter":   # small: bridge the two ends linearly
                mt = float(np.interp(t, [a.end, b.start], [a.knots_m[-1], b.knots_m[0]]))
                out.append(FrameTime(float(t), round(mt, 3), "inferred", k,
                                     None if np.isnan(p) else float(p), mt <= seen_until))
                seen_until = max(seen_until, mt)
                continue
            source = "uncertain"
        mt = segs[k].match_time(float(t))
        out.append(FrameTime(float(t), round(mt, 3), source, k,
                             None if np.isnan(p) else float(p), mt <= seen_until))
        seen_until = max(seen_until, mt)
    return out


# --- running on a video ----------------------------------------------------------

def run(video: str | Path, fps: float = 10.0, out: Path | None = None) -> tuple[Clock, list[FrameTime], BarLayout | None]:
    video = Path(video)
    layout = locate_bar(f.image for f in iter_frames(video, fps=2.0))
    vt, px = [], []
    for f in iter_frames(video, fps=fps):
        vt.append(f.video_time)
        r = read_bar(f.image, layout) if layout else None
        px.append(np.nan if r is None else r)
    vt, px = np.array(vt), np.array(px)
    clock = build_clock(vt, px)
    times = assign_times(vt, px, clock)
    if out:
        save(out, video, clock, times, layout)
    return clock, times, layout


def save(out: Path, video: Path, clock: Clock, times: list[FrameTime], layout: BarLayout | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "clock.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(times[0])))
        w.writeheader()
        w.writerows(asdict(t) for t in times)
    summary = dict(
        video=str(video), layout=asdict(layout) if layout else None, rate_px_per_s=clock.rate,
        segments=[dict(start=s.start, end=s.end, readings=s.readings,
                       match_start=round(s.knots_m[0], 3), match_end=round(s.knots_m[-1], 3))
                  for s in clock.segments],
        events=[asdict(e) | {"jump": round(e.jump, 2)} for e in clock.events],
        glitches=clock.glitches,
        sources={s: sum(t.source == s for t in times) for s in
                 ("bar", "inferred", "extrapolated", "uncertain", "none")},
    )
    (out / "clock.json").write_text(json.dumps(summary, indent=2))
    plot(out / "clock.png", video.name, clock, times)


def plot(path: Path, title: str, clock: Clock, times: list[FrameTime]) -> None:
    """Match time vs video time: the debug view for checking the clock by eye."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    vt = np.array([t.video_time for t in times])
    if clock.rate:
        raw = np.array([np.nan if t.bar_px is None else t.bar_px / clock.rate for t in times])
        ax.scatter(vt, raw, s=6, c="0.6", label="bar reading", zorder=2)
    styles = {"bar": "-", "inferred": "--", "extrapolated": ":", "uncertain": ":"}
    colors = {"bar": "tab:blue", "inferred": "tab:blue", "extrapolated": "tab:orange", "uncertain": "tab:red"}
    for src in styles:
        y = np.array([t.match_time if t.source == src else np.nan for t in times], float)
        if not np.isnan(y).all():
            ax.plot(vt, y, styles[src], color=colors[src], lw=2, label=src, zorder=3)
    for g in clock.glitches:
        ax.axvline(g, color="0.8", lw=0.8)
    for e in clock.events:
        ax.axvspan(e.video_from, e.video_to, color="tab:red" if e.type == "rewind" else "tab:green", alpha=0.15)
        ax.annotate(f"{e.type} {e.jump:+.1f}s", ((e.video_from + e.video_to) / 2, e.match_to),
                    fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("video time (s)")
    ax.set_ylabel("match time (s)")
    rate = f"{clock.rate:.2f} px/s" if clock.rate else "no clock"
    jumps = sum(e.type != "stutter" for e in clock.events)
    stutters = len(clock.events) - jumps
    ax.set_title(f"{title}  ({rate}, {jumps} skips/rewinds, {stutters} stutters)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def holdout_check(vt: np.ndarray, px: np.ndarray, clock: Clock, lengths=(2, 5, 10),
                  trials: int = 60, seed: int = 0) -> dict:
    """Accuracy of the clock where the bar is hidden, measured on this video.

    Hides random stretches of real readings inside continuous playback, lets the
    clock infer them, and compares with the hidden readings (glitches excluded).
    """
    rng = np.random.default_rng(seed)
    truth_px = px.copy()
    truth_px[np.isin(vt, clock.glitches)] = np.nan
    result = {}
    for length in lengths:
        errs = []
        for _ in range(trials):
            t0 = rng.uniform(vt[0], vt[-1] - length)
            win = (vt >= t0) & (vt < t0 + length)
            inside = any(s.start < t0 and s.end > t0 + length for s in clock.segments)
            if not inside or np.isnan(truth_px[win]).mean() > 0.5:
                continue
            masked = px.copy()
            masked[win] = np.nan
            times = assign_times(vt, masked, build_clock(vt, masked, rate=clock.rate))
            pred = np.array([times[i].match_time for i in np.flatnonzero(win)], float)
            truth = truth_px[win] / clock.rate
            ok = ~np.isnan(truth)
            errs += list(np.abs(pred[ok] - truth[ok]))
        if errs:
            e = np.array(errs)
            result[f"{length}s"] = dict(frames=len(e), median=round(float(np.median(e)), 3),
                                        p95=round(float(np.percentile(e, 95)), 3),
                                        max=round(float(e.max()), 3),
                                        within_half_second=round(float(np.mean(e <= 0.5)), 4))
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Match time per frame from the replay progress bar.")
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=Path("runs/clock"))
    ap.add_argument("--check", action="store_true",
                    help="also measure accuracy over hidden stretches (writes accuracy.json)")
    args = ap.parse_args(argv)
    for video in args.videos:
        clock, times, layout = run(video, args.fps, args.out / video.stem)
        src = {s: sum(t.source == s for t in times) for s in ("bar", "inferred", "extrapolated", "uncertain", "none")}
        print(f"{video.name}: bar at {layout}, rate {clock.rate and round(clock.rate, 2)} px/s")
        print(f"  frames by source: {src}")
        for e in clock.events:
            print(f"  {e.type:12s} at video {e.video_from:.1f}-{e.video_to:.1f}s: "
                  f"match {e.match_from:.1f} -> {e.match_to:.1f}s ({e.jump:+.1f}s)")
        if clock.glitches:
            print(f"  dropped {len(clock.glitches)} glitch readings")
        if args.check and clock.rate:
            vt = np.array([t.video_time for t in times])
            px = np.array([np.nan if t.bar_px is None else t.bar_px for t in times])
            acc = holdout_check(vt, px, clock)
            (args.out / video.stem / "accuracy.json").write_text(json.dumps(acc, indent=2))
            for k, a in acc.items():
                print(f"  hidden {k:>3s}: median error {a['median']:.2f}s, 95% {a['p95']:.2f}s, "
                      f"max {a['max']:.2f}s, within 0.5s {a['within_half_second']:.1%}")
        print(f"  -> {args.out / video.stem}")


if __name__ == "__main__":
    main()
