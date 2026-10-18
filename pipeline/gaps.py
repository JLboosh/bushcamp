"""Turn scattered position observations into an evenly spaced track on match time.

Every output sample carries a flag:
    observed      at least one detection fell in this time step
    interpolated  inside a short gap between observations (straight line)
    unknown       off screen / not seen for longer than max_gap, or a jump no
                  brawler could make (a respawn or a mismatch), so no guess

Observations repeated at the same match time (a rewound replay shows them
twice) are averaged into the same step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OBSERVED, INTERPOLATED, UNKNOWN = "observed", "interpolated", "unknown"


@dataclass
class Track:
    t: np.ndarray        # match time of each step (s)
    x: np.ndarray        # tiles; NaN where unknown
    y: np.ndarray
    flag: np.ndarray     # observed | interpolated | unknown
    n_obs: np.ndarray    # detections averaged into each observed step


def drop_spikes(t: np.ndarray, x: np.ndarray, y: np.ndarray, window: float = 0.35,
                tol: float = 1.5) -> np.ndarray:
    """Mask of observations to keep: drops a point that sits more than `tol`
    tiles from the median of its neighbours within `window` seconds on both
    sides (a box on the wrong thing for a frame or two)."""
    keep = np.ones(len(t), bool)
    for i in range(len(t)):
        before = (t < t[i]) & (t >= t[i] - window)
        after = (t > t[i]) & (t <= t[i] + window)
        if before.sum() >= 1 and after.sum() >= 1:
            nb = before | after
            mx, my = np.median(x[nb]), np.median(y[nb])
            if np.hypot(x[i] - mx, y[i] - my) > tol:
                keep[i] = False
    return keep


def resample(t: np.ndarray, x: np.ndarray, y: np.ndarray, step: float = 0.1, max_gap: float = 2.0,
             max_speed: float = 6.0, t_start: float | None = None, t_end: float | None = None) -> Track:
    """Bin observations (match time t, position x/y in tiles) onto a regular grid.

    max_gap:   longest gap (s) bridged by interpolation (brief: default 2 s).
    max_speed: tiles/s; a gap needing faster movement is left unknown
               (brawlers move ~2-3 tiles/s; dashes and supers are faster for a moment).
    """
    t, x, y = (np.asarray(a, float) for a in (t, x, y))
    order = np.argsort(t, kind="stable")
    t, x, y = t[order], x[order], y[order]
    t0 = (t_start if t_start is not None else (t.min() if len(t) else 0.0))
    t1 = (t_end if t_end is not None else (t.max() if len(t) else 0.0))
    n = int(np.floor((t1 - t0) / step + 1e-9)) + 1
    grid = t0 + step * np.arange(n)
    sx, sy, cnt = np.zeros(n), np.zeros(n), np.zeros(n, int)
    idx = np.round((t - t0) / step).astype(int)
    ok = (idx >= 0) & (idx < n)
    np.add.at(sx, idx[ok], x[ok])
    np.add.at(sy, idx[ok], y[ok])
    np.add.at(cnt, idx[ok], 1)
    seen = cnt > 0
    out_x = np.where(seen, sx / np.maximum(cnt, 1), np.nan)
    out_y = np.where(seen, sy / np.maximum(cnt, 1), np.nan)
    flag = np.where(seen, OBSERVED, UNKNOWN).astype(object)

    obs = np.flatnonzero(seen)
    for a, b in zip(obs[:-1], obs[1:]):
        if b - a <= 1:
            continue
        gap = (b - a) * step
        dist = np.hypot(out_x[b] - out_x[a], out_y[b] - out_y[a])
        if gap <= max_gap + 1e-9 and dist / gap <= max_speed:
            f = (np.arange(a + 1, b) - a) / (b - a)
            out_x[a + 1:b] = out_x[a] + f * (out_x[b] - out_x[a])
            out_y[a + 1:b] = out_y[a] + f * (out_y[b] - out_y[a])
            flag[a + 1:b] = INTERPOLATED
    return Track(grid, out_x, out_y, flag.astype(str), cnt)
