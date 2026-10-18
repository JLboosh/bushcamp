import cv2
import numpy as np
import pytest

from pipeline.timer import BarLayout, assign_times, build_clock, estimate_rate, locate_bar, read_bar

H, W = 720, 1562
X0, Y0, Y1 = 510, 613, 646
YELLOW = (40, 210, 250)   # BGR, hue ~26


def frame(fill_px: int | None, buttons: bool = True, seed: int = 0) -> np.ndarray:
    """Replay frame: busy background, yellow skip buttons, progress bar with `fill_px` filled."""
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.integers(60, 160, (36, 78, 3), dtype=np.uint8), (W, H))
    if buttons:
        for x in (108, 210, 1355, 1458):
            cv2.rectangle(img, (x, 550), (x + 90, 605), YELLOW, -1)
            cv2.rectangle(img, (x + 30, 565), (x + 60, 590), (30, 30, 30), -1)   # icon
    if fill_px is not None:
        cv2.rectangle(img, (X0, Y0), (1125, Y1), (70, 60, 60), -1)          # dark track
        if fill_px > 0:
            cv2.rectangle(img, (X0, Y0), (X0 + fill_px - 1, Y1), YELLOW, -1)
    return img


def test_locate_bar_ignores_static_buttons():
    layout = locate_bar(frame(n, seed=n) for n in range(5, 120, 8))
    assert layout is not None
    assert abs(layout.x0 - X0) <= 1
    assert abs(layout.y0 - Y0) <= 2 and abs(layout.y1 - Y1) <= 2


def test_locate_bar_none_without_bar():
    assert locate_bar(frame(None, seed=k) for k in range(6)) is None


def test_read_bar():
    layout = BarLayout(X0, Y0, Y1)
    for n in (3, 57, 400, 615):
        assert read_bar(frame(n), layout) == pytest.approx(n, abs=1)
    assert read_bar(frame(None), layout) is None
    # a big yellow effect over the bar area isn't a bar reading
    img = frame(50)
    cv2.rectangle(img, (X0 - 20, Y0 - 60), (X0 + 200, Y1 + 60), YELLOW, -1)
    assert read_bar(img, layout) is None


RATE = 8.0


def playback(rewind=True):
    """60 s of video at 10 fps with a skip, a rewind, a freeze-and-catch-up and a hidden stretch.

    Returns video times, bar readings (NaN when hidden) and the true match time.
    """
    vt = np.round(np.arange(0, 60, 0.1), 3)
    match = vt.copy()
    match[vt >= 20] += 10           # skip forward 10 s at 20 s
    if rewind:
        match[vt >= 40] -= 15       # rewind 15 s at 40 s
    freeze = (vt >= 50) & (vt < 51)
    match[freeze] = match[np.flatnonzero(freeze)[0]]   # game freezes 1 s at 50 s, then catches up
    px = np.round(match * RATE)
    px[(vt >= 25) & (vt < 30)] = np.nan   # controls hidden
    px[vt == 12.0] = 2                    # one half-faded reading
    return vt, px, match


def test_estimate_rate_ignores_jumps():
    vt, px, _ = playback()
    assert estimate_rate(vt, px) == pytest.approx(RATE, rel=0.03)


def test_clock_events_and_accuracy():
    vt, px, match = playback()
    clock = build_clock(vt, px)
    assert clock.rate == pytest.approx(RATE, rel=0.01)
    assert clock.glitches == [12.0]
    kinds = [(e.type, round(e.jump)) for e in clock.events]
    assert kinds == [("skip_forward", 10), ("rewind", -15), ("stutter", 1)]

    times = assign_times(vt, px, clock)
    got = np.array([t.match_time for t in times])
    assert np.max(np.abs(got - match)) < 0.3
    src = {t.video_time: t.source for t in times}
    assert src[10.0] == "bar" and src[27.0] == "inferred" and src[12.0] == "inferred"
    # after the rewind, match times 25-30 s are shown a second time
    rep = {t.video_time: t.repeat for t in times}
    assert not rep[39.0] and rep[41.0]


def test_skip_inside_hidden_stretch_is_uncertain():
    vt = np.round(np.arange(0, 30, 0.1), 3)
    match = vt + np.where(vt >= 15, 20.0, 0.0)
    px = np.round(match * RATE)
    px[(vt >= 12) & (vt < 18)] = np.nan            # the skip happens while hidden
    clock = build_clock(vt, px)
    assert [(e.type, round(e.jump)) for e in clock.events] == [("skip_forward", 20)]
    times = {t.video_time: t for t in assign_times(vt, px, clock)}
    assert times[14.0].source == "uncertain"
    assert times[20.0].source == "bar" and times[20.0].match_time == pytest.approx(40, abs=0.2)


def test_no_readings_means_no_clock():
    vt = np.arange(0, 10, 0.1)
    px = np.full_like(vt, np.nan)
    clock = build_clock(vt, px)
    assert clock.rate is None
    assert {t.source for t in assign_times(vt, px, clock)} == {"none"}
