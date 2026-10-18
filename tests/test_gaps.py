import numpy as np
import pytest

from pipeline.gaps import INTERPOLATED, OBSERVED, UNKNOWN, drop_spikes, resample


def flags_at(track, times):
    return [track.flag[int(round((t - track.t[0]) / 0.1))] for t in times]


def test_short_gap_interpolated_long_gap_unknown():
    t = np.r_[np.arange(0, 2, 0.1), np.arange(3.5, 5, 0.1), np.arange(9, 10, 0.1)]
    x = t * 1.0          # walking 1 tile/s
    y = np.zeros_like(t)
    tr = resample(t, x, y)
    assert flags_at(tr, [1.0, 2.5, 6.0, 9.5]) == [OBSERVED, INTERPOLATED, UNKNOWN, OBSERVED]
    j = int(round(2.5 / 0.1))
    assert tr.x[j] == pytest.approx(2.5, abs=0.05)          # straight line across the gap
    assert np.isnan(tr.x[int(round(6.0 / 0.1))])            # unknown stays unknown


def test_impossible_jump_not_bridged():
    # 1 s gap but 15 tiles apart: a respawn or a mismatch, not a walk
    t = np.r_[np.arange(0, 1, 0.1), np.arange(2, 3, 0.1)]
    x = np.r_[np.zeros(10), np.full(10, 15.0)]
    tr = resample(t, x, np.zeros_like(t))
    assert flags_at(tr, [1.5]) == [UNKNOWN]


def test_repeated_match_times_are_averaged():
    # a rewound replay shows the same moment twice
    t = np.array([1.0, 1.1, 1.0, 1.1])
    x = np.array([2.0, 2.1, 2.2, 2.3])
    tr = resample(t, x, np.zeros(4))
    assert tr.n_obs.tolist() == [2, 2]
    assert tr.x.tolist() == pytest.approx([2.1, 2.2])


def test_common_time_range():
    tr = resample(np.array([5.0, 5.5]), np.array([1.0, 1.5]), np.zeros(2), t_start=4.0, t_end=6.0)
    assert len(tr.t) == 21 and tr.t[0] == 4.0
    assert tr.flag[0] == UNKNOWN and tr.flag[-1] == UNKNOWN


def test_drop_spikes():
    t = np.arange(0, 1, 0.1)
    x = np.ones(10)
    x[5] = 9.0                       # one box on the wrong thing
    keep = drop_spikes(t, x, np.zeros(10))
    assert keep.tolist() == [True] * 5 + [False] + [True] * 4
