"""Analysis and event rules on synthetic tracks with known answers."""
import numpy as np
import pytest

from pipeline.analyze import Death, Match, Player, moments, rounds_and_deaths
from pipeline.events import detect, intervals
from pipeline.gaps import OBSERVED, UNKNOWN
from pipeline.maps import load

STEP = 0.1
T = np.round(np.arange(0, 20, STEP), 2)
N = len(T)
CFG = {
    "overextended": {"ahead_tiles": 3.0, "min_s": 1.0},
    "outnumbered": {"min_enemies": 2, "support_tiles": 4.0, "min_s": 0.5},
    "isolated_before_death": {"support_tiles": 4.0, "window_s": 5.0, "min_isolated_s": 3.0},
    "good_spacing": {"band_tiles": 1.5, "min_s": 4.0},
    "too_passive": {"min_s": 6.0},
    "merge_gap_s": 0.5,
}


def player(pid, team, xs, ys, rng=9.0, me=False, seen=None):
    xs = np.broadcast_to(np.asarray(xs, float), (N,)).copy()
    ys = np.broadcast_to(np.asarray(ys, float), (N,)).copy()
    flag = np.full(N, OBSERVED, dtype=object)
    if seen is not None:
        flag[~seen] = UNKNOWN
        xs[~seen] = np.nan
        ys[~seen] = np.nan
    return Player(pid, pid, team, me, "X", rng, None, xs, ys, flag.astype(str), np.ones(N, bool))


def match(players, view_bound=20.0, deaths=()):
    """Forward = +x (own spawn on the left, enemy spawn on the right)."""
    vb = {p.id: np.full(N, view_bound) for p in players}
    return Match(T, players, load("streets_with_no_name"), [(0.0, 20.0)], list(deaths), vb,
                 np.array([1.0, 0.0]), np.array([0.0, 16.5]))


def kinds(evs):
    return [e.type for e in evs if e.type != "round_start"]


def test_intervals():
    mask = np.zeros(N, bool)
    mask[10:30] = True        # 2 s
    mask[33:40] = True        # joined across a 0.3 s gap
    mask[100:103] = True      # too short
    assert intervals(mask, T, 1.0, 0.5) == [(10, 39)]


def test_overextended_when_ahead_and_in_range():
    me_x = np.where(T < 5, 5.0, 10.0)            # pushes 5 tiles ahead at t = 5
    m = match([player("me", "ally", me_x, 16, me=True), player("mate", "ally", 5.0, 16),
               player("foe", "enemy", 17.0, 16, rng=9.0)])
    evs = detect(m, m.me, moments(m, m.me), CFG)
    over = [e for e in evs if e.type == "overextended"]
    assert len(over) == 1
    assert over[0].start == pytest.approx(5.0) and over[0].end == pytest.approx(19.9)
    assert over[0].measurements["ahead_max"] == pytest.approx(5.0)
    assert over[0].players == ["me", "foe"]


def test_no_overextended_when_an_unseen_teammate_could_be_closer():
    # the visible teammate is far behind, the other is off screen but the
    # screen edge is only 2 tiles away: they could be right beside us
    m = match([player("me", "ally", 10.0, 16, me=True), player("far", "ally", 2.0, 16),
               player("hidden", "ally", 0.0, 0, seen=np.zeros(N, bool)),
               player("foe", "enemy", 17.0, 16)], view_bound=2.0)
    assert "overextended" not in kinds(detect(m, m.me, moments(m, m.me), CFG))


def test_outnumbered_needs_two_in_range_and_no_support():
    def run(mate_x):
        m = match([player("me", "ally", 10.0, 16, me=True), player("mate", "ally", mate_x, 16),
                   player("a", "enemy", 15.0, 16), player("b", "enemy", 14.0, 18)])
        return kinds(detect(m, m.me, moments(m, m.me), CFG))
    assert "outnumbered" in run(mate_x=2.0)        # teammate 8 tiles away
    assert "outnumbered" not in run(mate_x=12.0)   # teammate 2 tiles away


def test_off_screen_enemy_counts_as_out_of_range_only_if_provably_far():
    seen = np.zeros(N, bool)
    def run(view_bound):
        m = match([player("me", "ally", 3.0, 16, me=True, rng=6.0), player("mate", "ally", 12.0, 16),
                   player("near", "enemy", 15.0, 16, rng=9.0),            # trading with the teammate
                   player("ghost", "enemy", 0, 0, seen=seen)], view_bound=view_bound)
        return m, moments(m, m.me)
    m, mo = run(view_bound=12.0)                   # unseen enemy is 12+ tiles away: out of everyone's reach
    assert mo.can_hit_none.all() and mo.ally_fighting.all()
    assert "too_passive" in kinds(detect(m, m.me, mo, CFG))
    m, mo = run(view_bound=3.0)                    # could be 3 tiles away: undecided
    assert not mo.can_hit_none.any()
    assert "too_passive" not in kinds(detect(m, m.me, mo, CFG))


def test_good_spacing_just_outside_range():
    m = match([player("me", "ally", 10.0, 16, me=True), player("mate", "ally", 9.0, 16),
               player("piper", "enemy", 20.0, 16, rng=9.0)])       # 10 tiles: 1 outside her range
    good = [e for e in detect(m, m.me, moments(m, m.me), CFG) if e.type == "good_spacing"]
    assert len(good) == 1 and good[0].players == ["me", "piper"]
    assert good[0].measurements["distance_min"] == pytest.approx(10.0)


def test_isolated_before_death():
    def run(mate_x):
        me = player("me", "ally", 10.0, 16, me=True)
        me.alive = T < 10.0
        m = match([me, player("mate", "ally", mate_x, 16), player("foe", "enemy", 15.0, 16)],
                  deaths=[Death("me", 10.0, 30.0, "bar")])
        return [k for k in kinds(detect(m, m.me, moments(m, m.me), CFG)) if k in ("eliminated", "isolated_before_death")]
    assert run(mate_x=4.0) == ["isolated_before_death", "eliminated"]   # 6 tiles away the whole time
    assert run(mate_x=8.0) == ["eliminated"]                            # 2 tiles away: supported


def test_rounds_and_deaths_from_slot_timeline(tmp_path):
    hud = tmp_path / "hud.csv"
    lines = ["video_time,ally1,ally2,ally3,enemy1,enemy2,enemy3"]
    for i in range(80):
        vt = i * 0.5
        if vt < 2:
            s = "0,0,0,0,0,0"          # VS screen
        elif vt < 10:
            s = "1,1,1,1,1,1"
        elif vt < 20:
            s = "1,0,1,1,1,1"          # ally2 down at 10 s
        elif vt < 25:
            s = "1,0,1,0,0,0"          # all enemies down at 20 s
        else:
            s = "1,1,1,1,1,1"          # round 2 at 25 s
        lines.append(f"{vt},{s}")
    hud.write_text("\n".join(lines) + "\n")
    clock = [{"video_time": str(i * 0.1), "match_time": str(i * 0.1 + 5), "source": "bar"} for i in range(400)]
    players = [Player("me", "you", "ally", True, "X", 9, 1, *(np.zeros(1),) * 2, np.array(["observed"])),
               Player("e1", "e1", "enemy", False, "X", 9, 0, *(np.zeros(1),) * 2, np.array(["observed"]))]
    rounds, deaths = rounds_and_deaths(hud, clock, players)
    assert rounds == [(pytest.approx(7.0), pytest.approx(30.0)), (pytest.approx(30.0), pytest.approx(44.5))]
    assert [(d.player, round(d.match_time, 1)) for d in deaths] == [("me", 15.0), ("e1", 25.0)]


def test_walls_block_shots_and_throwers_lob_over():
    ref = load("streets_with_no_name")
    # row 7 has walls at columns 4-8
    assert not ref.clear_shot(6.5, 5.5, 6.5, 9.5)[0]
    assert ref.clear_shot(2.5, 5.5, 2.5, 9.5)[0]
    me = player("me", "ally", 6.5, 9.5, me=True)
    foe = player("foe", "enemy", 6.5, 5.5, rng=9.0)
    m = match([me, player("mate", "ally", 15.5, 30.5), foe])
    assert not any(moments(m, m.me).in_range_of)          # in range, but behind the wall
    foe.lobs = True
    assert all(x == ["foe"] for x in moments(m, m.me).in_range_of)


def test_attack_axis_follows_the_map_not_the_diagonal():
    om, sw = load("opening_move"), load("streets_with_no_name")
    # Opening Move: spawns in opposite corners, play runs up the map
    assert om.attack_axis(np.array([3.0, 30.0]), np.array([17.0, 2.5])).tolist() == [0.0, -1.0]
    # Streets: spawns left and right
    assert sw.attack_axis(np.array([1.5, 17.5]), np.array([19.5, 15.5])).tolist() == [1.0, 0.0]


def test_crossfire_and_supported_fire():
    # enemy at the centre, you below it, teammate to its side: both can shoot it, ~90° apart
    me = player("me", "ally", 10.5, 22.5, me=True)
    mate = player("mate", "ally", 4.5, 17.5)
    foe = player("foe", "enemy", 10.5, 17.5, rng=9.0)
    m = match([me, mate, foe])
    mo = moments(m, m.me)
    assert mo.in_range_of[0] == ["foe"] and mo.unsupported[0] == []      # its shots can be traded back
    assert mo.crossfire[0][0][:2] == ("foe", "mate") and mo.crossfire[0][0][2] >= 45
    cfg = dict(CFG, crossfire={"min_s": 1.0})
    assert "crossfire" in kinds(detect(m, m.me, mo, cfg))


def test_enemy_who_goes_out_of_sight_is_remembered():
    # Enemy seen 4 tiles in front, then disappears (bush) for the rest of the match; you then push forward.
    seen = T < 2.0
    foe = player("foe", "enemy", 14.0, 16.5, rng=6.0, seen=seen)
    me_x = np.where(T < 2.0, 6.0, 9.0)                      # 5 tiles from where the enemy vanished
    m = match([player("me", "ally", me_x, 16.5, me=True), player("mate", "ally", 4.0, 16.5), foe],
              view_bound=0.0)                              # the view doesn't rule anything out (live footage)
    m.view_rules_out = False
    mo = moments(m, m.me)
    j = int(np.argmin(np.abs(T - 2.5)))
    assert mo.hidden_threats[j][0][:2] == ("foe", 0.6)      # last seen at 1.9 s, 5 tiles away: 5 - 0.6 s * 2.4 <= 6 range
    late = int(np.argmin(np.abs(T - 9.0)))
    assert mo.hidden_threats[late] == []                    # too long ago: no longer used
    cfg = dict(CFG, risky_push={"ahead_tiles": 3.0, "min_s": 0.5})
    assert "risky_push" in kinds(detect(m, m.me, mo, cfg))
