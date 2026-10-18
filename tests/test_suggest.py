import numpy as np

from pipeline.analyze import moments
from pipeline.events import Event
from pipeline.suggest import DEFAULTS, reachable, suggestions
from tests.test_events import match, player


def test_reachable_respects_walls_and_distance():
    m = match([player("me", "ally", 10.5, 16.5, me=True)])
    tiles = reachable(m, (10.5, 16.5), 3.6)
    assert (10, 16) in tiles and tiles[(10, 16)][0] == 0.0
    assert all(d <= 3.6 for d, _ in tiles.values())
    assert all(m.ref.layout[r][c] in ".Bbr" for c, r in tiles)
    assert (10, 13) in tiles and (10, 12) not in tiles      # obstacle at row 12 blocks the way


def test_suggests_stepping_out_of_range_towards_team():
    # you at x=10, teammate behind at x=5, enemy with range 9 at x=18.5: 8.5 tiles, in range
    me = player("me", "ally", 10.5, 16.5, me=True, rng=10.0)
    m = match([me, player("mate", "ally", 5.5, 16.5), player("foe", "enemy", 18.5, 16.5, rng=9.0)])
    ev = Event("overextended", 0.0, 2.0, ["me", "foe"], {})
    (s,) = suggestions(m, m.me, moments(m, m.me), [ev])
    assert s.current.threats == ["foe"] and s.better.threats == []
    assert s.better.x < 10.5                                 # moved back, towards the team
    assert s.better.in_reach == ["foe"]                      # still within own 10-tile range
    assert s.better.path <= DEFAULTS["speed_tiles_per_s"] * DEFAULTS["horizon_s"]


def test_no_suggestion_when_already_fine():
    m = match([player("me", "ally", 10.5, 16.5, me=True), player("mate", "ally", 9.5, 16.5),
               player("foe", "enemy", 20.5, 16.5, rng=9.0)])
    ev = Event("overextended", 0.0, 2.0, ["me"], {})
    assert suggestions(m, m.me, moments(m, m.me), [ev]) == []
