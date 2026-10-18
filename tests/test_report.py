import numpy as np

from pipeline.events import Event
from pipeline.report import clock_text, sentence, takeaways
from tests.test_events import CFG, T, match, player


def m3():
    return match([player("me", "ally", 10.0, 16, me=True), player("mate", "ally", 5.0, 16),
                  player("foe", "enemy", 17.0, 16)])


def test_clock_text():
    assert clock_text(0) == "0:00" and clock_text(75.9) == "1:15" and clock_text(-1) == "0:00"


def test_sentences_only_use_measurements():
    m = m3()
    m.players[2].brawler = "Piper"
    m.players[2].label = "Marvin"
    over = Event("overextended", 5.0, 7.9, ["me", "foe"],
                 dict(ahead_mean=5.2, ahead_max=7.0, nearest_ally=["mate"], enemies_in_range=["foe"]))
    s = sentence(m, over, CFG)
    assert "5 tiles ahead" in s and "Piper having a clear shot" in s and "Marvin" not in s and "for 3 s" in s
    out = Event("outnumbered", 1, 2, ["me", "foe"], dict(enemies_in_range_max=2))
    assert "2 enemies had a clear shot at you (Piper)" in sentence(m, out, CFG)
    good = Event("good_spacing", 0, 5.9, ["me", "foe"], dict(enemy="foe", enemy_range=9.0, distance_min=9.6, distance_max=10.4))
    assert "just outside their 9-tile range, for 6 seconds" in sentence(m, good, CFG)


def test_takeaways_rank_and_count():
    m = m3()
    evs = [Event("overextended", a, a + 2.9, ["me"], dict(ahead_mean=6.0, ahead_max=8.0)) for a in (1, 6, 11)]
    evs += [Event("eliminated", 15, 15, ["me"], dict(clock="bar")),
            Event("isolated_before_death", 10, 15, ["me"], dict(isolated_s=4.0))]
    metrics = dict(alive_s=20.0, time_in_enemy_range_s=4.0)
    out = takeaways(m, evs, metrics, CFG)
    assert out[0] == "Your only elimination came after taking fire your team couldn't return."   # a death outranks 9 s overextended
    assert out[1].startswith("You overextended 3 times (9 s in all)")
    assert len(out) == 3
