import numpy as np

from pipeline.brawlers import hud_slots, parse_class, parse_range
from pipeline.track import Identity, assign_brawlers


def test_parse_wiki_fields():
    text = "{{Attack|AttackRange = 9 (Long)<br/>10 (with Magnum Special)|SuperRange = 11 (Very Long)}}|Class = [[Damage Dealer]]"
    assert parse_range(text) == (9.0, "9 (Long)")
    assert parse_range("|AttackRange = 7.67 (Long)|") == (7.67, "7.67 (Long)")
    assert parse_range("no range here") == (None, None)
    assert parse_class(text) == "Damage Dealer"


def test_hud_slots_follow_screen_edges():
    a = hud_slots(1562, 720)
    assert a["ally"][0][0] == 105 and a["enemy"][2][2] == 1562 - 76 + 52 - 3
    wide = hud_slots(2000, 720)                         # wider phone: enemies stay at the right edge
    assert wide["ally"] == a["ally"]
    assert wide["enemy"][0][0] - a["enemy"][0][0] == 2000 - 1562


def roster(ally, enemy):
    return {"ally": [{"brawler": b} for b in ally], "enemy": [{"brawler": b} for b in enemy]}


def test_assign_from_kill_feed_then_elimination():
    ids = [Identity("ally1", "ally", "LOUIHBOOSHES"), Identity("ally2", "ally", "L21Emi"),
           Identity("ally3", "ally", "SMOKE"), Identity("enemy1", "enemy", "mugiwara"),
           Identity("enemy2", "enemy", None), Identity("enemy3", "enemy", "Silvio")]
    hits = [(1.0, "LOUIHBOOSHES", "ally", 1, 0.8)] * 5 + [(2.0, "SMOKE", "ally", 2, 0.7),
            (3.0, "mugiwara", "enemy", 1, 0.9), (4.0, "silvio", "enemy", 0, 0.9),
            (5.0, "LOUIHBOOSHES", "ally", 0, 0.5)]      # one stray vote loses to five
    assign_brawlers(ids, roster(["Gus", "Pierce", "Leon"], ["Belle", "Kit", "Ruffs"]), hits)
    got = {i.id: (i.brawler, i.brawler_source) for i in ids}
    assert got["ally1"] == ("Pierce", "kill_feed")
    assert got["ally3"] == ("Leon", "kill_feed")
    assert got["ally2"] == ("Gus", "elimination")
    assert got["enemy1"] == ("Kit", "kill_feed") and got["enemy3"] == ("Belle", "kill_feed")
    assert got["enemy2"] == ("Ruffs", "elimination")    # unnamed player, last slot left


def test_no_guess_when_two_are_left():
    ids = [Identity("ally1", "ally", "A"), Identity("ally2", "ally", "B"), Identity("ally3", "ally", "C")]
    assign_brawlers(ids, roster(["Gus", "Pierce", "Leon"], ["Belle", "Kit", "Ruffs"]),
                    [(1.0, "A", "ally", 0, 0.9)])
    assert [i.brawler for i in ids] == ["Gus", None, None]
