import numpy as np

from pipeline.track import Piece, link_identities, similar, vote


def piece(tid, team, frames, name=None, fp=None):
    frames = list(frames)
    return Piece(tid, team, dets=[tid * 1000 + f for f in frames], frames=set(frames), name=name, fingerprint=fp)


def positions(pieces, where):
    """Detection index -> (time, x, y): each piece stands still at where[tid]."""
    return {d: (f / 10, *where[p.track_id]) for p in pieces for d, f in zip(p.dets, sorted(p.frames))}


def test_name_reads():
    assert similar("TennisToner10", "TennisTonerloa") > 0.75
    assert similar("Marvin", "SurgeIsHim14") < 0.4
    assert vote([("LOUIHBOOSHES", 0.97), ("LOUIHBOOSHE5", 0.9), ("7030", 0.9)]) == "LOUIHBOOSHES"
    assert vote([("Xroma", 0.7)]) is None                     # one read isn't enough
    assert vote([("ab", 0.99), ("ab", 0.99)]) is None         # too short to trust


def test_pieces_link_by_name_and_find_me():
    ps = [piece(1, "ally", range(0, 50), "LOUIHBOOSHES"), piece(2, "ally", range(60, 90), "LOUIHB00SHES"),
          piece(3, "ally", range(0, 40), "Xroma"), piece(4, "enemy", range(0, 40), "Marvin")]
    ids, un = link_identities(ps, "LOUIHBOOSHES")
    me = [i for i in ids if i.is_me]
    assert len(me) == 1 and {p.track_id for p in me[0].pieces} == {1, 2}
    assert un == []


def test_misread_name_on_wrong_team_is_dropped():
    # an enemy crop caught the ally's name tag standing next to it
    rng = np.random.default_rng(0)
    ps = [piece(1, "ally", range(0, 80), "LOUIHBOOSHES"), piece(2, "enemy", range(0, 15), "LOUIHBOOSHES", rng.random((8, 48))),
          piece(3, "enemy", range(0, 40), "Marvin")]
    ids, _ = link_identities(ps, "LOUIHBOOSHES")
    names = {i.team: i.name for i in ids if i.name}
    assert names == {"ally": "LOUIHBOOSHES", "enemy": "Marvin"}
    assert not any(i.team == "enemy" and i.name == "LOUIHBOOSHES" for i in ids)


def test_elimination_and_spawned_objects():
    rng = np.random.default_rng(1)
    fp = lambda: rng.random((8, 48))
    ps = [piece(1, "enemy", range(0, 100), "Marvin"), piece(2, "enemy", range(0, 100), "Con991"),
          piece(3, "enemy", range(0, 30), None, fp()),           # unreadable name, starts a 3rd player
          piece(4, "enemy", range(50, 90), None, fp()),          # on screen with 1 and 2: must be player 3
          piece(5, "enemy", range(0, 100), None, None)]          # health bar, no name: a spawned object
    pos = positions(ps, {1: (2, 2), 2: (5, 5), 3: (10, 10), 4: (11, 11), 5: (8, 3)})
    ids, un = link_identities(ps, None, pos)
    third = [i for i in ids if i.name is None]
    assert len(ids) == 3 and len(third) == 1
    assert {p.track_id for p in third[0].pieces} == {3, 4}
    assert [p.track_id for p in un] == [5]


def test_movement_decides_between_free_players():
    rng = np.random.default_rng(2)
    ps = [piece(1, "ally", range(0, 100), "LOUIHBOOSHES"),
          piece(2, "ally", range(0, 30), "Xroma"), piece(3, "ally", range(0, 30), "L21Emi"),
          piece(4, "ally", range(35, 60), None, rng.random((8, 48)))]   # unread, after both left the screen
    pos = positions(ps, {1: (10, 10), 2: (2, 20), 3: (18, 5), 4: (17, 6)})
    ids, un = link_identities(ps, "LOUIHBOOSHES", pos)
    owner = next(i for i in ids if any(p.track_id == 4 for p in i.pieces))
    assert owner.name == "L21Emi"      # 1 tile from where L21Emi was last seen; Xroma is 20 away


def test_name_cleaning():
    from pipeline.track import clean_read, plausible_name
    assert clean_read("6500MPBOBA") == "MPBOBA"
    assert clean_read("ThatDudei3800") == "ThatDudei"
    assert not plausible_name("6900")
    assert plausible_name("大雷")            # short, but not Latin script
    assert not plausible_name("ab")


def test_tracker_switch_is_split():
    # one tracker id follows an enemy, then jumps onto a different brawler 6 tiles away
    from pipeline.detect import Detection
    from pipeline.track import build_pieces
    dets = [Detection(f, f / 10, 7, "enemy", 0.9, 0, 0, 10, 10, None) for f in range(20)]
    pos = {i: (i / 10, 3.0 if i < 10 else 9.0, 5.0) for i in range(20)}
    pieces = build_pieces(dets, {}, {}, pos)
    assert sorted(len(p.dets) for p in pieces) == [10, 10]
    # and a team change under the same id is split too
    dets2 = [Detection(f, f / 10, 7, "enemy" if f < 5 else "ally", 0.9, 0, 0, 10, 10, None) for f in range(10)]
    assert sorted((p.team, len(p.dets)) for p in build_pieces(dets2, {}, {})) == [("ally", 5), ("enemy", 5)]


def test_same_name_tolerates_garbled_separators():
    from pipeline.track import same_name
    assert same_name("PBOBA", "SMPIBOBA")
    assert same_name("Tadeo", "HgiLTadeo")
    assert same_name("TennisTonerl0", "TennisTonerlo")
    assert not same_name("SMPJAEMIN", "SMPIBOBA")
    assert not same_name("Marvin", "Maalox")
