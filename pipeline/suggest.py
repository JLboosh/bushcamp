"""Positioning suggestions: where you could have stood instead, computed from the data.

For each problem moment (overextended, outnumbered, isolated before an
elimination, too passive) this takes every tile you could have walked to in
about 1.5 s (walls and water block the way) and scores it with what was
measured at that moment:

    threat     enemies with the tile in range AND a clear shot (walls block,
               throwers lob over); much worse if no teammate can shoot back
               at that enemy (an untradeable hit)
    useful     an enemy you'd have in range with a clear shot     (keep)
    crossfire  ...that a teammate is also shooting from 45°+ away (bonus:
               the enemy can't hide from both; this is "creating space")
    cover      standing in a bush or next to a wall                (small bonus)
    distance   how far you'd have to move                          (prefer small moves)

Staying close to teammates is not rewarded by itself: spreading out to make
angles is often right. What matters is that fire on you can be returned.

The best tile is suggested only if it's clearly better than where you were,
with the reasons spelled out from the same numbers. It's a snapshot: enemies
would have reacted, and walls blocking shots aren't modelled yet.

Weights and limits: data/analysis.json "suggest".
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from pipeline.analyze import Match, Moments, Player
from pipeline.events import Event

DEFAULTS = dict(
    speed_tiles_per_s=2.4,      # normal brawler move speed (720 units/s, 300 units per tile)
    horizon_s=1.5,              # how far ahead a correction is looked for
    safety_margin=0.5,          # tiles outside an enemy's range that count as safe
    w_threat=10.0, w_threat_traded=4.0, w_hidden=5.0, w_edge=3.0, w_useful=3.0, w_useful_gap=0.5, w_crossfire=2.0,
    w_bush=0.5, w_cover=0.5, w_move=0.3,
    min_gain=2.0,               # score improvement needed before suggesting anything
)
PASSABLE = set(".Bbr")
FOR_EVENTS = ("overextended", "outnumbered", "isolated_before_death", "too_passive", "risky_push")


@dataclass
class Spot:
    x: float
    y: float
    path: float                         # walking distance (tiles)
    threats: list[str]                  # enemies with this spot in range and a clear shot
    untraded: list[str]                 # ...that none of your teammates can shoot back at
    edges: list[str]                    # enemies within safety_margin of having it in range (clear shot)
    in_reach: list[str]                 # enemies you'd have in range with a clear shot
    crossfire: list[tuple[str, str]]    # (enemy, teammate) you'd be shooting from 45°+ apart
    hidden: list[str]                   # enemies out of sight who could have walked into range of this spot
    nearest_ally: float | None
    nearest_ally_id: str | None
    ahead: float | None
    bush: bool
    cover: bool
    score: float = 0.0


@dataclass
class Suggestion:
    t: float
    event: str                          # the event it answers
    current: Spot
    better: Spot
    route: list[tuple[float, float]] = field(default_factory=list)
    enemies_off_screen: int = 0         # alive enemies whose position wasn't known


def reachable(m: Match, start: tuple[float, float], limit: float) -> dict[tuple[int, int], tuple[float, tuple | None]]:
    """Walking distance (tiles) to every tile within `limit`, 8-way moves, no corner cutting."""
    ref = m.ref
    sc, sr = int(np.floor(start[0])), int(np.floor(start[1]))
    if not (0 <= sc < ref.cols and 0 <= sr < ref.rows):
        return {}
    ok = lambda c, r: 0 <= c < ref.cols and 0 <= r < ref.rows and ref.layout[r][c] in PASSABLE
    first = float(np.hypot(sc + 0.5 - start[0], sr + 0.5 - start[1]))
    best = {(sc, sr): (first, None)}
    heap = [(first, sc, sr)]
    while heap:
        d, c, r = heapq.heappop(heap)
        if d > best[(c, r)][0]:
            continue
        for dc in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if (dc, dr) == (0, 0):
                    continue
                nc, nr = c + dc, r + dr
                if not ok(nc, nr) or (dc and dr and not (ok(c + dc, r) and ok(c, r + dr))):
                    continue
                nd = d + (1.4142 if dc and dr else 1.0)
                if nd <= limit and nd < best.get((nc, nr), (np.inf,))[0]:
                    best[(nc, nr)] = (nd, (c, r))
                    heapq.heappush(heap, (nd, nc, nr))
    return best


def _route(tiles: dict, end: tuple[int, int], start: tuple[float, float]) -> list[tuple[float, float]]:
    out, k = [], end
    while k is not None:
        out.append((k[0] + 0.5, k[1] + 0.5))
        k = tiles[k][1]
    out.append(start)
    return out[::-1]


def last_sightings(m: Match, who: Player, j: int) -> list[tuple[Player, float, float, float]]:
    """Enemies alive but not seen at step j: (enemy, seconds since seen, last x, last y), within memory."""
    from pipeline.analyze import MEMORY_S
    out = []
    back = int(round(MEMORY_S / m.step()))
    for e in m.players:
        if e.team == who.team or not e.alive[j] or e.known[j] or e.range is None:
            continue
        for k in range(j - 1, max(-1, j - back - 1), -1):
            if not e.alive[k]:
                break
            if e.known[k]:
                out.append((e, (j - k) * m.step(), float(e.x[k]), float(e.y[k])))
                break
    return out


def evaluate(m: Match, who: Player, j: int, x: float, y: float, path: float, cfg: dict,
             ghosts: list | None = None) -> Spot:
    from pipeline.analyze import CROSSFIRE_DEG, MOVE_SPEED
    enemies = [e for e in m.players if e.team != who.team and e.alive[j] and e.known[j]]
    allies = [a for a in m.players if a.team == who.team and a.id != who.id and a.alive[j] and a.known[j]]
    ref = m.ref

    def shot(ax, ay, bx, by, lobs):
        return lobs or bool(ref.clear_shot(ax, ay, bx, by)[0])

    # Which enemies a teammate can already shoot (doesn't depend on where you stand).
    covered = {e.id: [a.id for a in allies if a.range is not None
                      and np.hypot(a.x[j] - e.x[j], a.y[j] - e.y[j]) <= a.range
                      and shot(a.x[j], a.y[j], e.x[j], e.y[j], a.lobs)] for e in enemies}
    threats, edges, reach, cross = [], [], [], []
    for e in enemies:
        d = float(np.hypot(e.x[j] - x, e.y[j] - y))
        clear = shot(e.x[j], e.y[j], x, y, False)
        if e.range is not None and (clear or e.lobs):
            if d <= e.range:
                threats.append(e.id)
            elif d <= e.range + cfg["safety_margin"]:
                edges.append(e.id)
        if who.range is not None and d <= who.range and (clear or who.lobs):
            reach.append(e.id)
            for mate in covered[e.id]:
                a = next(p for p in allies if p.id == mate)
                v1 = np.array([x - e.x[j], y - e.y[j]])
                v2 = np.array([a.x[j] - e.x[j], a.y[j] - e.y[j]])
                cos = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
                if np.degrees(np.arccos(np.clip(cos, -1, 1))) >= CROSSFIRE_DEG:
                    cross.append((e.id, mate))
    untraded = [e for e in threats if not covered[e]]
    hidden = [e.id for e, age, lx, ly in (ghosts or [])
              if np.hypot(x - lx, y - ly) - MOVE_SPEED * age <= e.range]
    na, na_id = None, None
    for a in allies:
        d = float(np.hypot(a.x[j] - x, a.y[j] - y))
        if na is None or d < na:
            na, na_id = d, a.id
    ahead = None
    if na_id:
        mate = next(a for a in allies if a.id == na_id)
        ahead = float((np.array([x, y]) - np.array([mate.x[j], mate.y[j]])) @ m.forward)
    c, r = int(np.floor(x)), int(np.floor(y))
    cover = any(ref.tile(c + dc + 0.5, r + dr + 0.5) in ("W", "O") for dc in (-1, 0, 1) for dr in (-1, 0, 1))
    spot = Spot(round(x, 2), round(y, 2), round(path, 2), threats, untraded, edges, reach, cross, hidden,
                None if na is None else round(na, 2), na_id, None if ahead is None else round(ahead, 2),
                ref.tile(x, y) == "B", cover)
    spot.score = score(spot, m, who, j, x, y, enemies, cfg)
    return spot


def score(s: Spot, m: Match, who: Player, j: int, x: float, y: float, enemies: list[Player], cfg: dict) -> float:
    traded = len(s.threats) - len(s.untraded)
    v = -cfg["w_threat"] * len(s.untraded) - cfg["w_threat_traded"] * traded - cfg["w_edge"] * len(s.edges)
    v -= cfg["w_hidden"] * len(s.hidden)
    if s.in_reach:
        v += cfg["w_useful"]
    elif enemies and who.range is not None:
        gap = min(np.hypot(e.x[j] - x, e.y[j] - y) for e in enemies) - who.range
        v -= cfg["w_useful_gap"] * max(0.0, gap)
    if s.crossfire:
        v += cfg["w_crossfire"]
    v += cfg["w_bush"] * s.bush + cfg["w_cover"] * s.cover - cfg["w_move"] * s.path
    return round(float(v), 3)


def suggest_at(m: Match, who: Player, j: int, event: str, cfg: dict) -> Suggestion | None:
    if not (who.alive[j] and who.known[j]):
        return None
    here = (float(who.x[j]), float(who.y[j]))
    ghosts = last_sightings(m, who, j)
    current = evaluate(m, who, j, *here, 0.0, cfg, ghosts)
    tiles = reachable(m, here, cfg["speed_tiles_per_s"] * cfg["horizon_s"])
    best, best_key = current, None
    for (c, r), (d, _) in tiles.items():
        s = evaluate(m, who, j, c + 0.5, r + 0.5, d, cfg, ghosts)
        if s.score > best.score:
            best, best_key = s, (c, r)
    if best_key is None or best.score - current.score < cfg["min_gain"]:
        return None
    # The fix has to address the problem the event was about.
    if event in ("overextended", "outnumbered", "isolated_before_death") and len(best.untraded) >= len(current.untraded):
        return None
    if event == "too_passive" and not best.in_reach:
        return None
    if event == "risky_push" and len(best.hidden) >= len(current.hidden):
        return None
    remembered = {e.id for e, *_ in ghosts}
    off = sum(1 for e in m.players if e.team != who.team and e.alive[j] and not e.known[j] and e.id not in remembered)
    return Suggestion(round(float(m.t[j]), 2), event, current, best, _route(tiles, best_key, here), off)


def suggestions(m: Match, who: Player, mo: Moments, events: list[Event], cfg: dict | None = None) -> list[Suggestion]:
    cfg = {**DEFAULTS, **(cfg or {})}
    out = []
    for e in events:
        if e.type not in FOR_EVENTS:
            continue
        if out and e.start - out[-1].t < 1.0:
            continue                    # one suggestion per moment (events often start together)
        j = int(np.argmin(np.abs(m.t - e.start)))
        s = suggest_at(m, who, j, e.type, cfg)
        if s:
            out.append(s)
    return out
