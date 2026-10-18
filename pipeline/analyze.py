"""Positioning metrics for one match, computed from the tracks (no language model).

Inputs (runs/match/<video>/, written by pipeline/track.py):
    tracks.csv, identities.json   players' positions in tiles, brawlers, ranges
    clock.csv, poses.csv, camera.json   match time and the camera's view per frame
    hud.csv                        top-bar slot states: eliminations and rounds

Two things make the numbers honest about what isn't seen:
- Unknown positions stay unknown; every figure says how much of the time it
  could be measured.
- Off screen is information. In replays bushes don't hide anyone, so an enemy
  who isn't on screen is outside the camera's view: at least as far away as
  the nearest edge of the view. That's often enough to say "out of range".

All distances are in tiles, like in-game ranges.

    python -m pipeline.analyze runs/match/<video> --map streets_with_no_name
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pipeline import localize
from pipeline.gaps import INTERPOLATED, OBSERVED
from pipeline.maps import MapRef, load as load_map

CROSSFIRE_DEG = 45.0       # angle between two shooters (seen from the target) that counts as an angle
MOVE_SPEED = 2.4           # tiles/s a brawler walks (720 units/s, 300 per tile)
MEMORY_S = 4.0             # how long a last sighting is used before it's too old to say anything

# Part of the screen where brawlers are reliably detected: the top bar and the
# bottom controls hide them, so "not detected" only means "not there" inside this.
VIEW_TOP, VIEW_BOTTOM = 0.11, 0.85
VIEW_MARGIN = 0.5          # tiles: a brawler half off the edge may be missed


@dataclass
class Player:
    id: str
    label: str
    team: str
    is_me: bool
    brawler: str | None
    range: float | None
    slot: int | None
    x: np.ndarray
    y: np.ndarray
    flag: np.ndarray
    alive: np.ndarray = field(default=None)       # bool per time step
    lobs: bool = False                              # attacks go over walls (throwers)

    @property
    def known(self) -> np.ndarray:
        return (self.flag == OBSERVED) | (self.flag == INTERPOLATED)


@dataclass
class Death:
    player: str
    match_time: float
    video_time: float
    clock: str               # clock source at that moment (how sure the time is)


@dataclass
class Match:
    t: np.ndarray
    players: list[Player]
    ref: MapRef
    rounds: list[tuple[float, float]]
    deaths: list[Death]
    view_bound: dict[str, np.ndarray]    # player id -> distance (tiles) from them to the view's edge; NaN = no view
    forward: np.ndarray                  # unit vector along the map's direction of play, towards the enemy side
    origin: np.ndarray                   # own spawn centre
    # Does "not on screen" prove "outside the camera's view"? In replays yes (bushes
    # don't hide anyone); in live footage no (an enemy in a bush is invisible).
    view_rules_out: bool = True

    @property
    def me(self) -> Player | None:
        return next((p for p in self.players if p.is_me), None)

    def team(self, team: str, alive_only: bool = True) -> list[Player]:
        return [p for p in self.players if p.team == team]

    def step(self) -> float:
        return float(self.t[1] - self.t[0]) if len(self.t) > 1 else 0.1


# --- loading ----------------------------------------------------------------------

def _float(v: str) -> float:
    return float(v) if v not in ("", "None") else np.nan


def load_match(folder: Path, map_id: str) -> Match:
    ref = load_map(map_id)
    summary = json.loads((folder / "identities.json").read_text())
    info = {i["id"]: i for i in summary["identities"]}

    rows: dict[str, list[dict]] = {}
    with (folder / "tracks.csv").open() as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["identity"], []).append(r)
    t = np.array([float(r["match_time"]) for r in next(iter(rows.values()))])
    players = []
    for pid, rs in rows.items():
        i = info[pid]
        players.append(Player(pid, rs[0]["label"], i["team"], i["is_me"], i.get("brawler"), i.get("attack_range"),
                              i.get("slot"), np.array([_float(r["x"]) for r in rs]),
                              np.array([_float(r["y"]) for r in rs]), np.array([r["flag"] for r in rs])))

    clock = list(csv.DictReader((folder / "clock.csv").open()))
    rounds, deaths = rounds_and_deaths(folder / "hud.csv", clock, players)
    for p in players:
        p.alive = alive_mask(t, p, rounds, deaths)

    # Forward direction: from the spawn the allies start at to the other one.
    side = summary.get("ally_spawn") or "b"
    spawn = {s: np.array([(c + 0.5, r + 0.5) for r, row in enumerate(ref.layout)
                          for c, ch in enumerate(row) if ch == s]).mean(axis=0) for s in "br"}
    own, other = spawn[side], spawn["r" if side == "b" else "b"]
    fwd = ref.attack_axis(own, other)
    from pipeline.brawlers import load as load_brawlers
    table = load_brawlers()
    for p in players:
        info_b = table.get(p.brawler or "", {})
        p.lobs = bool(info_b.get("over_walls", info_b.get("cls") == "Artillery"))

    return Match(t, players, ref, rounds, deaths, view_bounds(folder, t, players, ref, clock), fwd, own)


def rounds_and_deaths(hud_csv: Path, clock: list[dict], players: list[Player],
                      persist: int = 2) -> tuple[list[tuple[float, float]], list[Death]]:
    """Rounds (match-time start, end) and eliminations from the top-bar slot states.

    A slot crossed out for `persist` samples in a row (1 s at 2 fps) is an
    elimination; all six clear again after eliminations is a new round.
    Gameplay starts at the first moment all six slots are clear.
    """
    rows = list(csv.DictReader(hud_csv.open()))
    keys = ["ally1", "ally2", "ally3", "enemy1", "enemy2", "enemy3"]
    vt = np.array([float(r["video_time"]) for r in rows])
    clear = np.array([[r[k] == "1" for k in keys] for r in rows])
    cvt = np.array([float(c["video_time"]) for c in clock])

    def at(v):
        c = clock[int(np.argmin(np.abs(cvt - v)))]
        return (_float(c["match_time"]), c["source"])

    start = next((i for i in range(len(rows)) if clear[i].all()), None)
    if start is None:
        return [], []
    slot_owner = {(p.team, p.slot): p.id for p in players if p.slot is not None}
    rounds, deaths = [], []
    round_start = at(vt[start])[0]
    down = np.zeros(6, bool)
    i = start
    while i < len(rows):
        if down.any() and clear[i:i + persist].all() and len(clear[i:i + persist]) == persist:
            mt = at(vt[i])[0]
            rounds.append((round_start, mt))
            round_start, down = mt, np.zeros(6, bool)
        for k in range(6):
            if not down[k] and clear[i - 1, k] and not clear[i:i + persist, k].any() and len(clear[i:i + persist]) == persist:
                down[k] = True
                team, slot = ("ally", k) if k < 3 else ("enemy", k - 3)
                if (team, slot) in slot_owner:
                    mt, src = at(vt[i])
                    deaths.append(Death(slot_owner[(team, slot)], mt, float(vt[i]), src))
        i += 1
    rounds.append((round_start, at(vt[-1])[0]))
    return rounds, deaths


def alive_mask(t: np.ndarray, p: Player, rounds: list[tuple[float, float]], deaths: list[Death]) -> np.ndarray:
    """Alive from each round's start until eliminated in that round."""
    alive = np.zeros(len(t), bool)
    for r0, r1 in rounds:
        end = min([d.match_time for d in deaths if d.player == p.id and r0 <= d.match_time <= r1], default=r1)
        alive |= (t >= r0) & (t < end)
    return alive


def view_bounds(folder: Path, t: np.ndarray, players: list[Player], ref: MapRef,
                clock: list[dict]) -> dict[str, np.ndarray]:
    """For each player and time step: how far they are from the edge of the
    camera's reliable view (0 if outside it, NaN if the view isn't known).
    An enemy not seen at that moment is at least this far away."""
    cj = json.loads((folder / "camera.json").read_text())
    cam = localize.Camera(cj["width"], cj["height"], cj["kx"], cj["kv"], cj["p"])
    poses = list(csv.DictReader((folder / "poses.csv").open()))
    corners = np.array([[0, VIEW_TOP * cam.height], [cam.width, VIEW_TOP * cam.height],
                        [cam.width, VIEW_BOTTOM * cam.height], [0, VIEW_BOTTOM * cam.height]])
    views: list[np.ndarray | None] = [None] * len(t)
    step = float(t[1] - t[0]) if len(t) > 1 else 0.1
    for c, p in zip(clock, poses):
        if c["match_time"] == "" or p["status"] != "ok" or c["source"] not in ("bar", "inferred"):
            continue
        j = int(round((float(c["match_time"]) - t[0]) / step))
        if 0 <= j < len(t):
            poly = cam.to_map(corners, float(p["x"]), float(p["y"]))
            if cj["rotated"]:
                poly = np.stack([ref.cols - poly[:, 0], ref.rows - poly[:, 1]], axis=1)
            views[j] = poly
    out = {}
    for pl in players:
        b = np.full(len(t), np.nan)
        for j, poly in enumerate(views):
            if poly is not None and pl.known[j]:
                b[j] = max(0.0, _inside_distance(poly, np.array([pl.x[j], pl.y[j]])) - VIEW_MARGIN)
        out[pl.id] = b
    return out


def _inside_distance(poly: np.ndarray, pt: np.ndarray) -> float:
    """Distance from pt to the polygon's boundary if pt is inside, else 0."""
    import cv2
    d = cv2.pointPolygonTest(poly.astype(np.float32).reshape(-1, 1, 2), (float(pt[0]), float(pt[1])), True)
    return max(0.0, float(d))


# --- per-moment facts ---------------------------------------------------------------

@dataclass
class Moments:
    """Facts about one player at every time step (NaN / None where not measurable)."""
    t: np.ndarray
    known: np.ndarray                 # the player's own position is known and they're alive
    observed: np.ndarray              # ...and actually seen (not interpolated)
    nearest_ally: np.ndarray          # distance to nearest alive teammate (known positions)
    nearest_ally_id: list
    ally_lower_bound: np.ndarray      # every alive teammate is at least this far (seen: exact, unseen: view edge)
    nearest_ally_certain: np.ndarray  # the nearest seen teammate is certainly the nearest one
    nearest_enemy: np.ndarray
    nearest_enemy_id: list
    in_range_of: list[list[str]]      # enemies with the player in range AND a clear shot (or a lob) (certain)
    unsupported: list[list[str]]      # ...of those, enemies none of the teammates can shoot back at (certain)
    can_hit: list[list[str]]          # enemies the player has in range with a clear shot
    crossfire: list[list[tuple]]      # (enemy, teammate, angle°): both have a clear shot at it from different sides
    hidden_threats: list[list[tuple]] # (enemy, seconds since seen, last x, last y, last tile): an enemy out of
                                      # sight who could have moved into range of you since you last saw them
    range_unknown: np.ndarray         # some alive enemy's range state can't be decided
    out_of_all_ranges: np.ndarray     # certainly outside every alive enemy's range
    can_hit_none: np.ndarray          # certainly no alive enemy within own range (+ margin)
    ahead: np.ndarray                 # tiles further forward than the nearest alive teammate
    in_bush: np.ndarray
    near_cover: np.ndarray
    ally_fighting: np.ndarray         # a teammate is trading shots with a seen enemy


def distance(a: Player, b: Player) -> np.ndarray:
    return np.hypot(a.x - b.x, a.y - b.y)


def moments(m: Match, who: Player, cover_tiles: float = 1.0, reach_margin: float = 1.0) -> Moments:
    n = len(m.t)
    alive_known = who.alive & who.known
    allies = [p for p in m.players if p.team == who.team and p.id != who.id]
    enemies = [p for p in m.players if p.team != who.team]

    def nearest(group):
        best = np.full(n, np.nan)
        best_id = [None] * n
        for p in group:
            d = np.where(p.alive & p.known & alive_known, distance(who, p), np.nan)
            better = np.isnan(best) & ~np.isnan(d) | (d < best)
            best = np.where(better, d, best)
            for j in np.flatnonzero(better):
                best_id[j] = p.id
        return best, best_id

    na, na_id = nearest(allies)
    ne, ne_id = nearest(enemies)
    bound = m.view_bound[who.id]

    # Every alive teammate is at least `lower` away: seen ones exactly, unseen
    # ones at least as far as the view's edge. inf = no teammate alive;
    # NaN = an unseen teammate while the view isn't known.
    lower = np.full(n, np.nan)
    certain = np.zeros(n, bool)
    for j in range(n):
        seen, unseen, unknown = [], [], False
        for p in allies:
            if not p.alive[j]:
                continue
            if p.known[j]:
                seen.append(float(np.hypot(who.x[j] - p.x[j], who.y[j] - p.y[j])))
            elif np.isnan(bound[j]):
                unknown = True
            else:
                unseen.append(float(bound[j]))
        if unknown:
            continue
        lower[j] = min(seen + unseen, default=np.inf)
        certain[j] = bool(seen) and min(seen) <= min(unseen, default=np.inf)
    in_range_of = [[] for _ in range(n)]
    can_hit = [[] for _ in range(n)]
    undecided = np.zeros(n, bool)
    all_out = alive_known.copy()
    none_hittable = alive_known.copy()
    own_reach = (who.range or 0.0) + reach_margin
    enemy_fire: dict[str, np.ndarray] = {}           # enemy id -> per step: has a clear shot at us
    for e in enemies:
        r = e.range
        seen = e.alive & e.known
        d = distance(who, e)
        los = m.ref.clear_shot(who.x, who.y, e.x, e.y)
        hits_me = seen & alive_known & (r is not None) & (d <= (r or 0)) & (los | e.lobs)
        enemy_fire[e.id] = hits_me
        mine = seen & alive_known & (who.range is not None) & (d <= (who.range or 0)) & (los | who.lobs)
        for j in np.flatnonzero(hits_me):
            in_range_of[j].append(e.id)
        for j in np.flatnonzero(mine):
            can_hit[j].append(e.id)
        for j in range(n):
            if not alive_known[j] or not e.alive[j]:
                continue
            if seen[j]:
                if r is None or d[j] <= r:
                    all_out[j] = False if r is not None else all_out[j]
                if d[j] <= own_reach:
                    none_hittable[j] = False
            else:
                # Not seen: at least `bound` away (if the view is known).
                lb = bound[j]
                if r is None or np.isnan(lb) or lb < r:
                    all_out[j] = False
                    undecided[j] = True
                if np.isnan(lb) or lb < own_reach:
                    none_hittable[j] = False
    all_out &= alive_known
    none_hittable &= alive_known

    # Support = a teammate can shoot back at the enemy shooting at you.
    # mate_hits[(mate, enemy)]: certainly can; mate_maybe: an unseen teammate can't be ruled out
    # (it's at least `bound` from you, so at least bound - d(you, enemy) from the enemy).
    mate_hits, mate_maybe = {}, {}
    for a in allies:
        for e in enemies:
            both = a.alive & a.known & e.alive & e.known
            d = distance(a, e)
            los = m.ref.clear_shot(a.x, a.y, e.x, e.y)
            mate_hits[(a.id, e.id)] = both & (a.range is not None) & (d <= (a.range or 0)) & (los | a.lobs)
            unseen = a.alive & ~a.known & e.alive & e.known
            gap = bound - distance(who, e)
            mate_maybe[(a.id, e.id)] = unseen & (np.isnan(gap) | (gap <= (a.range or np.inf)))
    unsupported = [[] for _ in range(n)]
    crossfire = [[] for _ in range(n)]
    for e in enemies:
        covered = np.zeros(n, bool)
        maybe = np.zeros(n, bool)
        for a in allies:
            covered |= mate_hits[(a.id, e.id)]
            maybe |= mate_maybe[(a.id, e.id)]
        for j in np.flatnonzero(enemy_fire[e.id] & ~covered & ~maybe):
            unsupported[j].append(e.id)
        # Crossfire: you and a teammate both have a clear shot at e from directions >= 45° apart.
        for a in allies:
            both = mate_hits[(a.id, e.id)]
            for j in np.flatnonzero(both):
                if e.id not in can_hit[j]:
                    continue
                v1 = np.array([who.x[j] - e.x[j], who.y[j] - e.y[j]])
                v2 = np.array([a.x[j] - e.x[j], a.y[j] - e.y[j]])
                cos = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
                ang = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
                if ang >= CROSSFIRE_DEG:
                    crossfire[j].append((e.id, a.id, round(ang)))

    # Forward progress along the map's direction of play.
    prog = lambda p: (p.x - m.origin[0]) * m.forward[0] + (p.y - m.origin[1]) * m.forward[1]
    ahead = np.full(n, np.nan)
    for j in range(n):
        if na_id[j] is not None and certain[j]:
            mate = next(p for p in allies if p.id == na_id[j])
            ahead[j] = prog(who)[j] - prog(mate)[j]

    in_bush = np.zeros(n, bool)
    near_cover = np.zeros(n, bool)
    walls = np.array([(c + 0.5, r + 0.5) for r, row in enumerate(m.ref.layout) for c, ch in enumerate(row) if ch in "WO"])
    for j in np.flatnonzero(alive_known):
        in_bush[j] = m.ref.tile(who.x[j], who.y[j]) == "B"
        if len(walls):
            # distance to the nearest wall tile's edge (square tiles)
            dx = np.maximum(np.abs(walls[:, 0] - who.x[j]) - 0.5, 0)
            dy = np.maximum(np.abs(walls[:, 1] - who.y[j]) - 0.5, 0)
            near_cover[j] = float(np.min(np.hypot(dx, dy))) <= cover_tiles

    fighting = np.zeros(n, bool)
    for a in allies:
        for e in enemies:
            both = a.alive & a.known & e.alive & e.known
            d = distance(a, e)
            reach = max(a.range or 0, e.range or 0)
            fighting |= both & (d <= reach)

    # Map awareness: an enemy who went into a bush or off screen is where you last
    # saw them, plus however far they could have walked since. They're a possible
    # threat if that area reaches within their range of you. (In replays an unseen
    # enemy is also known to be outside the camera's view; that's used when it
    # rules the threat out.)
    hidden = [[] for _ in range(n)]
    for e in enemies:
        if e.range is None:
            continue
        last = None                          # (step, x, y)
        for j in range(n):
            if not e.alive[j]:
                last = None
                continue
            if e.known[j]:
                last = (j, e.x[j], e.y[j])
                continue
            if last is None or not alive_known[j]:
                continue
            age = (j - last[0]) * m.step()
            if age > MEMORY_S:
                continue
            gap = float(np.hypot(who.x[j] - last[1], who.y[j] - last[2])) - MOVE_SPEED * age
            ruled_out = m.view_rules_out and not np.isnan(bound[j]) and bound[j] > e.range
            if gap <= e.range and not ruled_out:
                hidden[j].append((e.id, round(age, 1), round(float(last[1]), 1), round(float(last[2]), 1),
                                  m.ref.tile(last[1], last[2])))

    return Moments(m.t, alive_known, alive_known & (who.flag == OBSERVED), na, na_id,
                   np.where(alive_known, lower, np.nan), certain & alive_known, ne, ne_id, in_range_of,
                   unsupported, can_hit, crossfire, hidden,
                   undecided & alive_known, all_out, none_hittable, ahead, in_bush, near_cover, fighting)


# --- summary metrics ------------------------------------------------------------------

def summarize(m: Match, who: Player, mo: Moments) -> dict:
    step = m.step()
    alive = who.alive
    k = mo.known

    def seconds(mask):
        return round(float(mask.sum() * step), 1)

    def stats(arr):
        v = arr[~np.isnan(arr)]
        return None if not len(v) else dict(mean=round(float(v.mean()), 2), median=round(float(np.median(v)), 2),
                                            max=round(float(v.max()), 2), min=round(float(v.min()), 2),
                                            measured_s=seconds(~np.isnan(arr)))

    in_any = np.array([bool(x) for x in mo.in_range_of])
    heat, xe, ye = np.histogram2d(who.x[k], who.y[k], bins=[m.ref.cols, m.ref.rows],
                                  range=[[0, m.ref.cols], [0, m.ref.rows]])
    my_deaths = [d for d in m.deaths if d.player == who.id]
    return dict(
        player=who.label, brawler=who.brawler, attack_range=who.range,
        alive_s=seconds(alive), position_known_s=seconds(k), position_seen_s=seconds(mo.observed),
        spacing_to_nearest_ally=stats(mo.nearest_ally),
        distance_to_nearest_enemy=stats(mo.nearest_enemy),
        time_in_enemy_range_s=seconds(in_any & k),          # in range with a clear shot (or a lob)
        time_exposed_unsupported_s=seconds(np.array([bool(x) for x in mo.unsupported]) & k),
        time_in_crossfire_s=seconds(np.array([bool(x) for x in mo.crossfire]) & k),
        time_safe_from_all_enemies_s=seconds(mo.out_of_all_ranges),
        time_range_undecided_s=seconds(mo.range_unknown & ~in_any),
        time_in_bush_s=seconds(mo.in_bush), time_near_cover_s=seconds(mo.near_cover),
        heatmap=heat.T.astype(int).tolist(),          # rows = map rows (y), cols = map columns (x)
        deaths=[death_context(m, who, mo, d) for d in my_deaths],
        rounds=[dict(start=round(a, 1), end=round(b, 1)) for a, b in m.rounds],
    )


def death_context(m: Match, who: Player, mo: Moments, d: Death, window: float = 5.0) -> dict:
    """Where the player was and what was around them in the seconds before an elimination."""
    sel = (m.t >= d.match_time - window) & (m.t < d.match_time) & mo.known
    last = np.flatnonzero(mo.known & (m.t < d.match_time))
    pos = (round(float(who.x[last[-1]]), 2), round(float(who.y[last[-1]]), 2)) if len(last) else None
    enemies_close = [set(x) for x, s in zip(mo.in_range_of, sel) if s]
    return dict(
        match_time=round(d.match_time, 1), clock=d.clock, position=pos,
        tile=m.ref.tile(*pos) if pos else None,
        seen_share=round(float(sel.mean() and mo.observed[sel].mean()), 2) if sel.any() else 0.0,
        nearest_ally_mean=None if np.isnan(mo.nearest_ally[sel]).all() else round(float(np.nanmean(mo.nearest_ally[sel])), 2),
        nearest_ally_min=None if np.isnan(mo.nearest_ally[sel]).all() else round(float(np.nanmin(mo.nearest_ally[sel])), 2),
        enemies_in_range_max=max((len(s) for s in enemies_close), default=0),
        enemies_involved=sorted(set().union(*enemies_close)) if enemies_close else [],
        ahead_mean=None if np.isnan(mo.ahead[sel]).all() else round(float(np.nanmean(mo.ahead[sel])), 2),
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Positioning metrics for one match.")
    ap.add_argument("folder", type=Path, help="runs/match/<video>")
    ap.add_argument("--map", required=True)
    args = ap.parse_args(argv)
    m = load_match(args.folder, args.map)
    me = m.me
    if me is None:
        raise SystemExit("the recording player wasn't identified (data/player.json)")
    result = summarize(m, me, moments(m, me))
    (args.folder / "metrics.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "heatmap"}, indent=1))


if __name__ == "__main__":
    main()
