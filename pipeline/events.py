"""Rule-based positioning events, timestamped on match time.

Every rule is a condition checked at each time step (pipeline/analyze.py
Moments) that has to hold for a minimum duration; thresholds live in
data/analysis.json. Each event stores its time span, the players involved,
key measurements, and confidence: the share of the span where the player
was actually seen (not interpolated).

    overextended           more than X tiles ahead of the nearest teammate (along the
                           map's direction of play) while an enemy has a clear shot at
                           you that none of your teammates can return
    outnumbered            2+ enemies have a clear shot at you and no teammate can
                           shoot back at any of them
    isolated_before_death  under fire nobody could return for N of the seconds
                           before an elimination
    risky_push             ahead of the team where an enemy you lost sight of (in a
                           bush or off screen, seen in the last 4 s) could have
                           walked into range of you
    crossfire              (good) you and a teammate both have a clear shot at the
                           same enemy from directions 45°+ apart
    good_spacing           held just outside the nearest enemy's range (within a
                           band) for N+ seconds
    too_passive            out of reach of every enemy for N+ seconds while a
                           teammate was fighting
    eliminated, round_start

    python -m pipeline.events runs/match/<video> --map streets_with_no_name
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from pipeline.analyze import Match, Moments, Player, load_match, moments, summarize

CONFIG = Path(__file__).resolve().parent.parent / "data" / "analysis.json"


@dataclass
class Event:
    type: str
    start: float              # match time (s)
    end: float
    players: list[str]        # ids: the subject first, then others involved
    measurements: dict = field(default_factory=dict)
    confidence: float = 1.0   # share of the span where the subject was seen


def _tiles(v) -> float | str:
    """A distance for the event record; 'no teammate alive' when infinite."""
    return "no teammate alive" if np.isinf(v) else round(float(v), 1)


def load_config(path: Path = CONFIG) -> dict:
    return json.loads(path.read_text())


def intervals(mask: np.ndarray, t: np.ndarray, min_s: float, merge_gap_s: float) -> list[tuple[int, int]]:
    """(first, last) index of each run of True lasting at least min_s, after
    joining runs separated by gaps shorter than merge_gap_s."""
    step = float(t[1] - t[0]) if len(t) > 1 else 0.1
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    runs, a, b = [], idx[0], idx[0]
    for i in idx[1:]:
        if (i - b) * step <= merge_gap_s + 1e-9:
            b = i
        else:
            runs.append((a, b))
            a = b = i
    runs.append((a, b))
    return [(a, b) for a, b in runs if (b - a + 1) * step >= min_s - 1e-9]


def detect(m: Match, who: Player, mo: Moments, cfg: dict | None = None) -> list[Event]:
    cfg = cfg or load_config()
    gap = cfg.get("merge_gap_s", 0.5)
    t = m.t
    events: list[Event] = []
    names = {p.id: p for p in m.players}

    def conf(a, b):
        return round(float(mo.observed[a:b + 1].mean()), 2)

    in_range = np.array([len(x) for x in mo.in_range_of])

    unsupported = np.array([len(x) for x in mo.unsupported])

    # overextended: ahead of the team while taking fire nobody can return
    c = cfg["overextended"]
    mask = mo.known & (unsupported >= 1) & (mo.ahead > c["ahead_tiles"])
    for a, b in intervals(mask, t, c["min_s"], gap):
        foes = sorted(set().union(*[set(x) for x in mo.unsupported[a:b + 1]]))
        mates = sorted({mo.nearest_ally_id[j] for j in range(a, b + 1) if mo.nearest_ally_id[j]})
        events.append(Event("overextended", t[a], t[b], [who.id] + foes, dict(
            ahead_max=round(float(np.nanmax(mo.ahead[a:b + 1])), 1),
            ahead_mean=round(float(np.nanmean(mo.ahead[a:b + 1])), 1),
            nearest_ally=mates, enemies_in_range=foes), conf(a, b)))

    # risky push: ahead of the team towards enemies you'd lost sight of
    c = cfg.get("risky_push", {"ahead_tiles": 3.0, "min_s": 0.5})
    hidden = np.array([len(x) for x in mo.hidden_threats])
    mask = mo.known & (hidden >= 1) & (mo.ahead > c["ahead_tiles"]) & (unsupported == 0)
    for a, b in intervals(mask, t, c["min_s"], gap):
        seen = [h for j in range(a, b + 1) for h in mo.hidden_threats[j]]
        first = {}
        for h in seen:
            first.setdefault(h[0], h)          # earliest record per enemy: when you pushed into it
        events.append(Event("risky_push", t[a], t[b], [who.id] + sorted(first), dict(
            ahead_mean=round(float(np.nanmean(mo.ahead[a:b + 1])), 1),
            hidden=[dict(enemy=e, seconds_since_seen=h[1], last_seen=[h[2], h[3]], in_bush=h[4] == "B")
                    for e, h in sorted(first.items())]), conf(a, b)))

    # outnumbered: 2+ enemies shooting at you, none of them under fire from your team
    c = cfg["outnumbered"]
    mask = mo.known & (in_range >= c["min_enemies"]) & (unsupported == in_range)
    for a, b in intervals(mask, t, c["min_s"], gap):
        foes = sorted(set().union(*[set(x) for x in mo.in_range_of[a:b + 1]]))
        events.append(Event("outnumbered", t[a], t[b], [who.id] + foes, dict(
            enemies_in_range_max=int(in_range[a:b + 1].max())), conf(a, b)))

    # crossfire (positive): you and a teammate both shooting the same enemy from different sides
    c = cfg.get("crossfire", {"min_s": 2.0})
    xf = np.array([bool(x) for x in mo.crossfire]) & mo.known
    for a, b in intervals(xf, t, c["min_s"], gap):
        pairs = [(e, mate) for j in range(a, b + 1) for e, mate, _ in mo.crossfire[j]]
        e, mate = max(set(pairs), key=pairs.count)
        angles = [ang for j in range(a, b + 1) for ee, mm, ang in mo.crossfire[j] if (ee, mm) == (e, mate)]
        events.append(Event("crossfire", t[a], t[b], [who.id, e, mate], dict(
            enemy=e, teammate=mate, angle=int(np.median(angles))), conf(a, b)))

    # good spacing: just outside the nearest seen enemy's range
    c = cfg["good_spacing"]
    band = np.zeros(len(t), bool)
    for j in np.flatnonzero(mo.known):
        e = names.get(mo.nearest_enemy_id[j]) if mo.nearest_enemy_id[j] else None
        if e is not None and e.range is not None:
            band[j] = e.range < mo.nearest_enemy[j] <= e.range + c["band_tiles"]
    for a, b in intervals(band, t, c["min_s"], gap):
        foe_ids = [mo.nearest_enemy_id[j] for j in range(a, b + 1) if band[j]]
        foe = max(set(foe_ids), key=foe_ids.count)
        d = mo.nearest_enemy[a:b + 1][band[a:b + 1]]
        events.append(Event("good_spacing", t[a], t[b], [who.id, foe], dict(
            enemy=foe, enemy_range=names[foe].range, distance_min=round(float(d.min()), 1),
            distance_max=round(float(d.max()), 1)), conf(a, b)))

    # too passive: certainly can't reach any enemy while a teammate is fighting
    c = cfg["too_passive"]
    for a, b in intervals(mo.can_hit_none & mo.ally_fighting, t, c["min_s"], gap):
        events.append(Event("too_passive", t[a], t[b], [who.id], dict(
            nearest_enemy_min=None if np.isnan(mo.nearest_enemy[a:b + 1]).all()
            else round(float(np.nanmin(mo.nearest_enemy[a:b + 1])), 1),
            own_range=who.range), conf(a, b)))

    # eliminations (subject's), with isolation beforehand
    c = cfg["isolated_before_death"]
    for d in m.deaths:
        if d.player != who.id:
            continue
        j = int(np.argmin(np.abs(t - d.match_time)))
        events.append(Event("eliminated", d.match_time, d.match_time, [who.id],
                            dict(clock=d.clock), 1.0 if d.clock in ("bar", "inferred") else 0.5))
        win = (t >= d.match_time - c["window_s"]) & (t < d.match_time) & mo.known
        alone = win & (unsupported >= 1)
        measured = win
        step = m.step()
        if alone.sum() * step >= c["min_isolated_s"]:
            events.append(Event("isolated_before_death", float(t[win][0]) if win.any() else d.match_time,
                                d.match_time, [who.id], dict(
                                    isolated_s=round(float(alone.sum() * step), 1),
                                    measured_s=round(float(measured.sum() * step), 1),
                                    attackers=sorted(set().union(*[set(mo.unsupported[j]) for j in np.flatnonzero(alone)]))),
                                round(float(mo.observed[win].mean()), 2) if win.any() else 0.0))

    for k, (r0, _) in enumerate(m.rounds):
        events.append(Event("round_start", r0, r0, [], dict(round=k + 1)))
    events.sort(key=lambda e: (e.start, e.type))
    return events


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Rule-based positioning events for one match.")
    ap.add_argument("folder", type=Path)
    ap.add_argument("--map", required=True)
    args = ap.parse_args(argv)
    m = load_match(args.folder, args.map)
    me = m.me
    if me is None:
        raise SystemExit("the recording player wasn't identified")
    mo = moments(m, me)
    evs = detect(m, me, mo)
    (args.folder / "metrics.json").write_text(json.dumps(summarize(m, me, mo), indent=1))
    (args.folder / "events.json").write_text(json.dumps([asdict(e) for e in evs], indent=1, default=float))
    labels = {p.id: (p.label if not p.is_me else "you") + (f" ({p.brawler})" if p.brawler else "") for p in m.players}
    for e in evs:
        who = ", ".join(labels.get(p, p) for p in e.players[1:])
        print(f"{e.start:6.1f}-{e.end:6.1f}s  {e.type:22s} conf {e.confidence:.2f}  {who}  {e.measurements}")


if __name__ == "__main__":
    main()
