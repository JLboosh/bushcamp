"""Match report: a written summary + transcript, and a visual HTML page.

Everything in the report comes from computed data (pipeline/analyze.py,
pipeline/events.py); sentences are filled-in templates. No language model.

Written report
    summary     the 2-3 biggest takeaways, ranked by fixed rules
    transcript  one timestamped line per event; lines resting on partly
                interpolated data say so

Visual report (report.html, self-contained, works offline)
    map replay with every brawler, time slider and play controls
    heatmap of your positions, death markers with your last 5 seconds
    spacing chart (nearest teammate / nearest enemy, with range lines)
    your video, synced: click the chart, a transcript line or the map's
    slider and the video jumps there; play the video and the map follows

    python -m pipeline.report runs/match/<video> --map streets_with_no_name
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pipeline.analyze import Match, Moments, load_match, moments, summarize
from pipeline.events import Event, detect, load_config
from pipeline.gaps import INTERPOLATED, OBSERVED
from pipeline.suggest import Suggestion, suggestions as find_suggestions

LOW_CONFIDENCE = 0.6           # below this share of the span seen, a line is marked as partly estimated


# --- text ---------------------------------------------------------------------------

def clock_text(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 60)}:{int(t % 60):02d}"


def who(m: Match, pid: str) -> str:
    """How a player is named in the report: by brawler only, never by player name."""
    p = next((p for p in m.players if p.id == pid), None)
    if p is None:
        return "an unidentified player"
    if p.is_me:
        return "you"
    return p.brawler or ("an enemy" if p.team == "enemy" else "a teammate")


def tiles(v: float) -> str:
    """Distance for a sentence: one decimal below 10 tiles (so 4.2 doesn't read as 4)."""
    return f"{v:.1f}" if v < 10 else f"{v:.0f}"


def listing(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def sentence(m: Match, e: Event, cfg: dict, death_info: dict | None = None) -> str | None:
    """One transcript line for an event, from a template."""
    ms = e.measurements
    dur = e.end - e.start + m.step()
    others = [who(m, p) for p in e.players[1:]]
    if e.type == "overextended":
        mates = [who(m, p) for p in ms.get("nearest_ally", [])]
        mate = f" ({listing(mates)})" if mates else ""
        return (f"You were about {ms['ahead_mean']:.0f} tiles ahead of your nearest teammate{mate} with "
                f"{listing(others)} having a clear shot at you that none of your teammates could return, "
                f"for {dur:.0f} s.")
    if e.type == "outnumbered":
        return (f"{ms['enemies_in_range_max']} enemies had a clear shot at you ({listing(others)}) "
                f"and none of your teammates could shoot back at them.")
    if e.type == "isolated_before_death":
        c = cfg["isolated_before_death"]
        att = [who(m, p) for p in ms.get("attackers", [])]
        by = f" from {listing(att)}" if att else ""
        return (f"In the {c['window_s']:.0f} seconds before your elimination you were under fire{by} that "
                f"none of your teammates could return for {ms['isolated_s']:.0f} of them.")
    if e.type == "risky_push":
        parts = []
        for h in ms["hidden"]:
            age = h["seconds_since_seen"]
            when = "just before" if age < 1 else f"{age:.0f} s earlier"
            how = f"last seen going into a bush {when}" if h["in_bush"] else f"out of view since {when}" if age >= 1 else "just gone out of view"
            parts.append(f"{who(m, h['enemy'])} ({how})")
        return (f"You pushed about {ms['ahead_mean']:.0f} tiles ahead of your team where {listing(parts)} "
                f"could have reached you without you seeing it coming.")
    if e.type == "crossfire":
        return (f"Good angle: you and {who(m, ms['teammate'])} both had a clear shot at {who(m, ms['enemy'])} "
                f"from about {ms['angle']}° apart for {dur:.0f} s, so it had nowhere to hide from both of you.")
    if e.type == "eliminated":
        where = ""
        if death_info and death_info.get("tile") == "B":
            where = " in a bush"
        enemies = [who(m, p) for p in (death_info or {}).get("enemies_involved", [])]
        by = f" {listing(enemies)} had a clear shot at you just before." if enemies else ""
        approx = " (time approximate)" if ms.get("clock") not in ("bar", "inferred") else ""
        return f"Eliminated{where}.{by}{approx}"
    if e.type == "good_spacing":
        enemy = who(m, ms["enemy"])
        return (f"Good spacing: you held {ms['distance_min']:.0f}-{ms['distance_max']:.0f} tiles from {enemy}, "
                f"just outside their {ms['enemy_range']:g}-tile range, for {dur:.0f} seconds.")
    if e.type == "too_passive":
        return (f"For {dur:.0f} s you were out of reach of every enemy while a teammate was fighting"
                f" (your range: {ms['own_range']:g} tiles).")
    if e.type == "round_start":
        return f"Round {ms['round']} starts."
    return None


def suggestion_text(m: Match, s: Suggestion) -> str:
    """The suggestion line, built only from the two evaluated spots."""
    cur, new = s.current, s.better
    d = np.array([new.x - cur.x, new.y - cur.y])
    along = float(d @ m.forward)
    dist = float(np.hypot(*d))
    way = "back" if along < -0.5 else "forward" if along > 0.5 else "to the side"
    where = " into the bush" if new.bush and not cur.bush else " behind cover" if new.cover and not cur.cover else ""
    step = "less than a tile" if dist < 1 else f"{dist:.0f} tile{'s' * (round(dist) != 1)}"
    parts = [f"Better: {step} {way}{where}."]
    gone = [p for p in cur.threats if p not in new.threats]
    if gone:
        j = int(np.argmin(np.abs(m.t - s.t)))
        walled, ranged = [], []
        for pid in gone:
            e = next(p for p in m.players if p.id == pid)
            far = np.hypot(e.x[j] - new.x, e.y[j] - new.y) > e.range
            (ranged if far else walled).append(who(m, pid))
        if walled:
            parts.append(f"A wall would have blocked {listing(walled)}'s shots.")
        if ranged:
            parts.append(f"You'd have been out of {listing(ranged)}'s range.")
    newly_traded = [p for p in cur.untraded if p in new.threats and p not in new.untraded]
    if newly_traded:
        parts.append(f"Your team could shoot back at {listing([who(m, p) for p in newly_traded])}.")
    hidden_gone = [p for p in cur.hidden if p not in new.hidden]
    if hidden_gone:
        parts.append(f"Out of reach of where you last saw {listing([who(m, p) for p in hidden_gone])}.")
    if new.crossfire:
        e, mate = new.crossfire[0]
        parts.append(f"You'd have had {who(m, e)} in a crossfire with {who(m, mate)}.")
    elif new.in_reach:
        kept = [p for p in new.in_reach if p in cur.in_reach]
        verb = "still have" if kept else "have"
        parts.append(f"You'd {verb} a clear shot at {listing([who(m, p) for p in new.in_reach])}.")
    if s.enemies_off_screen:
        n = s.enemies_off_screen
        parts.append(f"({n} enem{'ies were' if n > 1 else 'y was'} off screen and not counted.)")
    return " ".join(parts)


def transcript(m: Match, events: list[Event], metrics: dict, cfg: dict, video_time,
               sugg: list[Suggestion] | None = None) -> list[dict]:
    deaths = {round(d["match_time"], 1): d for d in metrics["deaths"]}
    lines = []
    for e in events:
        text = sentence(m, e, cfg, deaths.get(round(e.start, 1)))
        if text is None:
            continue
        note = None
        if e.type not in ("round_start", "eliminated") and e.confidence < LOW_CONFIDENCE:
            note = f"partly estimated: you were seen for {e.confidence * 100:.0f}% of this"
        lines.append(dict(t=round(e.start, 2), end=round(e.end, 2), clock=clock_text(e.start), type=e.type,
                          text=text, note=note, video_time=video_time(e.start)))
        s = next((s for s in sugg or [] if s.event == e.type and abs(s.t - e.start) < 0.06), None)
        if s:
            lines.append(dict(t=round(e.start, 2), end=round(e.end, 2), clock=clock_text(e.start), type="suggestion",
                              text=suggestion_text(m, s), note=None, video_time=video_time(e.start)))
    return lines


def takeaways(m: Match, events: list[Event], metrics: dict, cfg: dict, top: int = 3) -> list[str]:
    """The biggest takeaways, ranked by fixed weights on computed totals."""
    step = m.step()
    cands: list[tuple[float, str]] = []

    def total(kind):
        evs = [e for e in events if e.type == kind]
        return evs, sum(e.end - e.start + step for e in evs)

    over, over_s = total("overextended")
    if over:
        times = "once" if len(over) == 1 else f"{len(over)} times"
        cands.append((over_s, f"You overextended {times} ({over_s:.0f} s in all): ahead of your team where an "
                              f"enemy could shoot you and none of your teammates could shoot back."))
    outn, outn_s = total("outnumbered")
    if outn:
        times = "once" if len(outn) == 1 else f"{len(outn)} times"
        cands.append((outn_s * 1.5 + 2 * len(outn), f"You were outnumbered {times}: 2 or more enemies had a "
                                                    f"clear shot at you and your team couldn't return fire."))
    risky, risky_s = total("risky_push")
    if risky:
        times = "once" if len(risky) == 1 else f"{len(risky)} times"
        cands.append((risky_s * 1.2 + 1.5 * len(risky), f"You pushed ahead into enemies you'd lost sight of {times}: "
                                                        f"they had gone into a bush or out of view and could have reached you."))
    xf, xf_s = total("crossfire")
    if xf:
        cands.append((xf_s * 0.8, f"Good angles: {xf_s:.0f} s in crossfire with a teammate, both shooting the same "
                                  f"enemy from different sides."))
    deaths = [e for e in events if e.type == "eliminated"]
    isolated = [e for e in events if e.type == "isolated_before_death"]
    if deaths:
        if isolated:
            share = ("Your only elimination" if len(deaths) == 1 else
                     f"All {len(deaths)} of your eliminations" if len(isolated) == len(deaths) else
                     f"{len(isolated)} of your {len(deaths)} eliminations")
            cands.append((10.0 * len(isolated), f"{share} came after taking fire your team couldn't return."))
        else:
            times = "once" if len(deaths) == 1 else f"{len(deaths)} times"
            cands.append((3.0 * len(deaths), f"You were eliminated {times}; "
                                             "your team could return fire beforehand."))
    good, good_s = total("good_spacing")
    if good:
        cands.append((good_s, f"Good spacing: {good_s:.0f} s spent just outside an enemy's range"
                              f" ({listing(sorted({who(m, e.measurements['enemy']) for e in good}))})."))
    passive, passive_s = total("too_passive")
    if passive:
        cands.append((passive_s, f"You spent {passive_s:.0f} s out of reach of every enemy while a teammate was fighting."))
    alive = metrics["alive_s"] or 1
    in_range = metrics["time_in_enemy_range_s"]
    cands.append((in_range / alive * 10, f"An enemy had a clear shot at you for {in_range:.0f} s "
                                         f"({in_range / alive * 100:.0f}% of your time alive)."))
    cands.sort(key=lambda c: -c[0])
    return [c[1] for c in cands[:top]]


# --- data for the page ------------------------------------------------------------

def page_data(folder: Path, map_id: str, m: Match, mo: Moments, metrics: dict, events: list[Event],
              lines: list[dict], summary: list[str], video_rel: str) -> dict:
    ref = m.ref
    clock = list(csv.DictReader((folder / "clock.csv").open()))
    me = m.me
    players = []
    for p in m.players:
        xs = [None if f not in (OBSERVED, INTERPOLATED) else round(float(x), 2) for x, f in zip(p.x, p.flag)]
        ys = [None if f not in (OBSERVED, INTERPOLATED) else round(float(y), 2) for y, f in zip(p.y, p.flag)]
        players.append(dict(id=p.id, name=who(m, p.id), team=p.team, me=p.is_me, brawler=p.brawler,
                            range=p.range, x=xs, y=ys, seen=[f == OBSERVED for f in p.flag],
                            alive=[bool(a) for a in p.alive]))
    enemy_range = []
    names = {p.id: p for p in m.players}
    for j in range(len(m.t)):
        e = names.get(mo.nearest_enemy_id[j]) if mo.nearest_enemy_id[j] else None
        enemy_range.append(e.range if e is not None else None)
    deaths = []
    for d in metrics["deaths"]:
        sel = (m.t >= d["match_time"] - 5) & (m.t <= d["match_time"]) & me.known
        deaths.append(dict(t=d["match_time"], pos=d["position"],
                           trail=[[round(float(me.x[j]), 2), round(float(me.y[j]), 2)] for j in np.flatnonzero(sel)]))
    nan = lambda a: [None if np.isnan(v) else round(float(v), 2) for v in a]
    from pipeline.maps import MAPS_DIR
    img = MAPS_DIR / map_id / json.loads((MAPS_DIR / map_id / "map.json").read_text())["image"]
    mime = {".webp": "image/webp", ".png": "image/png", ".jpg": "image/jpeg"}[img.suffix.lower()]
    return dict(
        title=f"{ref.name}: {me.brawler or 'you'}",
        map=dict(name=ref.name, cols=ref.cols, rows=ref.rows, border=ref.border, tile=ref.tile_px,
                 width=int(ref.image.shape[1]), height=int(ref.image.shape[0]),
                 image=f"data:{mime};base64," + base64.b64encode(img.read_bytes()).decode()),
        t=[round(float(v), 2) for v in m.t], step=m.step(), players=players,
        spacing=dict(ally=nan(mo.nearest_ally), enemy=nan(mo.nearest_enemy), enemy_range=enemy_range,
                     own_range=me.range),
        heatmap=metrics["heatmap"], deaths=deaths, rounds=metrics["rounds"],
        events=[dict(type=e.type, start=e.start, end=e.end) for e in events if e.type != "round_start"],
        transcript=lines, summary=summary,
        stats=dict(alive=metrics["alive_s"], known=metrics["position_known_s"],
                   spacing=metrics["spacing_to_nearest_ally"], enemy=metrics["distance_to_nearest_enemy"],
                   in_range=metrics["time_in_enemy_range_s"], safe=metrics["time_safe_from_all_enemies_s"],
                   bush=metrics["time_in_bush_s"], cover=metrics["time_near_cover_s"],
                   brawler=me.brawler, own_range=me.range),
        clock=dict(video=[round(float(c["video_time"]), 3) for c in clock],
                   match=[None if c["match_time"] == "" else round(float(c["match_time"]), 3) for c in clock]),
        video=video_rel,
    )


def video_lookup(folder: Path):
    """match time -> video time: the first frame showing that moment (a rewind shows it twice)."""
    clock = [c for c in csv.DictReader((folder / "clock.csv").open()) if c["match_time"] != ""]
    vt = np.array([float(c["video_time"]) for c in clock])
    mt = np.array([float(c["match_time"]) for c in clock])

    def lookup(t: float) -> float | None:
        if not len(mt):
            return None
        d = np.abs(mt - t)
        close = np.flatnonzero(d <= d.min() + 0.05)
        return round(float(vt[close[0]]), 2)
    return lookup


def build(folder: Path, map_id: str, video: Path | None = None) -> dict:
    cfg = load_config()
    m = load_match(folder, map_id)
    me = m.me
    if me is None:
        raise SystemExit("the recording player wasn't identified (data/player.json)")
    mo = moments(m, me)
    metrics = summarize(m, me, mo)
    events = detect(m, me, mo, cfg)
    lookup = video_lookup(folder)
    sugg = find_suggestions(m, me, mo, events, cfg.get("suggest"))
    lines = transcript(m, events, metrics, cfg, lookup, sugg)
    summary = takeaways(m, events, metrics, cfg)
    if video is None:
        video = Path(json.loads((folder / "identities.json").read_text())["video"])
    video_rel = os.path.relpath(Path(video).resolve(), folder.resolve())
    data = page_data(folder, map_id, m, mo, metrics, events, lines, summary, video_rel)
    data["suggestions"] = [dict(t=s.t, route=[[round(x, 2), round(y, 2)] for x, y in s.route],
                                to=[s.better.x, s.better.y]) for s in sugg]
    (folder / "report.json").write_text(json.dumps(dict(summary=summary, transcript=lines), indent=1, ensure_ascii=False))
    (folder / "report.md").write_text(markdown(data))
    (folder / "report.html").write_text(HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
    return data


def markdown(d: dict) -> str:
    out = [f"# {d['title']}", "", "## Summary", ""] + [f"- {s}" for s in d["summary"]] + ["", "## Transcript", ""]
    for line in d["transcript"]:
        note = f" _({line['note']})_" if line["note"] else ""
        out.append(f"- **{line['clock']}** {line['text']}{note}")
    return "\n".join(out) + "\n"


HTML = (Path(__file__).resolve().parent / "report_template.html").read_text() if \
    (Path(__file__).resolve().parent / "report_template.html").exists() else ""


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Written + visual report for one match.")
    ap.add_argument("folder", type=Path)
    ap.add_argument("--map", required=True)
    ap.add_argument("--video", type=Path, help="the recording (default: the one the match was built from)")
    args = ap.parse_args(argv)
    d = build(args.folder, args.map, args.video)
    print("\n".join(["Summary:"] + [f"  - {s}" for s in d["summary"]] + ["Transcript:"] +
                    [f"  {l['clock']}  {l['text']}" + (f"  ({l['note']})" if l["note"] else "") for l in d["transcript"]]))
    print(f"-> {args.folder / 'report.html'}")


if __name__ == "__main__":
    main()
