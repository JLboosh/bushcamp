"""Analyse a whole recording: find the matches in it, work out the map, run
every stage, and write a report per match. This is what the server's worker
calls; it's also usable from the command line.

    python -m pipeline.run data/raw/recording.mp4 --out runs/upload --player LOUIHBOOSHES

Steps
1. Scan the video at 1 fps for gameplay: the top bar shows both teams'
   portraits (blue left, red right). Loading, VS and result screens don't.
   Gameplay stretches (gaps under 15 s joined, at least 20 s long) are matches.
2. Per match: footage type (replay: the replay progress bar appears; live
   recordings aren't supported yet, milestone 9b), and the map, by fitting a
   few frames' bushes and water against every map in maps/.
3. Per match: clock, camera, detection, identities, brawlers, analysis,
   events, suggestions and the report (pipeline/track.py, pipeline/report.py),
   each pass limited to the match's time window.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from pipeline import localize, report, timer, track
from pipeline.brawlers import hud_slots
from pipeline.maps import MAPS_DIR, load as load_map
from pipeline.sampling import iter_frames, progress_hook, video_info, window

HUD_MIN = 0.17            # share of team colour in each side of the top bar during gameplay
MERGE_GAP_S = 15.0
MIN_MATCH_S = 20.0
PAD_S = 3.0               # keep a little before/after (the VS screen helps the clock and roster)

# Share of the work each stage takes (measured roughly on the MVP recordings).
STAGES = [("scan", 4), ("clock", 6), ("calibrate", 6), ("camera", 14), ("detect", 22), ("names", 10),
          ("brawlers", 6), ("kill_feed", 22), ("report", 10)]


class UnsupportedFootage(Exception):
    pass


@dataclass
class Segment:
    start: float
    end: float


def hud_present(image: np.ndarray) -> bool:
    H, W = image.shape[:2]
    for team, boxes in hud_slots(W, H).items():
        x0, x1 = boxes[0][0] - 6, boxes[-1][2] + 6
        y0, y1 = max(0, boxes[0][1] - 6), boxes[0][3] + 6
        hsv = cv2.cvtColor(image[y0:y1, max(0, x0):x1], cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        colour = (h >= 100) & (h <= 125) if team == "ally" else (h >= 170) | (h <= 5)
        if float((colour & (s > 120) & (v > 90)).mean()) < HUD_MIN:
            return False
    return True


def find_matches(video: Path) -> tuple[list[Segment], list[tuple[float, bool]]]:
    """Gameplay stretches in the recording, and the per-second scan behind them."""
    scan = [(f.video_time, hud_present(f.image)) for f in iter_frames(video, fps=1.0)]
    runs: list[list[float]] = []
    for t, on in scan:
        if not on:
            continue
        if runs and t - runs[-1][1] <= MERGE_GAP_S:
            runs[-1][1] = t
        else:
            runs.append([t, t])
    duration = scan[-1][0] if scan else 0.0
    segs = [Segment(max(0.0, a - PAD_S), min(duration + 1, b + PAD_S)) for a, b in runs if b - a >= MIN_MATCH_S]
    return segs, scan


def detect_map(video: Path, seg: Segment, n: int = 10) -> tuple[str, dict[str, float]]:
    """Which known map this is: mean layout-fit score of a few frames, per map (and orientation)."""
    with window(seg.start, seg.end):
        frames = [f.image for f in iter_frames(video, fps=max(0.2, n / max(1.0, seg.end - seg.start)))]
    frames = frames[len(frames) // 10: len(frames) - len(frames) // 10 or None][:n] or frames[:n]
    h, w = frames[0].shape[:2]
    cam = typical_camera(w, h)
    masks = [localize.feature_masks(f) for f in frames]
    scores = {}
    for d in sorted(p for p in MAPS_DIR.iterdir() if (p / "map.json").exists()):
        ref = load_map(d.name)
        best = 0.0
        for rotated in (False, True):
            layout = localize.layout_images(ref, rotated)
            vals = []
            for m in masks:
                fix = localize.match(m, cam, layout)
                vals.append(fix.score if fix else 0.0)
            best = max(best, float(np.mean(vals)))
        scores[d.name] = round(best, 3)
    return max(scores, key=scores.get), scores


def typical_camera(width: int, height: int) -> localize.Camera:
    """The replay camera's perspective as calibrated on the MVP recordings (1562x720),
    scaled to this resolution; good enough to tell maps apart before calibrating."""
    s = 720 / height
    return localize.Camera(width, height, kx=0.0198 * s, kv=0.0247 * s, p=0.00026 * s)


def footage_type(video: Path, seg: Segment) -> str:
    with window(seg.start, seg.end):
        layout = timer.locate_bar(f.image for f in iter_frames(video, fps=1.0))
    return "replay" if layout is not None else "live"


def analyze_video(video: Path, out_root: Path, player: str | None = None, map_id: str | None = None,
                  on_progress=None) -> list[dict]:
    """Analyse every match in the recording. on_progress(fraction, message) reports progress.
    Returns one summary per match (folder, map, brawler, summary lines) or raises UnsupportedFootage."""
    say = on_progress or (lambda f, msg: None)
    total = sum(w for _, w in STAGES)
    done_before = {name: sum(w for _, w in STAGES[:i]) for i, (name, _) in enumerate(STAGES)}
    weights = dict(STAGES)

    def reporter(match_i, n_matches):
        state = {"stage": "scan"}

        def overall(frac):
            base = done_before[state["stage"]] + weights[state["stage"]] * frac
            say((match_i + base / total) / max(1, n_matches), LABELS[state["stage"]])

        def stage(name):
            state["stage"] = name
            overall(0.0)
        return stage, overall

    stage, overall = reporter(0, 1)
    stage("scan")
    with progress_hook(overall):
        segs, _ = find_matches(video)
    if not segs:
        raise UnsupportedFootage("No gameplay found in this recording (the top bar with both teams never appeared).")

    results = []
    for i, seg in enumerate(segs):
        stage, overall = reporter(i, len(segs))
        stage("clock")
        kind = footage_type(video, seg)
        if kind != "replay":
            raise UnsupportedFootage("This looks like live gameplay, not a replay. Live recordings are coming "
                                     "in the next step (milestone 9b); for now, record the replay viewer.")
        chosen, scores = (map_id, {}) if map_id else detect_map(video, seg)
        out = out_root / f"match{i + 1}"
        with window(seg.start, seg.end), progress_hook(overall):
            summ = track.run(video, chosen, out=out, player=player, replay=False, on_stage=stage)
            stage("report")
            data = report.build(out, chosen, video)
        meta = dict(folder=str(out), map=chosen, map_name=load_map(chosen).name, map_scores=scores,
                    window=[round(seg.start, 1), round(seg.end, 1)], footage=kind,
                    brawler=data["stats"]["brawler"], me_found=summ["me_found"],
                    summary=data["summary"], title=data["title"])
        (out / "match.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        results.append(meta)
        overall(1.0)
    say(1.0, "Done")
    return results


LABELS = {"scan": "Finding matches in the recording", "clock": "Reading the replay clock",
          "calibrate": "Calibrating the camera", "camera": "Following the camera on the map",
          "detect": "Detecting brawlers", "names": "Reading name tags", "brawlers": "Identifying brawlers",
          "kill_feed": "Matching players to brawlers", "report": "Writing the report"}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Analyse a whole recording (all matches in it).")
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=Path("runs/upload"))
    ap.add_argument("--player", help="your in-game name (default: data/player.json)")
    ap.add_argument("--map", help="skip map detection")
    args = ap.parse_args(argv)
    last = {"pct": -1}

    def show(frac, msg):
        pct = int(frac * 100)
        if pct != last["pct"]:
            print(f"\r{pct:3d}%  {msg:45s}", end="", flush=True)
            last["pct"] = pct
    res = analyze_video(args.video, args.out / args.video.stem, args.player, args.map, show)
    print()
    for r in res:
        print(f"{r['folder']}: {r['map_name']} ({r['footage']}), {r['brawler']}  -> report.html")


if __name__ == "__main__":
    main()
