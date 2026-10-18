"""Player tracks in map tiles, indexed by match time.

Joins the three per-frame streams (same video, same sampling rate):
    clock     pipeline/timer.py     match time of each frame
    camera    pipeline/localize.py  where the camera is on the map
    boxes     pipeline/detect.py    brawlers on screen, with tracker ids

then:
1. Names: OCR a few name-tag crops per tracker id and vote.
2. Identities: tracker ids break whenever someone leaves the screen; link
   the pieces into players per team by name (or, when no name could be
   read, by name-tag fingerprint), never joining pieces seen at the same time.
   "You" is the ally whose name matches data/player.json.
3. Positions: the feet (bottom-centre of the box) through the camera's
   screen-to-map transform.
4. Tracks: resampled on match time with observed / interpolated / unknown
   flags (pipeline/gaps.py).
5. Debug replay: top-down map next to the video, to check by eye.

    python -m pipeline.track data/raw/match.mp4 --map streets_with_no_name
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pipeline import detect, gaps, localize, timer
from pipeline.maps import MapRef, load as load_map
from pipeline.sampling import iter_frames

PLAYER_FILE = Path("data/player.json")
NAME_MATCH = 0.75          # string similarity at which two reads are the same name
FINGERPRINT_MATCH = 0.6    # name-tag fingerprint correlation to link an unnamed piece
MIN_PIECE = 10             # detections a nameless piece needs to start a new player
TEAM_SIZE = 3


# --- names ---------------------------------------------------------------------

_ocr = None


def read_name(crop: np.ndarray, careful: bool = False) -> tuple[str, float]:
    """OCR one name-tag crop.

    Fast mode only recognises (the crop is already one line of text, ~10 ms);
    careful mode also finds the words first, which handles separators and
    spaces better (~200 ms), so it's kept for the final label.
    """
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if careful:
        res, _ = _ocr(big, use_cls=False)
        if not res:
            return "", 0.0
        words = sorted(res, key=lambda r: min(pt[0] for pt in r[0]))   # left to right
        words = [w for w in words if not norm(w[1]).isdigit()]          # HP digits creeping in
        return clean_read(" ".join(w[1] for w in words)), float(np.mean([w[2] for w in words])) if words else 0.0
    res, _ = _ocr(big, use_det=False, use_cls=False)
    if not res:
        return "", 0.0
    text, conf = res[0]
    return text, float(conf)


def read_name_boxes(img: np.ndarray) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """OCR with word positions: [(text, confidence, (x0, y0, x1, y1))] in img coordinates."""
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    res, _ = _ocr(img, use_cls=False)
    out = []
    for quad, text, conf in res or []:
        xs, ys = [q[0] for q in quad], [q[1] for q in quad]
        out.append((clean_read(text), float(conf), (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))))
    return out


def final_label(ident: "Identity", crops: dict[int, np.ndarray], n: int = 3) -> str | None:
    """Re-read the identity's clearest name crops carefully and keep the read that
    best agrees with the vote (falls back to the fast-mode name)."""
    idx = [i for p in sorted(ident.pieces, key=lambda p: -len(p.dets)) for i in p.dets if i in crops][:n]
    reads = [read_name(crops[i], careful=True) for i in idx]
    # Only trust a careful read that agrees closely with the vote: a looser one
    # may include a neighbour's tag caught in the same crop.
    reads = [(t, c) for t, c in reads if c >= 0.8 and similar(t, ident.name) >= 0.85]
    return max(reads, key=lambda r: (similar(r[0], ident.name), r[1]))[0] if reads else ident.name


def norm(name: str) -> str:
    """Lower-case letters and digits only (any script: names can be e.g. Chinese)."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def clean_read(text: str) -> str:
    """Drop HP digits that crept into a name crop (a run of 3+ digits at either end)."""
    return re.sub(r"^\W*\d{3,}\W*|\W*\d{3,}\W*$", "", text).strip()


def plausible_name(text: str) -> bool:
    n = norm(text)
    if not n or n.isdigit():
        return False
    return len(n) >= 3 or any(ord(ch) > 127 for ch in n)   # short names in other scripts are fine


def similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio() if a and b else 0.0


def same_name(a: str | None, b: str | None) -> bool:
    """Two reads of one name tag: similar overall, or sharing a long run of
    letters (OCR often garbles separators and prefixes: "SMP | BOBA" reads as
    "PBOBA" or "SMPIBOBA"; "Hg | Tadeo" as "Tadeo")."""
    if not a or not b:
        return False
    if similar(a, b) >= NAME_MATCH:
        return True
    x, y = norm(a), norm(b)
    m = difflib.SequenceMatcher(None, x, y).find_longest_match(0, len(x), 0, len(y))
    return m.size >= max(4, 0.7 * min(len(x), len(y)))


def vote(reads: list[tuple[str, float]]) -> str | None:
    """Most supported name among several OCR reads of the same player."""
    reads = [(clean_read(t), c) for t, c in reads]
    reads = [(t, c) for t, c in reads if plausible_name(t) and c >= 0.6]
    if not reads:
        return None
    support = [sum(c2 for t2, c2 in reads if similar(t, t2) >= NAME_MATCH) for t, _ in reads]
    best = int(np.argmax(support))
    if support[best] < 1.2:            # one shaky read isn't enough
        return None
    group = [(t, c) for t, c in reads if similar(t, reads[best][0]) >= NAME_MATCH]
    return max(group, key=lambda r: r[1])[0]


# --- identities -----------------------------------------------------------------

@dataclass
class Piece:
    """Detections sharing one tracker id."""
    track_id: int
    team: str
    dets: list[int]
    frames: set[int]
    name: str | None = None
    fingerprint: np.ndarray | None = None


@dataclass
class Identity:
    id: str
    team: str
    name: str | None
    pieces: list[Piece] = field(default_factory=list)
    is_me: bool = False
    brawler: str | None = None
    brawler_source: str | None = None      # kill_feed | elimination
    slot: int | None = None                # top-bar slot (0-2 within the team)

    @property
    def frames(self) -> set[int]:
        return set().union(*(p.frames for p in self.pieces)) if self.pieces else set()

    @property
    def fingerprint(self) -> np.ndarray | None:
        fps = [p.fingerprint for p in self.pieces if p.fingerprint is not None]
        return np.mean(fps, axis=0) if fps else None

    @property
    def label(self) -> str:
        return ("you" if self.is_me else self.name) or self.id


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def build_pieces(dets: list[detect.Detection], fingerprints: dict[int, np.ndarray],
                 crops: dict[int, np.ndarray],
                 positions: dict[int, tuple[float, float, float]] | None = None,
                 max_jump: float = 2.5, max_frame_gap: int = 20) -> list[Piece]:
    """Pieces of one player each: detections sharing a tracker id and team,
    split where the box jumps (the tracker switched to a nearby brawler; it
    ignores team) or where the id goes unseen for over 2 s."""
    positions = positions or {}
    by_key: dict[tuple[int, str], list[int]] = {}
    for i, d in enumerate(dets):
        if d.track_id >= 0:
            by_key.setdefault((d.track_id, d.team), []).append(i)
    groups: list[tuple[int, str, list[int]]] = []
    for (tid, team), idx in by_key.items():
        idx.sort(key=lambda i: dets[i].frame)
        current = [idx[0]]
        for a, b in zip(idx, idx[1:]):
            gap = dets[b].frame - dets[a].frame
            jump = (np.hypot(positions[b][1] - positions[a][1], positions[b][2] - positions[a][2])
                    if a in positions and b in positions else 0.0)
            if gap > max_frame_gap or (gap <= 3 and jump > max_jump):
                groups.append((tid, team, current))
                current = []
            current.append(b)
        groups.append((tid, team, current))
    pieces = []
    for tid, team, idx in groups:
        fps = [fingerprints[i] for i in idx if i in fingerprints]
        reads = [read_name(crops[i]) for i in idx if i in crops]
        pieces.append(Piece(tid, team, idx, {dets[i].frame for i in idx}, vote(reads),
                            np.median(fps, axis=0) if len(fps) >= 2 else None))
    return pieces


def link_identities(pieces: list[Piece], player_name: str | None,
                    positions: dict[int, tuple[float, float, float]] | None = None,
                    overlap_ok: int = 2) -> tuple[list[Identity], list[Piece]]:
    """Group pieces into players: same team, never on screen together, and
    matched by (in order of trust):
      1. the name read on the tag (a name belongs to one team only)
      2. the name-tag fingerprint, when the name couldn't be read
      3. elimination: on a full team, the only player it could be
      4. movement: the player who could have walked to where the piece starts

    positions: detection index -> (match time, x, y), used by rules 3-4.
    Rules 3-4 only apply to pieces showing some name text, so spawned objects
    (health bar, no name) don't get attached to a player.
    Returns (identities, unassigned pieces).
    """
    positions = positions or {}
    # A name belongs to one player: if both teams read it, the side with more
    # support keeps it (the other read was a neighbour's tag in the crop).
    support: dict[tuple[str, str], int] = {}
    for p in pieces:
        if p.name:
            key = next((k for k in support if k[0] == p.team and same_name(k[1], p.name)), (p.team, p.name))
            support[key] = support.get(key, 0) + len(p.dets)
    for p in pieces:
        if p.name:
            mine = sum(n for (team, name), n in support.items() if team == p.team and same_name(name, p.name))
            other = sum(n for (team, name), n in support.items() if team != p.team and same_name(name, p.name))
            if other > mine:
                p.name = None

    identities: list[Identity] = []
    unassigned: list[Piece] = []
    for team in ("ally", "enemy"):
        mine = sorted((p for p in pieces if p.team == team), key=lambda p: -len(p.dets))
        players: list[Identity] = []
        for p in (p for p in mine if p.name):
            same = [ident for ident in players if same_name(ident.name, p.name)]
            if same and len(same[0].frames & p.frames) > overlap_ok:
                p.name = None            # on screen together with that player: a neighbour's tag was read
            elif same:
                same[0].pieces.append(p)
            elif len(players) < TEAM_SIZE:
                players.append(Identity(f"{team}{len(players) + 1}", team, p.name, [p]))
            else:
                p.name = None            # a 4th name on a 3-player team: treat as unread
        for p in (p for p in mine if not p.name):
            free = [ident for ident in players if len(ident.frames & p.frames) <= overlap_ok]
            has_text = p.fingerprint is not None
            scored = [(_corr(p.fingerprint, ident.fingerprint), ident) for ident in free
                      if has_text and ident.fingerprint is not None]
            best_fp = max(scored, key=lambda s: s[0], default=(0.0, None))
            reachable = sorted(((s, ident) for ident in free
                                if (s := _speed_needed(p, ident, positions)) is not None and s <= MAX_LINK_SPEED),
                               key=lambda r: r[0])
            if best_fp[1] is not None and best_fp[0] >= FINGERPRINT_MATCH:
                best_fp[1].pieces.append(p)
            elif has_text and len(players) == TEAM_SIZE and len(free) == 1 and \
                    (_speed_needed(p, free[0], positions) or 0.0) <= MAX_LINK_SPEED:
                free[0].pieces.append(p)
            elif has_text and reachable and (len(players) == TEAM_SIZE or len(p.dets) < MIN_PIECE):
                reachable[0][1].pieces.append(p)
            elif has_text and len(players) < TEAM_SIZE and len(p.dets) >= MIN_PIECE:
                players.append(Identity(f"{team}{len(players) + 1}", team, None, [p]))
            else:
                unassigned.append(p)
        identities += players
    if player_name:
        allies = [(similar(i.name, player_name), i) for i in identities if i.team == "ally" and i.name]
        score, me = max(allies, key=lambda a: a[0], default=(0.0, None))
        if me is not None and score >= NAME_MATCH:
            me.is_me = True
    return identities, unassigned


MAX_LINK_SPEED = 8.0       # tiles/s a player could cover between two pieces (with slack for dashes)


def _speed_needed(p: Piece, ident: Identity, positions: dict[int, tuple[float, float, float]],
                  window: float = 6.0) -> float | None:
    """Slowest speed (tiles/s) that connects the piece to the player's nearest
    sightings before and after it; None if there's nothing close in time to compare."""
    mine = [positions[i] for i in p.dets if i in positions]
    theirs = [positions[i] for q in ident.pieces for i in q.dets if i in positions]
    if not mine or not theirs:
        return None
    t0 = min(mine, key=lambda m: m[0])
    t1 = max(mine, key=lambda m: m[0])
    worst = None
    before = [m for m in theirs if t0[0] - window <= m[0] < t0[0]]
    after = [m for m in theirs if t1[0] < m[0] <= t1[0] + window]
    for a, b in ((max(before, key=lambda m: m[0]), t0) if before else (None, None),
                 (t1, min(after, key=lambda m: m[0])) if after else (None, None)):
        if a is None:
            continue
        dt = max(b[0] - a[0], 0.1)
        v = max(0.0, np.hypot(b[1] - a[1], b[2] - a[2]) - 1.0) / dt   # 1 tile of slack for box jitter
        worst = v if worst is None else max(worst, v)
    return worst


# --- brawlers ---------------------------------------------------------------------

def assign_brawlers(identities: list[Identity], roster: dict[str, list[dict]],
                    hits: list[tuple[float, str, str, int, float]]) -> None:
    """Give each player the brawler of their top-bar slot.

    The kill feed prints each name under its brawler's portrait, so each
    sighting votes for a (player, slot) pair. Players never seen in the feed
    get the team's one remaining slot, if exactly one is left.
    """
    votes: dict[tuple[str, int], int] = {}
    for _, name, team, slot, _ in hits:
        for ident in identities:
            if ident.team == team and same_name(ident.name, name):
                votes[(ident.id, slot)] = votes.get((ident.id, slot), 0) + 1
    by_id = {i.id: i for i in identities}
    taken: dict[str, set[int]] = {"ally": set(), "enemy": set()}
    for (iid, slot), _ in sorted(votes.items(), key=lambda kv: -kv[1]):
        ident = by_id[iid]
        if ident.brawler is None and slot not in taken[ident.team]:
            ident.brawler = roster[ident.team][slot]["brawler"]
            ident.brawler_source = "kill_feed"
            ident.slot = slot
            taken[ident.team].add(slot)
    for team in ("ally", "enemy"):
        left = [k for k in range(len(roster[team])) if k not in taken[team]]
        unassigned = [i for i in identities if i.team == team and i.brawler is None]
        if len(left) == 1 and len(unassigned) == 1:
            unassigned[0].brawler = roster[team][left[0]]["brawler"]
            unassigned[0].brawler_source = "elimination"
            unassigned[0].slot = left[0]


def save_slots(path: Path, roster: dict[str, list[dict]]) -> None:
    """The six top-bar icons with the brawler each was identified as, for checking by eye."""
    tiles = []
    for team in ("ally", "enemy"):
        for s in roster[team]:
            icon = s["icon"] if s["icon"] is not None else np.zeros((48, 48, 3), np.uint8)
            tile = cv2.copyMakeBorder(cv2.resize(icon, (96, 96)), 0, 24, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
            mark = "" if s["confident"] else "?"
            cv2.putText(tile, f"{s['brawler']}{mark}", (3, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            tiles.append(tile)
    cv2.imwrite(str(path), np.hstack(tiles))


# --- positions and tracks -----------------------------------------------------------

@dataclass
class Observation:
    identity: str
    video_time: float
    match_time: float
    clock: str               # clock source (bar | inferred | extrapolated | uncertain)
    x: float                 # map tiles, in the diagram's orientation
    y: float
    hp: float | None


def detection_positions(dets: list[detect.Detection], times: list[timer.FrameTime], poses: list[localize.FramePose],
                        cam: localize.Camera, rotated: bool, ref: MapRef) -> dict[int, tuple[float, float, float]]:
    """Detection index -> (match time, x, y) in map tiles, where clock and camera are both known."""
    out = {}
    for i, d in enumerate(dets):
        ft, pose = times[d.frame], poses[d.frame]
        if ft.match_time is None or pose.status != "ok":
            continue
        vx, vy = cam.to_map(np.array([d.feet]), pose.x, pose.y)[0]
        x, y = (ref.cols - vx, ref.rows - vy) if rotated else (vx, vy)
        if -1 <= x <= ref.cols + 1 and -1 <= y <= ref.rows + 1:   # else a bad box or camera fix
            out[i] = (float(ft.match_time), float(x), float(y))
    return out


def observations(identities: list[Identity], dets: list[detect.Detection], times: list[timer.FrameTime],
                 positions: dict[int, tuple[float, float, float]]) -> list[Observation]:
    out = []
    for ident in identities:
        for p in ident.pieces:
            for i in p.dets:
                if i in positions:
                    mt, x, y = positions[i]
                    d = dets[i]
                    out.append(Observation(ident.id, d.video_time, mt, times[d.frame].source,
                                           round(x, 3), round(y, 3), d.hp))
    return out


def build_tracks(identities: list[Identity], obs: list[Observation], step: float = 0.1,
                 max_gap: float = 2.0) -> dict[str, gaps.Track]:
    if not obs:
        return {}
    t0 = min(o.match_time for o in obs)
    t1 = max(o.match_time for o in obs)
    tracks = {}
    for ident in identities:
        mine = [o for o in obs if o.identity == ident.id]
        if not mine:
            continue
        t = np.array([o.match_time for o in mine])
        x = np.array([o.x for o in mine])
        y = np.array([o.y for o in mine])
        keep = gaps.drop_spikes(t, x, y)
        tracks[ident.id] = gaps.resample(t[keep], x[keep], y[keep], step, max_gap, t_start=t0, t_end=t1)
    return tracks


def team_side(identities: list[Identity], tracks: dict[str, gaps.Track], ref: MapRef) -> str | None:
    """Which spawn (b or r, as drawn) the allies start from: their earliest seen positions."""
    spawns = {s: np.array([(c + 0.5, r + 0.5) for r, row in enumerate(ref.layout) for c, ch in enumerate(row) if ch == s])
              for s in "br"}
    if any(len(v) == 0 for v in spawns.values()):
        return None
    starts = []
    for ident in identities:
        tr = tracks.get(ident.id)
        if ident.team == "ally" and tr is not None:
            seen = np.flatnonzero(tr.flag == gaps.OBSERVED)
            if len(seen):
                starts.append((tr.x[seen[0]], tr.y[seen[0]]))
    if not starts:
        return None
    centre = np.mean(starts, axis=0)
    return min("br", key=lambda s: np.linalg.norm(spawns[s].mean(axis=0) - centre))


# --- debug replay ---------------------------------------------------------------

TEAM_BGR = {"me": (80, 255, 80), "ally": (255, 170, 40), "enemy": (60, 60, 255)}


def render_replay(video: Path, out_path: Path, ref: MapRef, identities: list[Identity],
                  tracks: dict[str, gaps.Track], dets: list[detect.Detection], times: list[timer.FrameTime],
                  poses: list[localize.FramePose], cam: localize.Camera, rotated: bool, fps: float) -> None:
    """Video frame (with identity labels) beside a top-down map with every player's position."""
    H = 720
    scale = H / ref.image.shape[0]
    base = cv2.resize(ref.image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    base = cv2.addWeighted(base, 0.6, np.full_like(base, 40), 0.4, 0)
    mw = base.shape[1]
    vw = 1000
    writer = None
    who = {i: ident for ident in identities for p in ident.pieces for i in p.dets}
    by_frame: dict[int, list[int]] = {}
    for i, d in enumerate(dets):
        by_frame.setdefault(d.frame, []).append(i)

    def to_px(x, y):
        return (int((ref.border + x * ref.tile_px) * scale), int((ref.border + y * ref.tile_px) * scale))

    for k, f in enumerate(iter_frames(video, fps=fps)):
        if k >= len(times):
            break
        panel = base.copy()
        ft, pose = times[k], poses[k]
        if pose.status == "ok":
            corners = cam.to_map(np.array([[0, 0], [cam.width, 0], [cam.width, cam.height], [0, cam.height]]), pose.x, pose.y)
            if rotated:
                corners = np.stack([ref.cols - corners[:, 0], ref.rows - corners[:, 1]], 1)
            cv2.polylines(panel, [np.array([to_px(*c) for c in corners], np.int32)], True, (200, 200, 200), 1)
        if ft.match_time is not None:
            for ident in identities:
                tr = tracks.get(ident.id)
                if tr is None:
                    continue
                j = int(round((ft.match_time - tr.t[0]) / (tr.t[1] - tr.t[0]))) if len(tr.t) > 1 else 0
                if not 0 <= j < len(tr.t) or tr.flag[j] == gaps.UNKNOWN:
                    continue
                c = TEAM_BGR["me" if ident.is_me else ident.team]
                pt = to_px(tr.x[j], tr.y[j])
                cv2.circle(panel, pt, 7, c, -1 if tr.flag[j] == gaps.OBSERVED else 2, cv2.LINE_AA)
                cv2.putText(panel, ident.label[:10], (pt[0] + 8, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
        frame = f.image.copy()
        for i in by_frame.get(k, []):
            d = dets[i]
            ident = who.get(i)
            c = TEAM_BGR["me" if ident and ident.is_me else d.team] if ident else (150, 150, 150)
            cv2.rectangle(frame, (int(d.x0), int(d.y0)), (int(d.x1), int(d.y1)), c, 2)
            cv2.putText(frame, ident.label if ident else f"? {d.track_id}", (int(d.x0), int(d.y1) + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
        small = cv2.resize(frame, (vw, int(frame.shape[0] * vw / frame.shape[1])), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((H, mw + vw, 3), np.uint8)
        canvas[:, :mw] = panel
        canvas[:small.shape[0], mw:] = small
        mt = "--" if ft.match_time is None else f"{int(ft.match_time // 60)}:{ft.match_time % 60:04.1f}"
        lines = [f"video {f.video_time:6.1f}s   match {mt}  (clock: {ft.source})",
                 f"camera: {pose.status}",
                 "map: filled = seen, ring = interpolated, hidden = unknown"]
        for n, line in enumerate(lines):
            cv2.putText(canvas, line, (mw + 12, small.shape[0] + 40 + 30 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (230, 230, 230), 1, cv2.LINE_AA)
        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas.shape[1], H))
        writer.write(canvas)
    if writer:
        writer.release()


# --- loading cached stages ----------------------------------------------------------

def load_times(path: Path) -> list[timer.FrameTime]:
    out = []
    with path.open() as f:
        for r in csv.DictReader(f):
            out.append(timer.FrameTime(float(r["video_time"]), None if r["match_time"] == "" else float(r["match_time"]),
                                       r["source"], int(r["segment"]),
                                       None if r["bar_px"] == "" else float(r["bar_px"]), r["repeat"] == "True"))
    return out


def load_poses(path: Path) -> list[localize.FramePose]:
    def num(v):
        return None if v == "" else float(v)
    with path.open() as f:
        return [localize.FramePose(float(r["video_time"]), num(r["x"]), num(r["y"]), num(r["raw_x"]), num(r["raw_y"]),
                                   float(r["score"]), r["status"]) for r in csv.DictReader(f)]


def run(video: Path, map_id: str, out_root: Path = Path("runs/match"), fps: float = 10.0,
        player: str | None = None, reuse: bool = False, replay: bool = True,
        out: Path | None = None, on_stage=None) -> dict:
    """on_stage(name) is called as each stage starts: clock, calibrate, camera,
    detect, names, brawlers, kill_feed (for progress reporting)."""
    out = out or out_root / video.stem
    out.mkdir(parents=True, exist_ok=True)
    ref = load_map(map_id)
    stage = on_stage or (lambda name: None)

    # 1. Clock
    stage("clock")
    if reuse and (out / "clock.csv").exists():
        times = load_times(out / "clock.csv")
    else:
        _, times, _ = timer.run(video, fps, out)

    # 2. Camera
    stage("calibrate")
    if reuse and (out / "poses.csv").exists() and (out / "camera.json").exists():
        poses = load_poses(out / "poses.csv")
        cj = json.loads((out / "camera.json").read_text())
        cam = localize.Camera(cj["width"], cj["height"], cj["kx"], cj["kv"], cj["p"])
        rotated = cj["rotated"]
    else:
        sample = [f.image for f in iter_frames(video, fps=0.5)]
        picks = [sample[i] for i in np.linspace(len(sample) * 0.1, len(sample) * 0.9, 16).astype(int)]
        cam, rotated, score = localize.calibrate(picks, ref)
        stage("camera")
        poses = localize.localize_video(video, ref, cam, rotated, fps)
        with (out / "poses.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(poses[0])))
            w.writeheader()
            w.writerows(asdict(p) for p in poses)
        (out / "camera.json").write_text(json.dumps(dict(asdict(cam), rotated=rotated, calib_score=score), indent=2))

    # 3. Boxes
    stage("detect")
    if reuse and (out / "detections.csv").exists():
        dets, fingerprints, crops = detect.load(out)
    else:
        dets, fingerprints, crops = detect.detect_video(video, fps=fps)
        detect.save(out, dets, fingerprints, crops)

    n = min(len(times), len(poses))
    if abs(len(times) - len(poses)) > 1:
        raise RuntimeError(f"clock has {len(times)} frames but camera has {len(poses)}")
    times, poses = times[:n], poses[:n]
    dets = [d for d in dets if d.frame < n]

    # 4. Identities, positions, tracks
    if player is None and PLAYER_FILE.exists():
        player = json.loads(PLAYER_FILE.read_text()).get("name")
    from pipeline import brawlers
    positions = detection_positions(dets, times, poses, cam, rotated, ref)
    stage("names")
    pieces = build_pieces(dets, fingerprints, crops, positions)
    identities, unassigned = link_identities(pieces, player, positions)
    for ident in identities:
        if ident.name:
            ident.name = final_label(ident, crops)
    hud_timeline: list = []
    stage("brawlers")
    roster = brawlers.hud_roster(video, fps=2.0, timeline=hud_timeline)
    with (out / "hud.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_time", "ally1", "ally2", "ally3", "enemy1", "enemy2", "enemy3"])
        for vt, states in hud_timeline:
            w.writerow([vt, *(int(s) for s in states)])
    stage("kill_feed")
    assign_brawlers(identities, roster, brawlers.kill_feed_names(video, roster))
    save_slots(out / "slots.png", roster)
    table = brawlers.load()
    obs = observations(identities, dets, times, positions)
    tracks = build_tracks(identities, obs)
    side = team_side(identities, tracks, ref)

    with (out / "observations.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(Observation.__dataclass_fields__))
        w.writeheader()
        w.writerows(asdict(o) for o in obs)
    with (out / "tracks.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["identity", "label", "team", "is_me", "brawler", "match_time", "x", "y", "flag"])
        for ident in identities:
            tr = tracks.get(ident.id)
            if tr is None:
                continue
            for j in range(len(tr.t)):
                w.writerow([ident.id, ident.label, ident.team, ident.is_me, ident.brawler or "", round(float(tr.t[j]), 2),
                            "" if np.isnan(tr.x[j]) else round(float(tr.x[j]), 3),
                            "" if np.isnan(tr.y[j]) else round(float(tr.y[j]), 3), tr.flag[j]])
    summary = dict(
        video=str(video), map=map_id, ally_spawn=side, rotated=rotated, player=player,
        me_found=any(i.is_me for i in identities),
        roster={team: [dict(brawler=s["brawler"], score=s["score"], confident=s["confident"]) for s in roster[team]]
                for team in roster},
        identities=[dict(id=i.id, team=i.team, name=i.name, is_me=i.is_me,
                         brawler=i.brawler, brawler_source=i.brawler_source, slot=i.slot,
                         attack_range=table.get(i.brawler, {}).get("range") if i.brawler else None,
                         range_forms=table.get(i.brawler, {}).get("forms") if i.brawler else None,
                         pieces=len(i.pieces),
                         detections=sum(len(p.dets) for p in i.pieces),
                         **({k: int((tracks[i.id].flag == k).sum()) for k in (gaps.OBSERVED, gaps.INTERPOLATED, gaps.UNKNOWN)}
                            if i.id in tracks else {}))
                    for i in identities],
        unassigned=dict(pieces=len(unassigned), detections=sum(len(p.dets) for p in unassigned)),
    )
    (out / "identities.json").write_text(json.dumps(summary, indent=2))
    if replay:
        render_replay(video, out / "replay.mp4", ref, identities, tracks, dets, times, poses, cam, rotated, fps)
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Player tracks in map tiles for a replay recording.")
    ap.add_argument("video", type=Path)
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/match"))
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--player", help="your in-game name (default: data/player.json)")
    ap.add_argument("--reuse", action="store_true", help="reuse clock/camera/detections already in the output folder")
    ap.add_argument("--no-replay", action="store_true")
    args = ap.parse_args(argv)
    s = run(args.video, args.map, args.out, args.fps, args.player, args.reuse, not args.no_replay)
    print(f"{Path(s['video']).name}: allies start at the {'blue' if s['ally_spawn'] == 'b' else 'red'} spawn"
          if s["ally_spawn"] else "team side unknown")
    for i in s["identities"]:
        tag = "  <- you" if i["is_me"] else ""
        b = f"{i['brawler']} ({i['attack_range']} tiles, {i['brawler_source']})" if i["brawler"] else "brawler ?"
        print(f"  {i['id']:7s} {str(i['name']):16s} {b:34s} pieces {i['pieces']:3d}  "
              f"steps observed {i.get('observed', 0):4d} / interpolated {i.get('interpolated', 0):4d} / "
              f"unknown {i.get('unknown', 0):4d}{tag}")
    print(f"  unassigned: {s['unassigned']['pieces']} pieces, {s['unassigned']['detections']} detections")
    if not s["me_found"]:
        print(f"  warning: no ally name matched '{s['player']}'")
    print(f"-> {args.out / Path(s['video']).stem}")


if __name__ == "__main__":
    main()
