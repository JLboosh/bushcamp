"""Brawler reference data and recognition.

Reference data (data/brawlers/):
    brawlers.json    name -> attack range (tiles), class, wiki page; fetched
                     from the Brawl Stars wiki (brawlstars.fandom.com, CC BY-SA)
    overrides.json   hand-maintained corrections, applied on top (the brief's
                     hand-kept range table); wins over the fetched data.
                     "forms" lists per-form ranges for brawlers that switch
                     (range = the longest); "available": false hides a
                     brawler that's no longer in the game
    portraits/       one portrait per brawler, the art used for in-game icons

Brawlify (the user's preferred source) blocks automated requests, so the
wiki is used for now; both can be swapped for an API later.

Recognition:
    hud_slots()       the six portrait squares at the top of the screen
    identify()        which brawler a portrait crop shows

    python -m pipeline.brawlers fetch            # (re)download reference data
    python -m pipeline.brawlers show Colt Piper  # print ranges
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "brawlers"
WIKI_API = "https://brawlstars.fandom.com/api.php"
USER_AGENT = "bushcamp/0.1 (personal Brawl Stars replay analyzer; low-volume reference fetch)"
PORTRAIT_PX = 128


# --- fetching -------------------------------------------------------------------

def _get(url: str, params: dict | None = None, pause: float = 0.5) -> bytes:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    time.sleep(pause)                 # be gentle with the wiki
    return data


def _api(**params) -> dict:
    return json.loads(_get(WIKI_API, dict(params, format="json")))


def parse_range(wikitext: str) -> tuple[float | None, str | None]:
    """Main attack range from a brawler page's `|AttackRange = 9 (Long)` field
    (the page also has SuperRange and gadget ranges)."""
    m = re.search(r"\|\s*AttackRange\s*=\s*([^|\n<}]*)", wikitext)
    if not m:
        return None, None
    text = m.group(1).strip()
    num = re.search(r"\d+(?:\.\d+)?", text)
    return (float(num.group()) if num else None), text


def parse_class(wikitext: str) -> str | None:
    m = re.search(r"\|\s*Class\s*=\s*\[*([A-Za-z ]+)", wikitext)
    return m.group(1).strip() if m else None


def fetch(dest: Path = DATA, portraits: bool = True) -> dict:
    """Download names, ranges, classes and portraits for every brawler on the wiki."""
    dest.mkdir(parents=True, exist_ok=True)
    names = [m["title"] for m in _api(action="query", list="categorymembers", cmtitle="Category:Brawlers",
                                       cmlimit=500, cmnamespace=0)["query"]["categorymembers"]]
    info: dict[str, dict] = {}
    for i in range(0, len(names), 50):
        batch = names[i:i + 50]
        pages = _api(action="query", prop="revisions", rvprop="content", rvslots="main",
                     titles="|".join(batch))["query"]["pages"]
        for pg in pages.values():
            text = pg.get("revisions", [{}])[0].get("slots", {}).get("main", {}).get("*", "")
            rng, rng_text = parse_range(text)
            info[pg["title"]] = dict(range=rng, range_text=rng_text, cls=parse_class(text),
                                     wiki=f"https://brawlstars.fandom.com/wiki/{urllib.parse.quote(pg['title'].replace(' ', '_'))}")
    if portraits:
        pdir = dest / "portraits"
        pdir.mkdir(exist_ok=True)
        for i in range(0, len(names), 50):
            files = [f"File:{n} Portrait.png" for n in names[i:i + 50]]
            pages = _api(action="query", prop="imageinfo", iiprop="url", titles="|".join(files))["query"]["pages"]
            for pg in pages.values():
                name = pg["title"][len("File:"):-len(" Portrait.png")]
                url = pg.get("imageinfo", [{}])[0].get("url")
                if not url or name not in info:
                    continue
                arr = np.frombuffer(_get(url, pause=0.3), np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                if img.ndim == 3 and img.shape[2] == 4:          # flatten transparency onto white
                    a = img[..., 3:4] / 255.0
                    img = (img[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
                scale = PORTRAIT_PX / max(img.shape[:2])
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(pdir / f"{_slug(name)}.png"), img)
                info[name]["portrait"] = f"portraits/{_slug(name)}.png"
    out = dict(source="Brawl Stars wiki (brawlstars.fandom.com), CC BY-SA 3.0",
               fetched=datetime.date.today().isoformat(), brawlers=info)
    (dest / "brawlers.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    if not (dest / "overrides.json").exists():
        (dest / "overrides.json").write_text(json.dumps(
            {"_note": "Hand-kept corrections; each entry overrides fields from brawlers.json, "
                      "e.g. {\"Colt\": {\"range\": 9.0}}"}, indent=2) + "\n")
    return out


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load(dest: Path = DATA) -> dict[str, dict]:
    """Brawler table with hand overrides applied."""
    table = json.loads((dest / "brawlers.json").read_text())["brawlers"]
    over = dest / "overrides.json"
    if over.exists():
        for name, fields in json.loads(over.read_text()).items():
            if not name.startswith("_"):
                table.setdefault(name, {}).update(fields)
    return table


# --- recognition ------------------------------------------------------------------

ICON = 48                      # comparison size (px)
# Top-bar portrait slots at 720p: own team from the left edge, enemies from the
# right edge (x of the 52 px frames), measured on the replay viewer.
HUD_LEFT_X = (102, 172, 240)
HUD_RIGHT_FROM_EDGE = (214, 144, 76)
HUD_Y, HUD_SIZE = 12, 52


def hud_slots(width: int, height: int) -> dict[str, list[tuple[int, int, int, int]]]:
    """Inner icon boxes (x0, y0, x1, y1) of the six top-bar portraits: {'ally': [...], 'enemy': [...]}."""
    s = height / 720
    inset = round(3 * s)
    def box(x):
        x, y, n = round(x * s), round(HUD_Y * s), round(HUD_SIZE * s)
        return (x + inset, y + inset, x + n - inset, y + n - inset)
    return {"ally": [box(x) for x in HUD_LEFT_X],
            "enemy": [box(width / s - d) for d in HUD_RIGHT_FROM_EDGE]}


def slot_is_clear(image: np.ndarray, box: tuple[int, int, int, int], team: str) -> bool:
    """The slot's coloured frame is intact (a dead player's slot is dimmed and crossed out)."""
    x0, y0, x1, y1 = box
    pad = max(2, (x1 - x0) // 16)
    ring = np.concatenate([image[y0 - pad:y0, x0:x1].reshape(-1, 3), image[y1:y1 + pad, x0:x1].reshape(-1, 3)])
    hsv = cv2.cvtColor(ring[None], cv2.COLOR_BGR2HSV)[0]
    h, sat, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    colour = ((h >= 100) & (h <= 125)) if team == "ally" else ((h >= 170) | (h <= 5))
    return float((colour & (sat > 120) & (v > 120)).mean()) > 0.6


def _features(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.resize(img, (ICON, ICON), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0)
    gray = (gray - gray.mean()) / (gray.std() + 1e-6)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], (hsv[..., 1] > 60).astype(np.uint8), [18, 4], [0, 180, 0, 256])
    return gray, cv2.normalize(hist, None).flatten()


_REFS: list[tuple[str, np.ndarray, np.ndarray]] | None = None


def _references(dest: Path = DATA) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Square crops of every portrait (the in-game icon is a square cut of the same art)."""
    global _REFS
    if _REFS is None:
        _REFS = []
        for name, v in load(dest).items():
            if "portrait" not in v or v.get("available") is False:
                continue
            img = cv2.imread(str(dest / v["portrait"]))
            h, w = img.shape[:2]
            n = min(h, w)
            for fx, fy, f in ((0.5, 0.5, 1.0), (0.0, 0.5, 1.0), (1.0, 0.5, 1.0), (0.5, 0.0, 1.0),
                              (0.5, 0.3, 0.85), (0.4, 0.3, 0.75), (0.6, 0.3, 0.75)):
                m = int(n * f)
                x0, y0 = int((w - m) * fx), int((h - m) * fy)
                _REFS.append((name, *_features(img[y0:y0 + m, x0:x0 + m])))
    return _REFS


HUD_LIBRARY = DATA / "hud_icons"     # in-game icons, confirmed: <slug>__<n>.png
CONFIDENT = dict(score=0.6, margin=0.05)


def _library() -> list[tuple[str, np.ndarray]]:
    out = []
    if HUD_LIBRARY.exists():
        names = {_slug(n): n for n in load()}
        for f in sorted(HUD_LIBRARY.glob("*.png")):
            name = names.get(f.stem.split("__")[0])
            if name:
                out.append((name, _features(cv2.imread(str(f)))[0]))
    return out


def identify(icon: np.ndarray, top: int = 3) -> list[tuple[str, float]]:
    """Best-matching brawlers for a portrait crop, as (name, score), best first.

    Against the wiki portraits, score mixes shape (grayscale correlation) and
    colour (hue/saturation histogram). In-game icons already confirmed (the
    library) are the same render as the crop, so a close match there wins.
    """
    gray, hist = _features(icon)
    best: dict[str, float] = {}
    for name, g, h in _references():
        score = 0.6 * float((gray * g).mean()) + 0.4 * float(cv2.compareHist(hist, h, cv2.HISTCMP_CORREL))
        best[name] = max(best.get(name, -1.0), score)
    for name, g in _library():
        score = float((gray * g).mean())
        if score > 0.85:
            best[name] = max(best.get(name, -1.0), 1.0 + score)   # above any wiki-portrait score
    return sorted(best.items(), key=lambda kv: -kv[1])[:top]


def learn(icon: np.ndarray, name: str) -> Path:
    """Save an in-game icon as a confirmed example of `name`."""
    if name not in load():
        raise KeyError(f"unknown brawler {name!r}")
    HUD_LIBRARY.mkdir(parents=True, exist_ok=True)
    n = len(list(HUD_LIBRARY.glob(f"{_slug(name)}__*.png")))
    path = HUD_LIBRARY / f"{_slug(name)}__{n}.png"
    cv2.imwrite(str(path), cv2.resize(icon, (ICON, ICON), interpolation=cv2.INTER_AREA))
    return path


def hud_roster(video, fps: float = 1.0, timeline: list | None = None) -> dict[str, list[dict]]:
    """Brawler in each top-bar slot, averaged over the recording.

    Returns {'ally': [slot, slot, slot], 'enemy': [...]}, each slot
    {brawler, score, runner_up, confident, frames, icon}: icon is the
    clearest crop (kept for learning and for matching the kill feed).

    If `timeline` is a list, (video_time, [clear?] * 6) is appended for every
    sampled frame (ally slots 1-3, then enemy 1-3): a slot stops being clear
    when its player is eliminated, until the round ends.
    """
    from pipeline.sampling import iter_frames
    sums: dict[tuple[str, int], dict[str, list[float]]] = {}
    icons: dict[tuple[str, int], tuple[float, np.ndarray]] = {}
    for f in iter_frames(video, fps=fps):
        img = f.image
        states = []
        for team, boxes in hud_slots(img.shape[1], img.shape[0]).items():
            for k, b in enumerate(boxes):
                clear = slot_is_clear(img, b, team)
                states.append(clear)
                if not clear:
                    continue
                x0, y0, x1, y1 = b
                crop = img[y0:y1, x0:x1]
                for name, sc in identify(crop, top=8):
                    sums.setdefault((team, k), {}).setdefault(name, []).append(sc)
                sat = float(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[..., 1].mean())
                if sat > icons.get((team, k), (-1.0, None))[0]:
                    icons[(team, k)] = (sat, crop.copy())
        if timeline is not None:
            timeline.append((round(f.video_time, 3), states))
    out: dict[str, list[dict]] = {"ally": [], "enemy": []}
    for team in ("ally", "enemy"):
        for k in range(3):
            scores = sums.get((team, k), {})
            n = max((len(v) for v in scores.values()), default=0)
            ranked = sorted(((sum(v) / n, name) for name, v in scores.items()), reverse=True) if n else []
            top = ranked[0] if ranked else (0.0, None)
            second = ranked[1] if len(ranked) > 1 else (0.0, None)
            out[team].append(dict(
                brawler=top[1], score=round(top[0], 3), runner_up=second[1],
                confident=bool(top[1]) and (top[0] >= 1.0 or (top[0] >= CONFIDENT["score"] and
                                                             top[0] - second[0] >= CONFIDENT["margin"])),
                frames=n, icon=icons.get((team, k), (0, None))[1]))
    return out


# Kill feed: blue panel (own team got the kill) top left, red panel top right,
# below the portrait bar. Each entry: killer portrait + name, victim portrait
# + name, the names printed just under the portraits.
KILLFEED_Y = (0.15, 0.45)        # fraction of height searched
KILLFEED_W = 0.24                # fraction of width from each side


def _icon_score(a: np.ndarray, b: np.ndarray) -> float:
    ga, ha = _features(a)
    gb, hb = _features(b)
    return 0.6 * float((ga * gb).mean()) + 0.4 * float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))


def kill_feed_names(video, roster: dict[str, list[dict]], fps: float = 2.0,
                    min_score: float = 0.5) -> list[tuple[float, str, str, int, float]]:
    """(video time, name read, team, slot index, match score) for every name
    seen in the kill feed, with the top-bar slot whose icon matches the
    portrait printed above that name."""
    from pipeline.sampling import iter_frames
    from pipeline.track import read_name_boxes   # OCR with word positions
    slots = [(team, k, s["icon"]) for team in ("ally", "enemy") for k, s in enumerate(roster[team])
             if s["icon"] is not None]
    out = []
    for f in iter_frames(video, fps=fps):
        img = f.image
        H, W = img.shape[:2]
        s = H / 720
        y0, y1 = int(KILLFEED_Y[0] * H), int(KILLFEED_Y[1] * H)
        for side, x0, x1 in (("left", 0, int(KILLFEED_W * W)), ("right", int((1 - KILLFEED_W) * W), W)):
            region = img[y0:y1, x0:x1]
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            panel = ((hsv[..., 0] >= 100) & (hsv[..., 0] <= 125) if side == "left"
                     else (hsv[..., 0] >= 170) | (hsv[..., 0] <= 5)) & (hsv[..., 1] > 150) & (hsv[..., 2] > 60)
            if panel.mean() < 0.08:          # no kill-feed entry showing
                continue
            for text, conf, (bx0, by0, bx1, by1) in read_name_boxes(region):
                if conf < 0.7:
                    continue
                cx = x0 + (bx0 + bx1) / 2
                size = int(58 * s)
                py1 = y0 + by0 - int(2 * s)
                px0 = int(cx - size / 2)
                if py1 - size < 0 or px0 < 0 or px0 + size > W:
                    continue
                crop = img[py1 - size:py1, px0:px0 + size]
                inner = crop[int(5 * s):-int(5 * s), int(5 * s):-int(5 * s)]
                scored = sorted(((_icon_score(inner, icon), team, k) for team, k, icon in slots), reverse=True)
                if scored and scored[0][0] >= min_score:
                    out.append((f.video_time, text, scored[0][1], scored[0][2], round(scored[0][0], 3)))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Brawler reference data.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--no-portraits", action="store_true")
    s = sub.add_parser("show")
    s.add_argument("names", nargs="*")
    r = sub.add_parser("roster", help="identify the six brawlers in a recording")
    r.add_argument("video", type=Path)
    lr = sub.add_parser("learn", help="confirm a slot's brawler, e.g. learn video.mp4 enemy 3 Gray")
    lr.add_argument("video", type=Path)
    lr.add_argument("team", choices=["ally", "enemy"])
    lr.add_argument("slot", type=int, help="1-3, left to right")
    lr.add_argument("name")
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        out = fetch(portraits=not args.no_portraits)
        b = out["brawlers"]
        print(f"{len(b)} brawlers; {sum(v['range'] is not None for v in b.values())} with a range, "
              f"{sum('portrait' in v for v in b.values())} portraits -> {DATA}")
    elif args.cmd in ("roster", "learn"):
        roster = hud_roster(args.video)
        if args.cmd == "learn":
            icon = roster[args.team][args.slot - 1]["icon"]
            if icon is None:
                raise SystemExit("that slot was never clearly visible")
            print(f"saved {learn(icon, args.name)}")
            return
        for team in ("ally", "enemy"):
            for k, s in enumerate(roster[team], 1):
                flag = "" if s["confident"] else f"   <- unsure (next best: {s['runner_up']}); confirm with: learn {team} {k} <name>"
                print(f"{team:5s} {k}: {s['brawler']!s:12s} score {s['score']:.2f}  ({s['frames']} frames){flag}")
    else:
        table = load()
        for n in args.names or sorted(table):
            v = table.get(n, {})
            print(f"{n:14s} range {v.get('range')!s:6s} ({v.get('range_text')})  class {v.get('cls')}")


if __name__ == "__main__":
    main()
