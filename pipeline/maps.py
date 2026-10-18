"""Map reference data: tile layout and coordinate system for each supported map.

Each map lives in maps/<map_id>/ with
    map.json     name, reference image, grid size
    <image>      top-down diagram (from Brawlify / the Brawl Stars wiki)
    layout.txt   one character per tile, generated from the image and
                 hand-editable; once it exists it's used as-is.

Coordinates are in tiles: x to the right (0..21), y down (0..33) as the
diagram is drawn, so the tile in column c, row r covers [c, c+1) x [r, r+1).
In-game ranges are also in tiles, so distances need no conversion.

Tile codes:
    .  floor          B  bush           ~  water
    W  wall           O  other obstacle (barrels, hedges, crates: blocks movement)
    b  blue spawn     r  red spawn      (spawn pads are floor you can walk on)

CLI (build layout.txt and a check image for every map that lacks one):
    python -m pipeline.maps
    python -m pipeline.maps --rebuild opening_move
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MAPS_DIR = Path(__file__).resolve().parent.parent / "maps"
CODES = {".": "floor", "B": "bush", "~": "water", "W": "wall", "O": "obstacle", "b": "blue spawn", "r": "red spawn"}
COLORS = {".": (90, 70, 80), "B": (200, 180, 40), "~": (230, 150, 60), "W": (80, 80, 220), "O": (40, 40, 40),
          "b": (255, 120, 0), "r": (0, 0, 255)}


@dataclass
class MapRef:
    id: str
    name: str
    image: np.ndarray          # BGR diagram
    cols: int
    rows: int
    border: int                # px of margin around the grid in the diagram
    tile_px: int               # px per tile in the diagram
    layout: list[str]          # rows of tile codes
    bush_raise: float = 0.52   # tiles the game draws bushes above their tile (see pipeline/localize.py)
    ground_check: str | None = None   # flat feature to measure accuracy against: "checker" or "water"
    attack_axis_override: str | None = None   # "x" or "y": direction of play (see attack_axis)

    def mask(self, codes: str, px_per_tile: int = 8, pad: int = 0) -> np.ndarray:
        """Binary image (uint8 0/1) of tiles with any of `codes`, at px_per_tile,
        with `pad` empty tiles added on every side."""
        grid = np.array([[ch in codes for ch in row] for row in self.layout], np.uint8)
        grid = np.pad(grid, pad)
        return cv2.resize(grid, None, fx=px_per_tile, fy=px_per_tile, interpolation=cv2.INTER_NEAREST)

    def clear_shot(self, ax, ay, bx, by, samples: int = 128) -> np.ndarray:
        """True where a straight shot from a to b crosses no wall or obstacle tile.

        Bushes, water and spawn pads don't block shots. Works on arrays (one
        result per pair); NaN coordinates give False.
        """
        ax, ay, bx, by = (np.atleast_1d(np.asarray(v, float)) for v in (ax, ay, bx, by))
        if not hasattr(self, "_blocks"):
            self._blocks = np.array([[ch in "WO" for ch in row] for row in self.layout])
        s = np.linspace(0.0, 1.0, samples)[1:-1, None]           # skip the two ends (the players' own tiles)
        xs, ys = ax + s * (bx - ax), ay + s * (by - ay)
        bad = np.isnan(xs) | np.isnan(ys)
        c = np.floor(np.where(bad, -1, xs)).astype(int)
        r = np.floor(np.where(bad, -1, ys)).astype(int)
        inside = (c >= 0) & (c < self.cols) & (r >= 0) & (r < self.rows)
        hit = np.zeros(xs.shape, bool)
        hit[inside] = self._blocks[r[inside], c[inside]]
        ok = ~hit.any(axis=0)
        return ok & ~(np.isnan(ax) | np.isnan(ay) | np.isnan(bx) | np.isnan(by))

    def attack_axis(self, own_spawn: np.ndarray, enemy_spawn: np.ndarray) -> np.ndarray:
        """Unit vector pointing from your side towards the enemy's, along the map's
        long direction of play: x or y, whichever separates the spawns more
        relative to the map's size (spawns sit in corners, so the straight line
        between them is diagonal and would count sideways moves as pushing).
        map.json "attack_axis" ("x" or "y") overrides."""
        d = enemy_spawn - own_spawn
        axis = self.attack_axis_override or ("x" if abs(d[0]) / self.cols > abs(d[1]) / self.rows else "y")
        k = 0 if axis == "x" else 1
        v = np.zeros(2)
        v[k] = 1.0 if d[k] >= 0 else -1.0
        return v

    def tile(self, x: float, y: float) -> str | None:
        c, r = int(np.floor(x)), int(np.floor(y))
        return self.layout[r][c] if 0 <= r < self.rows and 0 <= c < self.cols else None


def classify_tiles(image: np.ndarray, cols: int, rows: int, border: int, tile_px: int) -> list[str]:
    """Tile codes from the diagram's colours.

    The diagram draws raised things (walls, obstacles, bushes) half a tile
    higher than the tile they stand on, to suggest height; flat things (floor,
    water, spawn pads) are drawn in place. So each tile is sampled twice: at
    its centre for flat things and half a tile up for raised ones. Floor is the
    most common colour. Spawn pads are found as circles, since their colours
    resemble walls and bushes.
    """
    spawns = find_spawns(image, border, tile_px)
    # Spawn circles are bigger than a tile: paint them out so they don't leak
    # into the neighbouring tiles' samples.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    floor_px = np.median(hsv[border:-border:7, border:-border:7].reshape(-1, 3), axis=0)
    for _, _, (x, y, rad) in spawns:
        cv2.circle(hsv, (int(x), int(y)), int(rad) + 3, floor_px.tolist(), -1)
    q, half = tile_px // 3, tile_px // 2

    def sample(y0, y1, x0, x1):
        return np.median(hsv[max(border, y0):y1, x0:x1].reshape(-1, 3), axis=0)

    flat = np.zeros((rows, cols, 3), np.float32)
    raised = np.zeros((rows, cols, 3), np.float32)
    for r in range(rows):
        for c in range(cols):
            y, x = border + r * tile_px, border + c * tile_px
            flat[r, c] = sample(y + q, y + tile_px - q, x + q, x + tile_px - q)
            raised[r, c] = sample(y - half + q, y + half - q, x + q, x + tile_px - q)
    floor = np.median(flat.reshape(-1, 3), axis=0)   # the floor covers most tiles

    def is_floor(h, s, v):
        dh = min(abs(h - floor[0]), 180 - abs(h - floor[0]))
        return dh < 12 and abs(s - floor[1]) < 45 and abs(v - floor[2]) < 30

    def is_teal(h, s):
        return 90 <= h <= 106 and s >= 170

    out = [[""] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            fh, fs, fv = flat[r, c]
            h, s, v = raised[r, c]
            if is_teal(fh, fs) and fv >= 210:
                out[r][c] = "~"                       # water: flat, bright teal-blue
            elif is_floor(h, s, v) or is_floor(fh, fs, fv):
                # An object fills both samples; one alone is a neighbour's
                # shadow or a tall sprite reaching into this tile.
                out[r][c] = "."
            elif is_teal(h, s):
                out[r][c] = "B"
            elif (h >= 165 or h <= 5) and s >= 100 and v >= 175:
                out[r][c] = "W"
            else:
                out[r][c] = "O"
    for (c, r), team, _ in spawns:
        if 0 <= r < rows and 0 <= c < cols:
            out[r][c] = team
    return ["".join(row) for row in out]


def find_spawns(image: np.ndarray, border: int, tile_px: int) -> list[tuple[tuple[int, int], str, tuple]]:
    """Spawn pads: circles about 1.5 tiles across, filled with team colour.

    Returns ((col, row), team, (x, y, radius)); team 'b' (blue) or 'r' (red).
    Other round things (e.g. octagon pads with a blue centre) are mostly grey
    inside, so they fail the "filled with colour" test.
    """
    gray = cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=tile_px * 1.5, param1=80, param2=30,
                               minRadius=int(tile_px * 0.6), maxRadius=int(tile_px * 0.95))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    out = []
    for x, y, rad in ([] if circles is None else circles[0]):
        disc = np.zeros(gray.shape, np.uint8)
        cv2.circle(disc, (int(x), int(y)), int(rad * 0.9), 1, -1)
        px = hsv[disc == 1]
        coloured = px[px[:, 1] > 90]
        if len(coloured) < 0.75 * len(px):
            continue
        hue = np.median(coloured[:, 0])
        team = "b" if 95 <= hue <= 125 else "r" if hue >= 160 or hue <= 10 else None
        if team:
            out.append(((int((x - border) // tile_px), int((y - border) // tile_px)), team,
                        (float(x), float(y), float(rad))))
    return out


def load(map_id: str, maps_dir: Path = MAPS_DIR) -> MapRef:
    d = maps_dir / map_id
    meta = json.loads((d / "map.json").read_text())
    img = cv2.imread(str(d / meta["image"]), cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4:   # flatten transparency onto white
        a = img[..., 3:4] / 255.0
        img = (img[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
    layout_file = d / "layout.txt"
    if layout_file.exists():
        layout = [line for line in layout_file.read_text().splitlines() if line and not line.startswith("#")]
    else:
        layout = classify_tiles(img, meta["cols"], meta["rows"], meta["border"], meta["tile_px"])
    ref = MapRef(map_id, meta["name"], img, meta["cols"], meta["rows"], meta["border"], meta["tile_px"], layout,
                 meta.get("bush_raise", 0.52), meta.get("ground_check"), meta.get("attack_axis"))
    if len(layout) != ref.rows or any(len(r) != ref.cols for r in layout):
        raise ValueError(f"{layout_file}: expected {ref.rows} rows of {ref.cols} tiles")
    return ref


def save_layout(ref: MapRef, maps_dir: Path = MAPS_DIR) -> Path:
    path = maps_dir / ref.id / "layout.txt"
    legend = "# " + "  ".join(f"{k} {v}" for k, v in CODES.items())
    path.write_text(f"# {ref.name}: {ref.cols} x {ref.rows} tiles, row 0 at the top of the diagram\n"
                    f"{legend}\n" + "\n".join(ref.layout) + "\n")
    return path


def draw_layout(ref: MapRef) -> np.ndarray:
    """The diagram with each tile's code tinted on top and a tile grid, for checking by eye."""
    out = ref.image.copy()
    over = out.copy()
    t, b = ref.tile_px, ref.border
    for r, row in enumerate(ref.layout):
        for c, ch in enumerate(row):
            if ch != ".":
                cv2.rectangle(over, (b + c * t, b + r * t), (b + (c + 1) * t - 1, b + (r + 1) * t - 1), COLORS[ch], -1)
    out = cv2.addWeighted(over, 0.45, out, 0.55, 0)
    for c in range(ref.cols + 1):
        cv2.line(out, (b + c * t, b), (b + c * t, b + ref.rows * t), (255, 255, 255), 1)
    for r in range(ref.rows + 1):
        cv2.line(out, (b, b + r * t), (b + ref.cols * t, b + r * t), (255, 255, 255), 1)
    for r, row in enumerate(ref.layout):
        for c, ch in enumerate(row):
            if ch != ".":
                cv2.putText(out, ch, (b + c * t + 10, b + r * t + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build tile layouts for the reference maps.")
    ap.add_argument("--rebuild", nargs="*", default=None, help="regenerate layout.txt for these maps")
    args = ap.parse_args(argv)
    for d in sorted(p for p in MAPS_DIR.iterdir() if (p / "map.json").exists()):
        if args.rebuild is not None and d.name in args.rebuild:
            (d / "layout.txt").unlink(missing_ok=True)
        ref = load(d.name)
        if not (d / "layout.txt").exists():
            print(f"{d.name}: wrote {save_layout(ref)}")
        cv2.imwrite(str(d / "layout_check.png"), draw_layout(ref))
        counts = {CODES[k]: sum(row.count(k) for row in ref.layout) for k in CODES}
        print(f"{d.name}: {counts}")


if __name__ == "__main__":
    main()
