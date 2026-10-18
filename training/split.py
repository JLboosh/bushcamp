"""Split a labelled export into train/val/test *by match*.

Frames from one match never land in two splits; otherwise near-identical
frames would leak into the test set and make the detector look better than
it is. Roboflow's own split is ignored.

Input: one or more unzipped Roboflow "YOLOv8" exports (any mix of
train/valid/test folders, each with images/ and labels/). Roboflow renames
files to <original>_jpg.rf.<hash>.jpg; the original name carries the match id
(<match>_t<ms>), which is how frames are grouped.

    python -m training.split data/labeled/batch1 --out data/dataset
    python -m training.split data/labeled/* --test 1000075101 --val Screenrecording_20261013_222328_com
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

ROBOFLOW_SUFFIX = re.compile(r"_(jpg|jpeg|png)\.rf\.[0-9a-f]+$", re.IGNORECASE)
FRAME_NAME = re.compile(r"^(?P<match>.+)_t\d{7}$")
SPLITS = ("train", "val", "test")


def original_stem(path: Path) -> str:
    return ROBOFLOW_SUFFIX.sub("", path.stem)


def match_id(stem: str) -> str:
    m = FRAME_NAME.match(stem)
    if not m:
        raise ValueError(f"can't find a match id in {stem!r} (expected <match>_t<7 digits>)")
    return m["match"]


def collect(exports: list[Path]) -> dict[str, tuple[Path, Path | None]]:
    """original stem -> (image, label). Later exports override earlier ones."""
    frames: dict[str, tuple[Path, Path | None]] = {}
    for export in exports:
        for img in sorted(export.rglob("*")):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"} or img.parent.name != "images":
                continue
            label = img.parent.parent / "labels" / f"{img.stem}.txt"
            frames[original_stem(img)] = (img, label if label.exists() else None)
    return frames


def assign(matches: list[str], test: list[str], val: list[str], seed: int = 0,
           test_frac: float = 0.2, val_frac: float = 0.15) -> dict[str, str]:
    """match -> split. Explicit lists win; the rest are shuffled with a fixed seed."""
    unknown = (set(test) | set(val)) - set(matches)
    if unknown:
        raise ValueError(f"not in the export: {sorted(unknown)}")
    out = {m: "test" for m in test} | {m: "val" for m in val}
    rest = sorted(m for m in matches if m not in out)
    random.Random(seed).shuffle(rest)
    n = len(matches)
    need_test = max(1, round(test_frac * n)) - len(test)
    need_val = max(1, round(val_frac * n)) - len(val)
    for m in rest:
        if need_test > 0:
            out[m], need_test = "test", need_test - 1
        elif need_val > 0:
            out[m], need_val = "val", need_val - 1
        else:
            out[m] = "train"
    if "train" not in out.values():
        raise ValueError(f"only {n} matches: need at least 3 for train/val/test")
    return out


def write_dataset(frames: dict[str, tuple[Path, Path | None]], split_of: dict[str, str],
                  out: Path, names: list[str]) -> dict[str, dict]:
    if out.exists():
        if not (out / "split.json").exists() and any(out.iterdir()):
            raise FileExistsError(f"{out} exists and wasn't made by this script; not overwriting")
        shutil.rmtree(out)
    summary = {s: {"matches": sorted(m for m, v in split_of.items() if v == s), "images": 0, "boxes": 0}
               for s in SPLITS}
    for stem, (img, label) in sorted(frames.items()):
        split = split_of[match_id(stem)]
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, out / split / "images" / f"{stem}{img.suffix.lower()}")
        text = label.read_text() if label else ""   # no label file = image with no brawlers
        (out / split / "labels" / f"{stem}.txt").write_text(text)
        summary[split]["images"] += 1
        summary[split]["boxes"] += sum(1 for line in text.splitlines() if line.strip())
    (out / "split.json").write_text(json.dumps(summary, indent=2))
    (out / "dataset.yaml").write_text(dataset_yaml(out, names))
    return summary


def dataset_yaml(root: Path, names: list[str]) -> str:
    # Absolute path: Ultralytics resolves relative ones against its own datasets folder.
    lines = [f"path: {root.resolve()}", "train: train/images", "val: val/images", "test: test/images",
             "names:"] + [f"  {i}: {n}" for i, n in enumerate(names)]
    return "\n".join(lines) + "\n"


def read_names(exports: list[Path]) -> list[str]:
    """Class names from the export's data.yaml (Roboflow writes `names: ['ally', 'enemy']`)."""
    import yaml
    for export in exports:
        for f in export.rglob("data.yaml"):
            names = yaml.safe_load(f.read_text()).get("names")
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names)]
            if names:
                return list(names)
    raise FileNotFoundError("no data.yaml with class names in the export")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Split labelled frames into train/val/test by match.")
    ap.add_argument("exports", nargs="+", type=Path, help="unzipped Roboflow YOLOv8 export folder(s)")
    ap.add_argument("--out", type=Path, default=Path("data/dataset"))
    ap.add_argument("--test", action="append", default=[], help="match id to hold out for testing (repeatable)")
    ap.add_argument("--val", action="append", default=[], help="match id for validation (repeatable)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    frames = collect(args.exports)
    matches = sorted({match_id(s) for s in frames})
    split_of = assign(matches, args.test, args.val, args.seed)
    summary = write_dataset(frames, split_of, args.out, read_names(args.exports))
    for s in SPLITS:
        info = summary[s]
        print(f"{s:5s}: {info['images']:4d} images, {info['boxes']:5d} boxes  <- {', '.join(info['matches'])}")
    print(f"dataset: {args.out / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
