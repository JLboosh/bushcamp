"""Bundle frames and pre-labels into one folder to drag into Roboflow.

Output is a flat folder of <frame>.jpg + <frame>.txt (YOLO boxes) and a
data.yaml with the class names, which Roboflow reads as YOLO annotations.
File names keep the match id prefix, so the by-match split still works
after labels come back from Roboflow.

    python -m training.export_roboflow --out data/roboflow_upload/batch1
    python -m training.export_roboflow --match Screenrecording_20261013_222328_com --out ...
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pipeline.color_bootstrap import CLASSES


def export(frames: Path, labels: Path, out: Path, matches: list[str] | None = None) -> int:
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for match_dir in sorted(p for p in frames.iterdir() if p.is_dir()):
        if matches and match_dir.name not in matches:
            continue
        for img in sorted(match_dir.glob("*_t*.jpg")):
            label = labels / match_dir.name / f"{img.stem}.txt"
            shutil.copy2(img, out / img.name)
            if label.exists():
                shutil.copy2(label, out / label.name)
            n += 1
    names = "\n".join(f"  - {c}" for c in CLASSES)
    (out / "data.yaml").write_text(f"nc: {len(CLASSES)}\nnames:\n{names}\n")
    return n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", type=Path, default=Path("data/frames"))
    ap.add_argument("--labels", type=Path, default=Path("data/prelabels"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--match", action="append", help="only these match ids (repeatable)")
    args = ap.parse_args(argv)
    n = export(args.frames, args.labels, args.out, args.match)
    print(f"{n} frames -> {args.out}")


if __name__ == "__main__":
    main()
