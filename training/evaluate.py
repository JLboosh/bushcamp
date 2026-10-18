"""Evaluate the detector on held-out test matches and save failure examples.

    python -m training.evaluate runs/detect/brawlers/weights/best.pt data/dataset/dataset.yaml

Prints:
- Ultralytics precision / recall / mAP per class on the test split.
- Plain counts at one confidence threshold: found, missed, false boxes, and
  team mix-ups (box in the right place, wrong class).
Saves frames with mistakes, worst first, to runs/eval/<name>/failures/:
    green  = correct          red    = false box (nothing there)
    orange = wrong team       magenta = missed brawler (label with no detection)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

IOU_MATCH = 0.5
COLORS = {"ok": (0, 200, 0), "false": (0, 0, 255), "wrong_class": (0, 140, 255), "missed": (255, 0, 255)}


@dataclass
class ImageResult:
    path: Path
    ok: list = field(default_factory=list)            # (cls, box)
    false: list = field(default_factory=list)         # (pred cls, box)
    wrong_class: list = field(default_factory=list)   # (true cls, pred cls, box)
    missed: list = field(default_factory=list)        # (cls, box)

    @property
    def errors(self) -> int:
        return len(self.false) + len(self.wrong_class) + len(self.missed)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def read_labels(path: Path, w: int, h: int) -> list[tuple[int, np.ndarray]]:
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                c, cx, cy, bw, bh = map(float, line.split()[:5])
                out.append((int(c), np.array([(cx - bw / 2) * w, (cy - bh / 2) * h,
                                              (cx + bw / 2) * w, (cy + bh / 2) * h])))
    return out


def match_image(gt: list[tuple[int, np.ndarray]], preds: list[tuple[int, float, np.ndarray]],
                path: Path) -> ImageResult:
    """Greedy matching, most confident prediction first."""
    res = ImageResult(path)
    used = set()
    for pc, _, pbox in sorted(preds, key=lambda p: -p[1]):
        best, best_iou = None, IOU_MATCH
        for j, (_, gbox) in enumerate(gt):
            if j not in used and (v := iou(pbox, gbox)) >= best_iou:
                best, best_iou = j, v
        if best is None:
            res.false.append((pc, pbox))
            continue
        used.add(best)
        gc = gt[best][0]
        if gc == pc:
            res.ok.append((pc, pbox))
        else:
            res.wrong_class.append((gc, pc, pbox))
    res.missed = [g for j, g in enumerate(gt) if j not in used]
    return res


def draw(res: ImageResult, names: list[str]) -> np.ndarray:
    img = cv2.imread(str(res.path))

    def box(b, color, text, width=2):
        x0, y0, x1, y1 = map(int, b)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, width)
        cv2.putText(img, text, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    for c, b in res.ok:
        box(b, COLORS["ok"], names[c], 1)
    for c, b in res.false:
        box(b, COLORS["false"], f"false {names[c]}")
    for gc, pc, b in res.wrong_class:
        box(b, COLORS["wrong_class"], f"{names[gc]} seen as {names[pc]}")
    for c, b in res.missed:
        box(b, COLORS["missed"], f"missed {names[c]}", 3)
    cv2.putText(img, res.path.stem, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def evaluate(weights: Path, data: Path, split: str = "test", conf: float = 0.25, imgsz: int = 960,
             out: Path | None = None, max_failures: int = 40) -> dict:
    from ultralytics import YOLO
    cfg = yaml.safe_load(data.read_text())
    names = cfg["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    images_dir = Path(cfg["path"]) / cfg[split]
    out = out or Path("runs/eval") / weights.parent.parent.name
    model = YOLO(str(weights))

    # 1. Standard metrics.
    m = model.val(data=str(data), split=split, imgsz=imgsz, plots=False, verbose=False)
    standard = {}
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        standard[names[int(c)]] = dict(precision=round(float(p), 3), recall=round(float(r), 3),
                                       mAP50=round(float(ap50), 3), mAP50_95=round(float(ap), 3))

    # 2. Plain counts at a fixed threshold, plus failure images.
    results = []
    for r in model.predict(source=str(images_dir), conf=conf, imgsz=imgsz, stream=True, verbose=False):
        path = Path(r.path)
        h, w = r.orig_shape
        preds = [(int(c), float(s), b) for c, s, b in
                 zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy())]
        gt = read_labels(path.parent.parent / "labels" / f"{path.stem}.txt", w, h)
        results.append(match_image(gt, preds, path))

    counts = {}
    for i, n in enumerate(names):
        counts[n] = dict(
            labelled=sum(1 for r in results for c, _ in r.ok + r.missed if c == i)
            + sum(1 for r in results for gc, _, _ in r.wrong_class if gc == i),
            found=sum(1 for r in results for c, _ in r.ok if c == i),
            missed=sum(1 for r in results for c, _ in r.missed if c == i),
            false_boxes=sum(1 for r in results for c, _ in r.false if c == i),
            seen_as_other_team=sum(1 for r in results for gc, _, _ in r.wrong_class if gc == i),
        )

    fail_dir = out / "failures"
    fail_dir.mkdir(parents=True, exist_ok=True)
    for old in fail_dir.glob("*.jpg"):
        old.unlink()
    worst = sorted((r for r in results if r.errors), key=lambda r: -r.errors)[:max_failures]
    for k, r in enumerate(worst):
        cv2.imwrite(str(fail_dir / f"{k:02d}_{r.path.stem}.jpg"), draw(r, names))

    report = dict(split=split, images=len(results), conf=conf, iou=IOU_MATCH,
                  standard=standard, counts=counts,
                  images_with_errors=sum(1 for r in results if r.errors))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def print_report(report: dict, out: Path) -> None:
    print(f"\n{report['split']} split: {report['images']} images "
          f"({report['images_with_errors']} with at least one mistake)\n")
    print(f"{'class':8s} {'precision':>9s} {'recall':>7s} {'mAP50':>6s} {'mAP50-95':>8s}")
    for n, s in report["standard"].items():
        print(f"{n:8s} {s['precision']:9.3f} {s['recall']:7.3f} {s['mAP50']:6.3f} {s['mAP50_95']:8.3f}")
    print(f"\nat confidence {report['conf']} (box counts as found if it overlaps the label by IoU >= {report['iou']}):")
    print(f"{'class':8s} {'labelled':>8s} {'found':>6s} {'missed':>6s} {'false':>6s} {'wrong team':>10s}")
    for n, c in report["counts"].items():
        print(f"{n:8s} {c['labelled']:8d} {c['found']:6d} {c['missed']:6d} {c['false_boxes']:6d} "
              f"{c['seen_as_other_team']:10d}")
    print(f"\nfailure examples: {out / 'failures'}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate the detector on held-out matches.")
    ap.add_argument("weights", type=Path)
    ap.add_argument("data", type=Path, help="dataset.yaml from training.split")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    out = args.out or Path("runs/eval") / args.weights.parent.parent.name
    report = evaluate(args.weights, args.data, args.split, args.conf, args.imgsz, out)
    print_report(report, out)


if __name__ == "__main__":
    main()
