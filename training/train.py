"""Train the brawler detector (ally/enemy) with Ultralytics YOLO.

    python -m training.train data/dataset/dataset.yaml --epochs 100
    python -m training.train data/dataset/dataset.yaml --model yolo26s.pt   # bigger, slower, usually better

Runs on the Mac's GPU (Apple MPS), an NVIDIA GPU (e.g. Colab) or CPU.
Results land in runs/detect/<name>/ (weights/best.pt, training curves).
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Augmentation. Team is told apart by colour (cyan vs red bars and name tags),
# so anything that changes hue would swap teams and corrupt the labels.
AUGMENT = dict(
    hsv_h=0.0,     # no hue shifts: they turn cyan bars into other colours
    hsv_s=0.3,     # mild saturation change keeps hue (Ultralytics default 0.7)
    hsv_v=0.4,     # brightness: fine
    fliplr=0.0,    # no mirroring: name tags and digits never appear mirrored, and bars drain right-to-left
    flipud=0.0,
    degrees=0.0,   # the camera never rotates
    translate=0.1,
    scale=0.4,
    mosaic=1.0,    # stitches 4 frames together: helps a small dataset, colours untouched
    mixup=0.0,     # blends two frames: would mix team colours
)


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(data: Path, model: str = "yolo26n.pt", epochs: int = 100, imgsz: int = 960,
          batch: int = 8, name: str = "brawlers", device: str | None = None, **overrides):
    from ultralytics import YOLO
    yolo = YOLO(model)
    return yolo.train(
        data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, name=name,
        device=device or pick_device(), project=str(Path("runs/detect").resolve()),
        seed=0, patience=30, plots=True, **(AUGMENT | overrides),
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train the ally/enemy brawler detector.")
    ap.add_argument("data", type=Path, help="dataset.yaml from training.split")
    ap.add_argument("--model", default="yolo26n.pt", help="starting weights (yolo26n/s/m.pt)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=960,
                    help="training image size; brawlers are ~90 px tall at 720p, so don't go below 640")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="brawlers")
    ap.add_argument("--device", help="mps, 0 (CUDA) or cpu; default: best available")
    args = ap.parse_args(argv)
    results = train(args.data, args.model, args.epochs, args.imgsz, args.batch, args.name, args.device)
    print(f"weights: {Path(results.save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
