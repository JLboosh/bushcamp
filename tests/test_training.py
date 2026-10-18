import json

import numpy as np
import pytest

from training.evaluate import match_image
from training.split import assign, collect, match_id, original_stem, read_names, write_dataset


def fake_export(root, matches, frames_per_match=3, rf_split="train"):
    """Mimic a Roboflow YOLOv8 export: renamed files, images/ and labels/ folders, data.yaml."""
    (root / rf_split / "images").mkdir(parents=True)
    (root / rf_split / "labels").mkdir(parents=True)
    for m in matches:
        for k in range(frames_per_match):
            stem = f"{m}_t{k * 1000:07d}_jpg.rf.{k:032x}"
            (root / rf_split / "images" / f"{stem}.jpg").write_bytes(b"jpg")
            (root / rf_split / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.2\n1 0.2 0.2 0.1 0.2\n")
    (root / "data.yaml").write_text("nc: 2\nnames: ['ally', 'enemy']\n")


def test_names_from_roboflow_files():
    from pathlib import Path
    p = Path("Screenrecording_20261013_222013_com_t0028007_jpg.rf.0a1b2c3d4e5f.jpg")
    assert original_stem(p) == "Screenrecording_20261013_222013_com_t0028007"
    assert match_id(original_stem(p)) == "Screenrecording_20261013_222013_com"
    assert match_id("1000075100_t0019001") == "1000075100"
    with pytest.raises(ValueError):
        match_id("random_image")


def test_assign_keeps_explicit_choices_and_fills_each_split():
    matches = [f"m{i}" for i in range(10)]
    split = assign(matches, test=["m3"], val=[])
    assert split["m3"] == "test"
    assert sorted(split.values()).count("test") == 2      # 20% of 10
    assert sorted(split.values()).count("val") == 2       # 15% of 10, rounded
    assert assign(matches, test=["m3"], val=[]) == split   # deterministic
    with pytest.raises(ValueError):
        assign(["a", "b"], [], [])                         # too few matches
    with pytest.raises(ValueError):
        assign(matches, test=["nope"], val=[])


def test_split_never_mixes_a_match(tmp_path):
    # Roboflow scattered match frames across its own splits; ours must regroup them.
    fake_export(tmp_path / "exp", ["a", "b"], rf_split="train")
    fake_export(tmp_path / "exp2", ["c", "d"], rf_split="valid")
    exports = [tmp_path / "exp", tmp_path / "exp2"]
    frames = collect(exports)
    assert len(frames) == 12
    split_of = assign(sorted({match_id(s) for s in frames}), test=["a"], val=["b"])
    summary = write_dataset(frames, split_of, tmp_path / "ds", read_names(exports))

    seen = {}
    for s in ("train", "val", "test"):
        for img in (tmp_path / "ds" / s / "images").glob("*.jpg"):
            assert (tmp_path / "ds" / s / "labels" / f"{img.stem}.txt").exists()
            seen.setdefault(match_id(img.stem), set()).add(s)
    assert all(len(v) == 1 for v in seen.values())
    assert seen["a"] == {"test"} and seen["b"] == {"val"}
    assert summary["train"]["images"] == 6 and summary["train"]["boxes"] == 12
    yml = (tmp_path / "ds" / "dataset.yaml").read_text()
    assert "0: ally" in yml and "1: enemy" in yml and str(tmp_path) in yml
    assert json.loads((tmp_path / "ds" / "split.json").read_text())["test"]["matches"] == ["a"]


def test_split_refuses_to_overwrite_unrelated_folder(tmp_path):
    fake_export(tmp_path / "exp", ["a", "b", "c"])
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "important.txt").write_text("keep me")
    frames = collect([tmp_path / "exp"])
    with pytest.raises(FileExistsError):
        write_dataset(frames, assign(["a", "b", "c"], [], []), tmp_path / "ds", ["ally", "enemy"])


def test_match_image_counts_each_kind_of_mistake():
    box = lambda x: np.array([x, 0, x + 10, 10], float)
    gt = [(0, box(0)), (1, box(100)), (1, box(200))]
    preds = [(0, 0.9, box(1)),      # correct ally
             (0, 0.8, box(101)),    # enemy seen as ally
             (1, 0.7, box(500))]    # nothing there
    res = match_image(gt, preds, path=None)
    assert len(res.ok) == 1
    assert [(g, p) for g, p, _ in res.wrong_class] == [(1, 0)]
    assert len(res.false) == 1
    assert [c for c, _ in res.missed] == [1]               # the enemy at x=200
    assert res.errors == 3
