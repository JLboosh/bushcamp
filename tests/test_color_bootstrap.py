import cv2
import numpy as np

from pipeline.color_bootstrap import CLASSES, detect, find_bars, prelabel_dir

FLOOR = (125, 140, 125)          # HSV of the purple floor
FILL = {"ally": (90, 230, 250), "enemy": (173, 240, 210)}
DRAINED_TOP = (120, 70, 170)     # the drained part is light only in its top half
DRAINED_BOTTOM = (122, 170, 90)


def _hsv_color(hsv):
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0, 0])


def blank(h=720, w=1562):
    img = np.empty((h, w, 3), np.uint8)
    img[:] = _hsv_color(FLOOR)
    return img


def draw_bar(img, x, y, cls="ally", hp=1.0, name="Player", s=1.0):
    """Draw a health bar laid out like the replay UI: name, HP digits, bar."""
    bw, bh = int(58 * s), int(10 * s)
    fw = max(int(4 * s), int(bw * hp))
    cv2.rectangle(img, (x, y), (x + bw - 1, y + bh // 2 - 1), _hsv_color(DRAINED_TOP), -1)
    cv2.rectangle(img, (x, y + bh // 2), (x + bw - 1, y + bh - 1), _hsv_color(DRAINED_BOTTOM), -1)
    cv2.rectangle(img, (x, y), (x + fw - 1, y + bh - 1), _hsv_color(FILL[cls]), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "6000", (x + int(12 * s), y + int(4 * s)), font, 0.45 * s, (255, 255, 255),
                max(1, int(2 * s)))
    if name:
        cv2.putText(img, name, (x, y - int(13 * s)), font, 0.45 * s, (255, 255, 230), max(1, int(2 * s)))


def test_full_and_low_hp_bars():
    img = blank()
    draw_bar(img, 300, 300, "ally", hp=1.0)
    draw_bar(img, 900, 200, "enemy", hp=0.1)
    bars = sorted(find_bars(img), key=lambda b: b.x)
    assert [(b.cls, b.x, b.y) for b in bars] == [("ally", 300, 300), ("enemy", 900, 200)]
    assert bars[0].fill > 0.95
    assert 0.05 < bars[1].fill < 0.2


def test_box_covers_name_and_body():
    img = blank()
    draw_bar(img, 300, 300)
    (box,) = detect(img)
    assert box.x0 < 300 and box.x1 > 300 + 58
    assert box.y0 <= 300 - 25          # name tag above
    assert box.y1 >= 300 + 90          # brawler below


def test_spawnable_without_name_is_skipped():
    img = blank()
    draw_bar(img, 300, 300, "enemy", name=None)
    assert find_bars(img) == []


def test_red_effect_is_skipped():
    img = blank()
    cv2.circle(img, (500, 300), 60, _hsv_color(FILL["enemy"]), -1)
    cv2.putText(img, "Player", (450, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 230), 2)
    assert find_bars(img) == []


def test_round_indicator_zone_is_skipped():
    img = blank()
    draw_bar(img, 776, 37, "ally")
    assert find_bars(img) == []


def test_scales_with_resolution():
    img = blank(h=1440, w=3124)
    draw_bar(img, 600, 600, "enemy", hp=0.5, s=2.0)
    (bar,) = find_bars(img)
    assert (bar.cls, bar.x, bar.y) == ("enemy", 600, 600)
    assert 0.4 < bar.fill < 0.6


def test_prelabel_dir_writes_yolo(tmp_path):
    frames = tmp_path / "frames" / "match1"
    frames.mkdir(parents=True)
    img = blank()
    draw_bar(img, 300, 300, "ally")
    draw_bar(img, 900, 200, "enemy")
    cv2.imwrite(str(frames / "match1_t0001000.jpg"), img)
    cv2.imwrite(str(frames / "match1_contact_sheet.jpg"), img)   # must be ignored

    stats = prelabel_dir(tmp_path / "frames", tmp_path / "labels", viz=1, viz_dir=tmp_path / "viz")
    assert stats == {"images": 1, "ally": 1, "enemy": 1, "empty": 0}
    lines = (tmp_path / "labels" / "match1" / "match1_t0001000.txt").read_text().splitlines()
    rows = sorted([float(v) for v in line.split()] for line in lines)
    assert [int(r[0]) for r in rows] == [0, 1]
    assert all(0 < v < 1 for r in rows for v in r[1:])
    assert (tmp_path / "labels" / "match1" / "classes.txt").read_text().split() == CLASSES
    assert (tmp_path / "viz" / "match1_t0001000.jpg").exists()
