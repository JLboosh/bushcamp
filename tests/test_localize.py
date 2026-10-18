import cv2
import numpy as np
import pytest

from pipeline.localize import Camera, Fix, clean_track, feature_masks, layout_images, match
from pipeline.maps import load

CAM = Camera(1562, 720, kx=0.0198, kv=0.0250, p=0.00028)
BUSH_BGR = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[93, 212, 170]]]), cv2.COLOR_HSV2BGR)[0, 0])
WATER_BGR = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[88, 160, 200]]]), cv2.COLOR_HSV2BGR)[0, 0])
FLOOR_BGR = (120, 90, 150)


@pytest.fixture(scope="module")
def streets():
    return load("streets_with_no_name")


def render(ref, cam, x, y, rotated=False):
    """Fake replay frame: floor, water, and bushes drawn `bush_raise` tiles up the screen."""
    h, w = cam.height, cam.width
    vv, uu = np.mgrid[:h, :w]
    pts = np.stack([uu.ravel(), vv.ravel()], 1).astype(np.float64)
    ground = cam.to_map(pts, x, y)
    raised = cam.to_map(pts, x, y)
    raised[:, 1] += ref.bush_raise          # a pixel shows the bush standing further down the screen
    img = np.empty((h * w, 3), np.uint8)
    img[:] = FLOOR_BGR

    def codes(xy):
        gx, gy = xy[:, 0], xy[:, 1]
        if rotated:
            gx, gy = ref.cols - gx, ref.rows - gy
        c, r = np.floor(gx).astype(int), np.floor(gy).astype(int)
        inside = (c >= 0) & (c < ref.cols) & (r >= 0) & (r < ref.rows)
        grid = np.array([list(row) for row in ref.layout])
        out = np.full(len(xy), ".")
        out[inside] = grid[r[inside], c[inside]]
        return out

    img[codes(ground) == "~"] = WATER_BGR
    img[codes(raised) == "B"] = BUSH_BGR
    return img.reshape(h, w, 3)


def test_camera_round_trip():
    pts = np.array([[100.0, 50.0], [781, 360], [1500, 700]])
    back = CAM.to_screen(CAM.to_map(pts, 10.4, 18.0), 10.4, 18.0)
    assert np.allclose(back, pts, atol=1e-6)
    centre = CAM.to_map(np.array([[781, 360]]), 10.4, 18.0)[0]
    assert np.allclose(centre, [10.4, 18.0])


@pytest.mark.parametrize("x,y", [(10.4, 18.4), (10.5, 9.0), (10.2, 24.6)])
def test_match_recovers_camera_position(streets, x, y):
    frame = render(streets, CAM, x, y)
    fix = match(feature_masks(frame), CAM, layout_images(streets))
    assert fix is not None and fix.score > 0.7
    assert abs(fix.x - x) < 0.15 and abs(fix.y - y) < 0.15


def test_rotated_map(streets):
    # Knockout maps are (nearly) symmetric under a 180° turn, so both
    # orientations fit almost equally well; the right one still wins, and
    # team sides are settled from spawns later, not from this.
    frame = render(streets, CAM, 10.5, 12.0, rotated=True)
    masks = feature_masks(frame)
    straight = match(masks, CAM, layout_images(streets, rotated=False))
    turned = match(masks, CAM, layout_images(streets, rotated=True))
    assert turned.score > straight.score
    assert abs(turned.x - 10.5) < 0.15 and abs(turned.y - 12.0) < 0.15


def test_layouts_have_three_spawns_per_team():
    for map_id in ("opening_move", "streets_with_no_name"):
        ref = load(map_id)
        text = "".join(ref.layout)
        assert text.count("b") == 3 and text.count("r") == 3
        assert ref.mask("B", 8).shape == (ref.rows * 8, ref.cols * 8)


def test_clean_track_keeps_pans_and_drops_single_misfits():
    fixes = []
    for i in range(40):
        t = i / 10
        y = 10.0 if t < 1.5 else 10.0 + (t - 1.5) * 20 if t < 2.5 else 30.0   # fast 20 tiles/s pan
        if i == 8:
            y = 25.0                                                          # one bad fit
        fixes.append((t, Fix(10.4, y, 0.6, False)))
    fixes.append((4.0, None))
    poses = clean_track(fixes)
    status = [p.status for p in poses]
    assert status[8] == "outlier"
    assert status.count("outlier") == 1
    assert all(p.status == "ok" for p in poses[15:25])                         # the pan survives
    assert poses[-1].status == "no_features"
    assert poses[30].y == pytest.approx(30.0)
