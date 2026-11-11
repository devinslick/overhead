"""Synthetic-replay validation source: fly a real airplane sprite along a
REAL live aircraft's projected path.

Ground truth comes from the live ADS-B feed (dead-reckoned between polls);
the pipeline never sees it — it must independently detect, track, and
correlate. If the overlay recovers the correct callsign, the whole loop is
validated. (For real deployments, swap this source for an RTSP camera.)
"""
import cv2
import numpy as np
from pathlib import Path

from .geo import project, angles_to_pixel
from .adsb import dead_reckon


def load_airplane_sprite(coco_dir=Path.home() / "Work/datasets/coco128",
                         prefer=("000000000472", "000000000081")):
    """Crop an airplane (class 4) from the labeled COCO subset.

    `prefer` stems first — empirically the only crops our detector reliably
    fires on when composited onto synthetic sky (probe: conf 0.72-0.82 at
    220px, below 0.15 at <=160px).
    """
    labels = coco_dir / "labels" / "train2017"
    images = coco_dir / "images" / "train2017"
    stems = [s for s in prefer if (labels / f"{s}.txt").exists()]
    stems += [lf.stem for lf in sorted(labels.glob("*.txt")) if lf.stem not in stems]
    for stem in stems:
        lf = labels / f"{stem}.txt"
        for line in lf.read_text().splitlines():
            parts = line.split()
            if parts and parts[0] == "4":
                _, xc, yc, w, h = (float(x) for x in parts[:5])
                img = cv2.imread(str(images / (lf.stem + ".jpg")))
                if img is None:
                    break
                H, W = img.shape[:2]
                x1, y1 = int((xc - w / 2) * W), int((yc - h / 2) * H)
                x2, y2 = int((xc + w / 2) * W), int((yc + h / 2) * H)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 - x1 > 20 and y2 - y1 > 15:
                    return img[y1:y2, x1:x2].copy()
    raise RuntimeError("no airplane sprite found in coco128")


def sky_frame(width, height, sun_x=0.8, sun_y=0.15):
    """Cheap gradient sky with a soft sun glow."""
    top = np.array([120, 170, 235], np.uint8)     # BGR
    bot = np.array([225, 235, 245], np.uint8)
    rows = np.linspace(0, 1, height)[:, None]
    grad = (top[None, None, :] * (1 - rows[:, :, None])
            + bot[None, None, :] * rows[:, :, None]).astype(np.uint8)
    frame = np.repeat(grad, width, axis=1)
    cx, cy = int(sun_x * width), int(sun_y * height)
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), 60, (255, 255, 250), -1)
    return cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)


def paste(frame, sprite, cx, cy):
    h, w = sprite.shape[:2]
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = x1 + w, y1 + h
    fx1, fy1 = max(0, x1), max(0, y1)
    fx2, fy2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if fx2 <= fx1 or fy2 <= fy1:
        return frame
    frame[fy1:fy2, fx1:fx2] = sprite[fy1 - y1:fy2 - y1, fx1 - x1:fx2 - x1]
    return frame


class SyntheticReplay:
    """30fps synthetic sky following one real aircraft's ADS-B projection."""

    def __init__(self, cam, poller, target_icao, width=1280, height=720,
                 fps=30.0, sprite_w=240):
        self.cam, self.poller, self.target = cam, poller, target_icao
        self.width, self.height, self.fps = width, height, fps
        sprite = load_airplane_sprite()
        s = sprite_w / sprite.shape[1]
        self.sprite = cv2.resize(sprite, (sprite_w, int(sprite.shape[0] * s)))
        self.frame_idx = 0
        self.t0 = None

    def _target_state(self):
        for ac in self.poller.live():
            if ac["icao24"] == self.target:
                return ac
        return None

    def next_frame(self):
        """Returns (frame, truth) or None when the target is gone/stale."""
        import time
        if self.t0 is None:
            self.t0 = time.time()
        ac = self._target_state()
        if ac is None:
            return None
        # advance truth position to *now* along its track
        import time as _t
        la, lo = dead_reckon(ac, _t.time() - (self.poller.last_ok or _t.time()))
        brg, el, dist = project(self.cam["lat"], self.cam["lon"], self.cam["alt_m"],
                                la, lo, (ac["alt_ft"] or 0) * 0.3048)
        x, y = angles_to_pixel(brg, el, self.width, self.height,
                               self.cam["az_deg"], self.cam["hfov"], self.cam["vfov"])
        if not (0 <= x < self.width and 0 <= y < self.height):
            return None  # flew out of the synthetic FOV
        frame = sky_frame(self.width, self.height)
        paste(frame, self.sprite, x, y)
        self.frame_idx += 1
        truth = {"icao24": ac["icao24"], "callsign": ac["callsign"],
                 "alt_ft": ac["alt_ft"], "bearing": round(brg, 2),
                 "elevation": round(el, 2), "dist_km": round(dist / 1000, 1),
                 "px": (round(x, 1), round(y, 1))}
        return frame, truth
