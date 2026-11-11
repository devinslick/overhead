"""Minimal multi-object tracker: IoU-gated matching + linear-motion prediction.

Deliberately simple (no scipy dependency): greedy IoU matching, constant-
velocity prediction for missed frames, hit/age gating for track birth and
death. Good enough for slow-moving sky targets where the hard problem is
identity across occasional detection dropouts.
"""
import itertools


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


class Track:
    _next = 1

    def __init__(self, box):
        self.id = Track._next
        Track._next += 1
        self.box = list(box)
        self.vel = [0.0, 0.0]          # px/frame center velocity
        self.hits = 1
        self.misses = 0
        self.age = 0
        self.confirmed = False
        self.callsign = None           # attached by the correlator
        self.match_votes = {}          # icao24 -> consecutive-match count

    @property
    def center(self):
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)

    def predicted_box(self):
        cx, cy = self.center
        w = self.box[2] - self.box[0]
        h = self.box[3] - self.box[1]
        ncx, ncy = cx + self.vel[0], cy + self.vel[1]
        return [ncx - w / 2, ncy - h / 2, ncx + w / 2, ncy + h / 2]

    def update(self, box):
        ncx = (box[0] + box[2]) / 2.0
        ncy = (box[1] + box[3]) / 2.0
        ocx, ocy = self.center          # last MEASURED center
        self.vel = [0.6 * self.vel[0] + 0.4 * (ncx - ocx),
                    0.6 * self.vel[1] + 0.4 * (ncy - ocy)]
        self.box = list(box)
        self.hits += 1
        self.misses = 0
        if self.hits >= 3:
            self.confirmed = True

    def coast(self):
        self.box = self.predicted_box()  # advance last known state by velocity
        self.misses += 1


class IoUTracker:
    def __init__(self, iou_gate=0.05, max_misses=12):
        self.iou_gate = iou_gate
        self.max_misses = max_misses
        self.tracks = []

    def step(self, boxes):
        """Advance one frame with new detection boxes [(x1,y1,x2,y2), ...]."""
        existing = self.tracks
        pairs = []
        for ti, t in enumerate(existing):
            pb = t.predicted_box()
            for bi, b in enumerate(boxes):
                score = _iou(pb, b)
                if score > self.iou_gate:
                    pairs.append((score, ti, bi))
        pairs.sort(reverse=True)
        used_t, used_b = set(), set()
        for score, ti, bi in pairs:
            if ti in used_t or bi in used_b:
                continue
            existing[ti].update(boxes[bi])
            used_t.add(ti)
            used_b.add(bi)
        births = [Track(b) for bi, b in enumerate(boxes) if bi not in used_b]
        for ti, t in enumerate(existing):
            if ti not in used_t:
                t.coast()
        alive = [t for t in existing if t.misses <= self.max_misses]
        self.tracks = alive + births
        return [t for t in self.tracks if t.confirmed]
