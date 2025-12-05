# src/counting/counting_modes/zone.py

from collections import defaultdict
from typing import Tuple
import math
from src.utils.logger import log

Point = Tuple[float, float]
EDGE_NAMES_RECT = ["top", "right", "bottom", "left"]  # clockwise


# ------------------ Geometry utils ------------------
def _nearest_zone_edge(
    cx: float, cy: float, x1: float, y1: float, x2: float, y2: float
):
    """
    Given a centroid (cx,cy) and a rectangle (x1,y1,x2,y2),
    compute precise distances to each edge and return the nearest one.
    """
    d_top = abs(cy - y1)
    d_bottom = abs(y2 - cy)
    d_left = abs(cx - x1)
    d_right = abs(x2 - cx)

    distances = {"top": d_top, "bottom": d_bottom, "left": d_left, "right": d_right}

    # find the edge with minimum distance
    nearest_edge = min(distances, key=distances.get)
    nearest_dist = distances[nearest_edge]

    return nearest_edge, nearest_dist, distances


def _segment_intersection(p1: Point, p2: Point, q1: Point, q2: Point):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return False, None, None, None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / den
    if 0 <= t <= 1 and 0 <= u <= 1:
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        return True, t, u, (px, py)
    return False, None, None, None


def _nearest_point_on_segment(px, py, ax, ay, bx, by):
    abx, aby = (bx - ax), (by - ay)
    ab2 = abx * abx + aby * aby
    if ab2 <= 1e-9:
        return (ax, ay), 0.0
    t = ((px - ax) * abx + (py - ay) * aby) / ab2
    t = max(0.0, min(1.0, t))
    return (ax + t * abx, ay + t * aby), t


def _rect_edges_from_xyxy(x1, y1, x2, y2):
    return [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]


def _edge_idx_to_name(idx: int) -> str:
    return EDGE_NAMES_RECT[idx % 4]


def _point_in_rect(cx, cy, x1, y1, x2, y2):
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)


# ------------------ Direction classifier ------------------
def _edge_pair_to_direction(
    entry_idx: int, exit_idx: int, entry_pt=None, exit_pt=None, zone_h=None
) -> str:
    """
    Enhanced mapping of entry/exit edges to 'up'/'down'/'skip'.
    Rules (user requirement):
      - bottom→top = up
      - top→bottom = down
      - left→bottom = down
      - right→bottom = down
      - all others = skip
    """
    e, x = entry_idx % 4, exit_idx % 4

    # Entry = bottom -> always up
    if e == 2:
        return "up"

    # Same edge → fallback to motion
    if e == x and entry_pt and exit_pt and zone_h:
        return "skip"

    # Bottom entry
    if e == 2 and x == 0:
        return "up"  # bottom→top
    if e == 2:
        return "up"  # bottom→anywhere else → up (default bias)

    # Top entry
    if e == 0 and x == 2:
        return "down"  # top→bottom

    # Left entry
    if e == 3 and x == 2:
        return "down"  # left→bottom
    if e == 3 and x == 0:
        return "up"  # left→top
    if e == 3:
        return "skip"

    # Right entry
    if e == 1 and x == 2:
        return "down"  # right→bottom
    if e == 1 and x == 0:
        return "up"  # right→top
    if e == 1:
        return "skip"


# ------------------ Zone Counter ------------------
class ZoneCounter:
    def __init__(
        self,
        rect_roi_xyxy,
        born_inside_policy="count_entry",
        backfill_wait_frames=4,
        near_border_px=24,
        quiet=False,
    ):
        x1, y1, x2, y2 = rect_roi_xyxy
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        self.x1, self.y1, self.x2, self.y2 = float(x1), float(y1), float(x2), float(y2)
        self.edges = _rect_edges_from_xyxy(self.x1, self.y1, self.x2, self.y2)
        self.born_inside_policy = born_inside_policy
        self.backfill_wait = max(0, int(backfill_wait_frames))
        self.near_border_px = float(max(0, near_border_px))
        self.quiet = quiet

        self.prev_inside = defaultdict(lambda: False)
        self.prev_pt = {}
        self.entry_edge_idx = defaultdict(lambda: None)
        self.entry_point = defaultdict(lambda: None)
        self.entry_method = defaultdict(lambda: None)
        self.pending = {}
        self.ghost_ids = set()

    def _inside(self, cx, cy):
        return _point_in_rect(cx, cy, self.x1, self.y1, self.x2, self.y2)

    def update_for_frame(self, tid, cx, cy, frame_idx):
        p = (float(cx), float(cy))
        inside_now = self._inside(cx, cy)
        prev_in = self.prev_inside[tid]
        self.prev_inside[tid] = inside_now

        # Born inside (ghost goat case)
        if tid not in self.prev_pt:
            self.prev_pt[tid] = p
            if inside_now and self.born_inside_policy == "count_entry":
                nearest_edge, dist, all_dists = _nearest_zone_edge(
                    cx, cy, self.x1, self.y1, self.x2, self.y2
                )
                edge_idx = EDGE_NAMES_RECT.index(nearest_edge)
                self.entry_edge_idx[tid] = edge_idx
                self.entry_point[tid] = (cx, cy)
                self.entry_method[tid] = "ghost_born"
                self.ghost_ids.add(tid)

                if not self.quiet:
                    log.error(
                        "ZONE-GHOST",
                        f" tid={tid} ({cx:.1f},{cy:.1f}) dists={all_dists} -> nearest={nearest_edge} ({dist:.1f}px)",
                    )

                return dict(
                    type="entry",
                    edge_idx=edge_idx,
                    edge_name=nearest_edge,
                    method="ghost_born",
                    pt=(cx, cy),
                    backfill=True,
                    tid=tid,
                    first_pt=p,
                )
            return None

        # First ever point
        if tid not in self.prev_pt:
            self.prev_pt[tid] = p
            return None
        prev_p = self.prev_pt[tid]
        self.prev_pt[tid] = p

        # Entry
        if not prev_in and inside_now:
            e_idx, hit = self._classify(prev_p, p)
            self.entry_edge_idx[tid] = e_idx
            self.entry_point[tid] = hit
            self.entry_method[tid] = "perimeter"
            return dict(
                type="entry",
                edge_idx=e_idx,
                edge_name=_edge_idx_to_name(e_idx),
                method="perimeter",
                pt=hit,
                backfill=False,
                tid=tid,
            )

        # Exit
        if prev_in and not inside_now:
            x_idx, x_pt = self._classify(prev_p, p)
            e_idx = self.entry_edge_idx.get(tid, None)
            if e_idx is not None:
                direction = _edge_pair_to_direction(
                    e_idx,
                    x_idx,
                    entry_pt=self.entry_point.get(tid),
                    exit_pt=x_pt,
                    zone_h=(self.y2 - self.y1),
                )
                method = self.entry_method.get(tid, "?")
                self.entry_edge_idx[tid] = None
                self.entry_point[tid] = None
                return dict(
                    type="count",
                    direction=direction,
                    entry_edge=_edge_idx_to_name(e_idx),
                    exit_edge=_edge_idx_to_name(x_idx),
                    pt=x_pt,
                    tid=tid,
                    method=method,
                )
            return dict(
                type="exit",
                edge_idx=x_idx,
                edge_name=_edge_idx_to_name(x_idx),
                pt=x_pt,
                tid=tid,
            )
        return None

    def _classify(self, prev_p, p):
        """Pick the most likely edge crossing."""
        best_idx, best_d, best_q = None, 1e18, None
        for idx, (A, B) in enumerate(self.edges):
            inter, t, u, P = _segment_intersection(prev_p, p, A, B)
            if inter:
                return idx, P
            (qx, qy), _ = _nearest_point_on_segment(
                prev_p[0], prev_p[1], A[0], A[1], B[0], B[1]
            )
            d = math.hypot(prev_p[0] - qx, prev_p[1] - qy)
            if d < best_d:
                best_d, best_idx, best_q = d, idx, (qx, qy)
        return best_idx, best_q

    def rect_xyxy(self):
        return (self.x1, self.y1, self.x2, self.y2)
