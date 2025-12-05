# src/viz/animator.py
import cv2
from ..geometry.geom import project_point_to_segment


class CrossAnimator:
    def __init__(
        self,
        duration_frames=20,
        color_up=(40, 255, 40),
        color_down=(255, 60, 60),
        label_mode="words",
    ):
        self.duration = max(1, int(duration_frames))
        self.active = []
        self.color_up = tuple(color_up)
        self.color_down = tuple(color_down)
        self.label_mode = label_mode

    def trigger(self, cx, cy, line_roi, line_full, direction, frame_idx):
        ax, ay, bx, by = line_roi
        qx, qy = project_point_to_segment(cx, cy, ax, ay, bx, by)
        axF, ayF, bxF, byF = line_full
        qxF, qyF = project_point_to_segment(qx, qy, axF, ayF, bxF, byF)
        self.active.append(
            {
                "roi": (qx, qy),
                "full": (qxF, qyF),
                "start": frame_idx,
                "end": frame_idx + self.duration,
                "dir": "up" if direction == "up" else "down",
            }
        )

    def _label_for(self, direction):
        if self.label_mode == "ascii":
            return f"+1 {'^' if direction == 'up' else 'v'}"
        return f"+1 {'UP' if direction == 'up' else 'DOWN'}"

    def _color_for(self, direction):
        return self.color_up if direction == "up" else self.color_down

    def _draw_one(self, img, pt, t, direction):
        if img is None:
            return
        x, y = int(pt[0]), int(pt[1])
        color = self._color_for(direction)
        overlay = img.copy()
        r = int(8 + 22 * t)
        cv2.circle(overlay, (x, y), r, color, thickness=3, lineType=cv2.LINE_AA)
        alpha = 0.75 * (1.0 - t)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        txt = self._label_for(direction)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        y_txt = y - 10 - int(12 * t)
        cv2.putText(
            img,
            txt,
            (x - tw // 2 + 1, y_txt + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            txt,
            (x - tw // 2, y_txt),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    def draw(self, roi_img, full_img, rx, ry, frame_idx):
        if not self.active:
            return
        keep = []
        for item in self.active:
            if frame_idx >= item["end"]:
                continue
            t = (frame_idx - item["start"]) / float(self.duration)
            self._draw_one(roi_img, item["roi"], t, item["dir"])
            self._draw_one(full_img, item["full"], t, item["dir"])
            keep.append(item)
        self.active = keep
