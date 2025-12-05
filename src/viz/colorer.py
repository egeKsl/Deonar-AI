# src/viz/colorer.py
from typing import Dict


class CountColorer:
    def __init__(
        self,
        mode="persist",
        duration_frames=120,
        counted_color=(255, 120, 0),
        base_color=(0, 255, 0),
    ):
        self.mode = mode
        self.duration = max(1, int(duration_frames))
        self.counted_color = tuple(counted_color)
        self.base_color = tuple(base_color)
        self.marked_at: Dict[int, int] = {}

    def mark_counted(self, tid, frame_idx):
        if tid is None or tid < 0 or self.mode == "off":
            return
        self.marked_at.setdefault(int(tid), int(frame_idx))

    def color_for(self, tid, frame_idx):
        if self.mode == "off" or tid is None or tid < 0:
            return self.base_color
        first = self.marked_at.get(int(tid), None)
        if first is None:
            return self.base_color
        if self.mode == "persist":
            return self.counted_color
        # window mode
        if (frame_idx - int(first)) < self.duration:
            return self.counted_color
        return self.base_color
