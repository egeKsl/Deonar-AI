# src/counting/state.py

from collections import defaultdict, deque
from types import SimpleNamespace


def _init_count_state(min_side_frames):
    return SimpleNamespace(
        last_side={},  # tid -> last confirmed side (-1/0/+1)
        side_hist=defaultdict(lambda: deque(maxlen=max(3, min_side_frames))),
        age_frames=defaultdict(int),  # tid -> age in frames
        last_counted=defaultdict(lambda: -(10**9)),  # tid -> last counted frame index
        up_count=0,
        down_count=0,
    )
