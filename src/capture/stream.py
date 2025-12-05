# src/capture/stream.py


# ---------- ROI frame streamer ----------
class ROIStream:
    def __init__(self, cap, rx, ry, rw, rh, stride=1):
        self.cap = cap
        self.rx, self.ry, self.rw, self.rh = rx, ry, rw, rh
        self.stride = max(1, int(stride))
        self.extra_skip = 0
        self.full = None
        self.roi = None
        self.frame_in = 0
        self.out_index = 0

    def set_stride(self, s):
        self.stride = max(1, int(s))

    def request_skip(self, n):
        self.extra_skip += max(0, int(n))

    def __iter__(self):
        while True:
            while self.extra_skip > 0:
                ok = self.cap.grab()
                if not ok:
                    return
                self.frame_in += 1
                self.extra_skip -= 1
            for _ in range(self.stride - 1):
                ok = self.cap.grab()
                if not ok:
                    return
                self.frame_in += 1
            ok, frame = self.cap.read()
            if not ok:
                return
            self.frame_in += 1
            self.full = frame
            self.roi = frame[self.ry : self.ry + self.rh, self.rx : self.rx + self.rw]
            self.out_index += 1
            yield self.roi
