import threading
import time
from typing import Optional
import numpy as np
from collections import deque

class VideoQueue:
    def __init__(self, timeout_seconds: float = 10 * 60) -> None:
        self.frames: deque[np.ndarray] = deque()
        self.lock = threading.Lock()
        self.closed = False
        self.timeout_seconds = timeout_seconds
        self.last_write = time.perf_counter()
        self.n_stored_frames = 0

    def _mark_closed_if_idle(self) -> None:
        if time.perf_counter() - self.last_write > self.timeout_seconds:
            self.closed = True

    def _validate_frame(self, frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy ndarray")
        if frame.dtype != np.uint8:
            raise TypeError("frame dtype must be np.uint8")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have shape (height, width, 3) for BGR24 format")

    def enqueue(self, frame: np.ndarray) -> None:
        self._validate_frame(frame)
        with self.lock:
            self.frames.append(frame)
            self.last_write = time.perf_counter()
            self.n_stored_frames += 1

    def dequeue(self) -> Optional[np.ndarray]:
        with self.lock:
            self._mark_closed_if_idle()
            if not self.frames:
                return None
            self.n_stored_frames -= 1
            return self.frames.popleft()

    def peek(self) -> Optional[np.ndarray]:
        with self.lock:
            self._mark_closed_if_idle()
            if not self.frames:
                return None
            return self.frames[0]

    def __len__(self) -> int:
        with self.lock:
            return self.n_stored_frames

    def close(self) -> None:
        with self.lock:
            self.closed = True

    def is_closed(self) -> bool:
        with self.lock:
            self._mark_closed_if_idle()
            return self.closed