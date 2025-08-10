import threading
import time
from typing import Optional
import numpy as np

class VideoQueue:
    def __init__(self, timeout_seconds: float = 10 * 60) -> None:
        # List buffer storing frames (np.ndarray, dtype=uint8, shape=(H, W, 3))
        self.frames: list[np.ndarray] = []
        self.lock = threading.Lock()
        self.closed = False
        self.timeout_seconds = timeout_seconds
        self.last_write = time.perf_counter()

    def _mark_closed_if_idle(self) -> None:
        if time.perf_counter() - self.last_write > self.timeout_seconds:
            self.closed = True

    def _validate_frame(self, frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy ndarray")
        if frame.dtype != np.uint8:
            raise TypeError("frame dtype must be np.uint8")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "frame must have shape (height, width, 3) for BGR24 format"
            )

    # Enqueue a new frame (np.ndarray in BGR24)
    def enqueue(self, frame: np.ndarray) -> None:
        """
        Add a frame to the queue.

        Args:
            frame: np.ndarray, dtype=uint8, shape=(H, W, 3)
            copy: if True, store a copy to decouple from caller mutations
        """
        self._validate_frame(frame)
        with self.lock:
            self.frames.append(frame)
            self.last_write = time.perf_counter()

    # Dequeue a single frame (FIFO); returns None if queue is empty
    def dequeue(self) -> Optional[np.ndarray]:
        """
        Remove and return the next frame from the queue.

        Args:
            copy: if True, return a copy to prevent downstream mutation
                  from affecting stored data (mostly relevant for peeking or
                  when returning without popping in alternate APIs).
        Returns:
            np.ndarray or None if the queue is empty.
        """
        with self.lock:
            self._mark_closed_if_idle()
            if not self.frames:
                return None
            return self.frames.pop(0)

    # Peek at the next frame without removing; returns None if empty
    def peek(self) -> Optional[np.ndarray]:
        with self.lock:
            self._mark_closed_if_idle()
            if not self.frames:
                return None
            return self.frames[0]

    # Current number of enqueued frames
    def __len__(self) -> int:
        with self.lock:
            return len(self.frames)

    # Explicitly close the queue (e.g., upstream ended)
    def close(self) -> None:
        with self.lock:
            self.closed = True

    # Helper to check if queue is considered closed
    def is_closed(self) -> bool:
        with self.lock:
            self._mark_closed_if_idle()
            return self.closed