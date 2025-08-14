import threading
import time
from collections import deque

class AudioQueue:
    def __init__(self):
        self.queue = deque()  # store chunks of bytes
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.closed = False
        self.timeout_seconds = 10 * 60
        self.last_write = time.perf_counter()
        self.total_bytes = 0  # track total size without recomputing

    def enqueue(self, new_data: bytes):
        with self.not_empty:
            self.queue.append(new_data)
            self.total_bytes += len(new_data)
            self.last_write = time.perf_counter()
            self.not_empty.notify()  # wake up consumer if waiting

    def dequeue(self, size=-1, block=True, timeout=None):
        with self.not_empty:
            # Wait until data is available or closed
            if block:
                if not self.not_empty.wait_for(lambda: self.total_bytes > 0 or self.closed, timeout):
                    return b""

            if time.perf_counter() - self.last_write > self.timeout_seconds:
                self.closed = True

            if self.total_bytes == 0:
                return b""

            if size < 0 or size > self.total_bytes:
                size = self.total_bytes

            chunks = []
            remaining = size
            while remaining > 0 and self.queue:
                chunk = self.queue[0]
                if len(chunk) <= remaining:
                    chunks.append(chunk)
                    self.queue.popleft()
                    self.total_bytes -= len(chunk)
                    remaining -= len(chunk)
                else:
                    chunks.append(chunk[:remaining])
                    self.queue[0] = chunk[remaining:]
                    self.total_bytes -= remaining
                    remaining = 0

            return b"".join(chunks)

    def __len__(self):
        with self.lock:
            return self.total_bytes
        
    def close(self):
        with self.lock:
            self.closed = True