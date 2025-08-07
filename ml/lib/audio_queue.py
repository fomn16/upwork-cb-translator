import threading
import time

# ----------------- AudioQueue ----------------- #
# class responsible for handling the queue used for audio with thread safety
class AudioQueue:
    def __init__(self):
        self.data = bytearray()  # Array that stores the audio queue
        self.lock = threading.Lock()  # Lock used to control access between threads
        self.closed = False  # Indicates when the process must be stopped
        self.timeout_seconds = 10 * 60  # Stops the threads after 10 min of inactivity
        self.last_write = time.perf_counter()  # Saves last enqueue time

    # Appends new data to the queue
    def enqueue(self, new_data: bytes):
        with self.lock:
            self.data.extend(new_data)
            self.last_write = time.perf_counter()

    # Reads and removes the specified number of bytes from the queue
    def dequeue(self, size = -1):
        with self.lock:
            if time.perf_counter() - self.last_write > self.timeout_seconds:
                self.closed = True
            if len(self.data) == 0:
                return b""
            if size > len(self.data) or size < 0:
                size = len(self.data)
            dequeued_data = self.data[:size]
            self.data = self.data[size:]
            return dequeued_data