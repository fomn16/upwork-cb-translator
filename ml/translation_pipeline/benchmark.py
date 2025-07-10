import time
from typing import Optional, Type, Any

class Benchmark:
    def __init__(self, name: str):
        """
        :param name: Name of the benchmark (used in printed output)
        """
        self.name = name
        self.times = []  # Buffer to store the last 100 execution times
        self.min_time = float('inf')  # Global minimum time
        self.max_time = float('-inf')  # Global maximum time

    def __enter__(self):
        """
        Start timing when entering the `with` block.
        """
        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self, 
        exc_type: Optional[Type[BaseException]], 
        exc_val: Optional[BaseException], 
        exc_tb: Optional[Any]
    ) -> bool:
        """
        Stop timing when exiting the `with` block and record the elapsed time.
        """
        elapsed_time = time.perf_counter() - self.start_time
        self.times.append(elapsed_time)

        # Update global min and max times
        self.min_time = min(self.min_time, elapsed_time)
        self.max_time = max(self.max_time, elapsed_time)

        # Check if we have reached 100 calls
        if len(self.times) == 100:
            self.show()
            # Reset the buffer
            self.times = []

        return False  # Don't suppress exceptions
    
    def show(self):
        avg_time = sum(self.times) / len(self.times)
        print(
            f"[{self.name}] Last 100 blocks stats: "
            f"Average time: {avg_time:.6f} seconds | "
            f"Min time: {self.min_time:.6f} seconds | "
            f"Max time: {self.max_time:.6f} seconds"
        )