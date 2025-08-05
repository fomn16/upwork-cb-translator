import threading
from typing import Callable

import zmq

class CommunicationHelper:
    """
    A minimal ZeroMQ helper class.

    - On init: provide send_port, recv_port, and a recv_callback.
    - send(conn_id: int, payload: bytes) to send messages.
    - Receives messages as (int connection id, bytes) and invokes callback.

    Notes:
      - Uses PUSH (sender) / PULL (receiver) pattern.
      - Uses a background thread for receiving.
      - Messages are sent/received as two frames:
          [frame 0: 8-byte big-endian int][frame 1: raw bytes]

    example:
    
        from communication_helper import CommunicationHelper
        import time
        import pickle

        def received_data_callback(a: int, b: bytes):
            print(time.perf_counter() - pickle.loads(b))

        with CommunicationHelper("test_1", 8002, 8003, received_data_callback) as ch:
            i = 0
            while(True):
                ch.send(i, pickle.dumps(time.perf_counter(), protocol=pickle.HIGHEST_PROTOCOL))
                i+=1
                time.sleep(1)
    """

    def __init__(
        self,
        name: str,
        send_port: int,
        recv_port: int,
        recv_callback: Callable[[int, bytes], None],
        host: str = "127.0.0.1",
    ):
        """
        Args:
            send_port: Port used for sending messages.
            recv_port: Port used for receiving messages.
            recv_callback: Function invoked with (conn_id, payload).
            host: Host for connect/bind (default localhost).
        """
        self._ctx = zmq.Context.instance()
        self._send_sock = self._ctx.socket(zmq.PUSH)
        self._recv_sock = self._ctx.socket(zmq.PULL)
        self._recv_callback = recv_callback
        self._stop = threading.Event()
        self.name = name

        send_addr = f"tcp://{host}:{send_port}"
        recv_addr = f"tcp://{host}:{recv_port}"

        
        self._send_sock.connect(send_addr)

        self._recv_sock.bind(recv_addr)

        self._thread = threading.Thread(
            target=self._recv_loop, name=f"CommunicationHelperRecv_{name}", daemon=True
        )
        self._thread.start()

    def send(self, conn_id: int, payload: bytes) -> None:
        """
        Send a message as two frames: 8-byte big-endian int + raw bytes.
        """
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
        header = conn_id.to_bytes(8, byteorder="big", signed=True)
        self._send_sock.send_multipart([header, bytes(payload)])

    def close(self) -> None:
        """
        Stop the receiver thread and close sockets.
        """
        self._stop.set()
        try:
            # Trigger the receiver to unblock by using a short timeout option.
            self._recv_sock.setsockopt(zmq.RCVTIMEO, 100)
        except Exception:
            pass
        self._thread.join(timeout=1.0)
        self._send_sock.close(linger=0)
        self._recv_sock.close(linger=0)
        # Context is shared (instance), so we don't terminate it here.

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # Internal receive loop
    def _recv_loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._recv_sock, zmq.POLLIN)
        while not self._stop.is_set():
            socks = dict(poller.poll(timeout=200))
            if self._recv_sock in socks and socks[self._recv_sock] == zmq.POLLIN:
                try:
                    parts = self._recv_sock.recv_multipart(flags=zmq.NOBLOCK)
                    if len(parts) != 2:
                        continue
                    conn_id = int.from_bytes(parts[0], "big", signed=True)
                    payload = parts[1]
                    self._recv_callback(conn_id, payload)
                except zmq.Again:
                    continue
                except Exception as e:
                    print("Error in the interprocess communication: ", e)
                    continue