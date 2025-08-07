import threading
from typing import Callable

import zmq


class CommunicationHelper:
    """
    A minimal ZeroMQ helper class.

    - On init: provide send_port, recv_port, and a recv_callback.
    - send(conn_id: str, payload: bytes) to send messages.
    - Receives messages as (str connection id, bytes) and invokes callback.

    Notes:
      - Uses PUSH (sender) / PULL (receiver) pattern.
      - Uses a background thread for receiving.
      - Messages are sent/received as two frames:
          [frame 0: UTF-8 string bytes][frame 1: raw bytes]

    example:

        from communication_helper import CommunicationHelper
        import time
        import pickle

        def received_data_callback(a: str, b: bytes):
            print(time.perf_counter() - pickle.loads(b))

        with CommunicationHelper("test_1", 8003, 8002, received_data_callback) as ch:
            i = 0
            while True:
                ch.send(f"user-{i}", pickle.dumps(
                    time.perf_counter(), protocol=pickle.HIGHEST_PROTOCOL
                ))
                i += 1
                time.sleep(1)
    """

    def __init__(
        self,
        name: str,
        recv_port: int,
        send_port: int,
        recv_callback: Callable[[str, bytes], None],
        run_in_another_thread: bool = True,
        com_method: str = "ipc",
        host: str = "127.0.0.1",
        encoding: str = "utf-8",
        errors: str = "strict",
    ):
        """
        Args:
            send_port: Port used for sending messages.
            recv_port: Port used for receiving messages.
            recv_callback: Function invoked with (conn_id, payload).
            host: Host for connect/bind (default localhost).
            encoding: Encoding used for conn_id string (default utf-8).
            errors: Error handling for encode/decode (default 'strict').
        """
        self._ctx = zmq.Context.instance()

        self._recv_sock = self._ctx.socket(zmq.PULL)
        self._recv_callback = recv_callback
        self._send_sock = self._ctx.socket(zmq.PUSH)

        self._stop = threading.Event()
        self.online = True
        self.name = name
        self._encoding = encoding
        self._errors = errors

        if(com_method == "tcp"):
            send_addr = f"tcp://{host}:{send_port}"
            recv_addr = f"tcp://{host}:{recv_port}"
        else:
            send_addr = f"ipc:///tmp/com_zmq_{send_port}.ipc"
            recv_addr = f"ipc:///tmp/com_zmq_{recv_port}.ipc"

        self._recv_sock.bind(recv_addr)
        self._send_sock.connect(send_addr)

        self.send_lock = threading.Lock()

        if run_in_another_thread:
            self._thread = threading.Thread(
                target=self._recv_loop,
                name=f"CommunicationHelperRecv_{name}",
                daemon=True,
            )
            self._thread.start()
        else:
            self._recv_loop()

    def send(self, conn_id: str, payload: bytes) -> None:
        """
        Send a message as two frames: UTF-8 string bytes + raw bytes.
        """
        if not isinstance(conn_id, str):
            raise TypeError("conn_id must be a string")
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")

        header = conn_id.encode(self._encoding, errors=self._errors)
        with self.send_lock:
            self._send_sock.send_multipart([header, bytes(payload)])

    def close(self) -> None:
        """
        Stop the receiver thread and close sockets.
        """
        self._stop.set()
        try:
            self._recv_sock.setsockopt(zmq.RCVTIMEO, 100)
        except Exception:
            pass
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        self._send_sock.close(linger=0)
        self._recv_sock.close(linger=0)
        # Do not terminate shared context here.

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

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
                    conn_id_bytes = parts[0]
                    payload = parts[1]
                    conn_id = conn_id_bytes.decode(
                        self._encoding, errors=self._errors
                    )
                    self._recv_callback(conn_id, payload)
                except zmq.Again:
                    continue
                except Exception as e:
                    print("Error in the interprocess communication: ", e)
                    continue
        self.online = False