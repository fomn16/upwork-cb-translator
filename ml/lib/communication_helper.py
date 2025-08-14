import threading
import time
from typing import Callable
import heapq
import zmq

class CommunicationHelper:
    def __init__(
        self,
        name: str,
        recv_port: int | None,
        send_port: int | None,
        recv_callback: Callable[[str, bytes], None],
        run_in_another_thread: bool = True,
        com_method: str = "ipc",
        host: str = "127.0.0.1",
        encoding: str = "utf-8",
        errors: str = "strict",
        reorder_buffer_size: int = 10,        ### Max number of out-of-order messages to hold
    ):
        self._ctx = zmq.Context.instance()
        self._recv_sock = self._ctx.socket(zmq.PULL)
        self._recv_callback = recv_callback
        self._send_socks = []
        self._stop = threading.Event()
        self.online = True
        self.name = name
        self._encoding = encoding
        self._errors = errors

        ### Sequence tracking
        self._send_seq = {}
        self._expected_seq = {}   # expected next seq per conn_id
        self._reorder_buffers = {}  # conn_id -> {seq: (data, arrival_time)}
        self._reorder_buffer_size = reorder_buffer_size

        if com_method == "tcp":
            send_addr = f"tcp://{host}:{send_port}"
            recv_addr = f"tcp://{host}:{recv_port}"
        else:
            send_addr = f"ipc:///tmp/com_zmq_{send_port}.ipc"
            recv_addr = f"ipc:///tmp/com_zmq_{recv_port}.ipc"

        if send_port is not None:
            self.send_lock = threading.Lock()
            self.send_addr = send_addr
        else:
            self.send_lock = None

        if recv_port is not None:
            self._recv_sock.bind(recv_addr)
            if run_in_another_thread:
                self._thread = threading.Thread(
                    target=self._recv_loop,
                    name=f"CommunicationHelperRecv_{name}",
                    daemon=True,
                )
                self._thread.start()
            else:
                self._recv_loop()

    def borrow_socket(self):
        with self.send_lock:
            if self._send_socks:
                return self._send_socks.pop()
            new_socket = self._ctx.socket(zmq.PUSH)
            new_socket.connect(self.send_addr)
            return new_socket

    def return_socket(self, socket):
        with self.send_lock:
            self._send_socks.append(socket)

    def send(self, conn_id: str, payload: bytes) -> None:
        if self.send_lock is None:
            raise TypeError("send_port must be provided to send messages")
        if not isinstance(conn_id, str):
            raise TypeError("conn_id must be a string")
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")

        ### Get sequence number for this conn_id
        seq = self._send_seq.get(conn_id, 0)
        self._send_seq[conn_id] = (seq + 1) % (2**32)  # wrap-around safe

        header = conn_id.encode(self._encoding, errors=self._errors)
        seq_bytes = seq.to_bytes(4, "big", signed=False)

        socket = self.borrow_socket()
        try:
            socket.send_multipart([header, seq_bytes + bytes(payload)])
        finally:
            self.return_socket(socket)

    def close(self) -> None:
        self._stop.set()
        try:
            self._recv_sock.setsockopt(zmq.RCVTIMEO, 100)
        except Exception:
            pass
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        self._recv_sock.close(linger=0)
        with self.send_lock():
            for s in self._send_socks:
                s.close(linger=0)

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
                    conn_id = parts[0].decode(self._encoding, errors=self._errors)
                    payload = parts[1]

                    seq = int.from_bytes(payload[:4], "big", signed=False)
                    data = payload[4:]

                    self._handle_incoming(conn_id, seq, data)

                except zmq.Again:
                    continue
                except Exception as e:
                    print("Error in the interprocess communication: ", e)
                    continue
        self.online = False

    def _handle_incoming(self, conn_id: str, seq: int, data: bytes):
        expected = self._expected_seq.get(conn_id, 0)
        buf = self._reorder_buffers.setdefault(conn_id, [])

        if seq == expected:
            # deliver immediately
            self._deliver(conn_id, data)
            expected = (expected + 1) % (2**32)

            # flush any consecutive buffered frames
            while buf and buf[0][0] == expected:
                _, next_data = heapq.heappop(buf)
                self._deliver(conn_id, next_data)
                expected = (expected + 1) % (2**32)

        elif seq > expected:
            # store in heap
            heapq.heappush(buf, (seq, data))

            # if buffer too large, drop missing frames up to oldest
            if len(buf) > self._reorder_buffer_size:
                oldest_seq, oldest_data = heapq.heappop(buf)
                expected = (oldest_seq + 1) % (2**32)
                self._deliver(conn_id, oldest_data)

        # else seq < expected → already delivered, ignore

        self._expected_seq[conn_id] = expected

    def _deliver(self, conn_id: str, data: bytes):
        try:
            self._recv_callback(conn_id, data)
        except Exception as e:
            print(f"Error in recv_callback: {e}")