import os
import sys
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.communication_helper import CommunicationHelper
from config.connection_config import *
from config.audio_config import *

import multiprocessing as mp
import threading

import time

class SessionProcess:
    def __init__(self, session_id):
        self.session_id = session_id
        self.parent_conn, self.child_conn = mp.Pipe()
        self.process = mp.Process(target=self._start_worker, args=(self.child_conn,))
        self.process.start()

        self.listener_thread = threading.Thread(target=self._listen_from_worker, daemon=True)
        self.listener_thread.start()

    def _start_worker(self, conn):
        import worker
        print("worker started")
        worker.run_pipeline(conn)

    def _listen_from_worker(self):
        """Listen for messages from the worker and forward them to audio_socket."""
        while True:
            try:
                msg = self.parent_conn.recv()
            except EOFError:
                break  # Worker closed connection

            if msg == "close_ack":
                break

            audio_socket.send(self.session_id, msg)

    def send_audio(self, audio_bytes):
        self.parent_conn.send(audio_bytes)

    def close(self):
        self.parent_conn.send("close")
        self.process.join()

class SessionManager:
    def __init__(self):
        self.sessions = {}

    def ensure_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionProcess(session_id)

    def translate(self, session_id, audio_bytes):
        self.ensure_session(session_id)
        self.sessions[session_id].send_audio(audio_bytes)

    def close_all(self):
        for sess in self.sessions.values():
            sess.close()

if __name__ == "__main__":
    try:
        print("[Translate] Creating session manager")
        sessionManager = SessionManager()
        audio_socket = CommunicationHelper("audio_translate_socket", TRANSLATION_VOICE_CLONE_IN_PORT, TRANSLATION_VOICE_CLONE_OUT_PORT, sessionManager.translate)
        while(audio_socket.online):
            time.sleep(1)
    finally:
        print("[Translate] Closing all processing")
        sessionManager.close()
        audio_socket.close()