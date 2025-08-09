import os
import sys
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import threading
import numpy as np
import time

from typing import Dict
from lib.communication_helper import CommunicationHelper
from lib.audio_queue import AudioQueue
from lib.video_queue import VideoQueue

from config.video_config import *
from config.connection_config import *
from config.lipsync_config import *

from collections import deque

# histogram with fast max lookup
class MaxAudioQueueSizeHistory:
    def __init__(self):
        self.queue = deque()
        self.max_deque = deque()

    def push(self, value):
        self.queue.append(value)
        if len(self.queue) > AUDIO_QUEUE_HISTORY_SIZE:
            removed = self.queue.popleft()
            if removed == self.max_deque[0]:
                self.max_deque.popleft()

        while self.max_deque and self.max_deque[-1] < value:
            self.max_deque.pop()
        self.max_deque.append(value)

    def get_max(self):
        return self.max_deque[0] if self.max_deque else MINIMUM_AUDIO_BUFFER_SIZE

class LipSyncManager:
    def __init__(self, audio_in: AudioQueue, video_in: VideoQueue):
        self.translated_audio = audio_in
        self.video = video_in
        self.audio_len_history = MaxAudioQueueSizeHistory()
        self.video_speed_change = 0

class Session:
    def __init__(self, session_id:str):
        self.session_id = session_id
        self.raw_audio_in = AudioQueue()
        self.translated_audio_in = AudioQueue()
        self.video_in = VideoQueue()

        threading.Thread(
            target=self.process,
            daemon=True,
        ).start()

    def add_raw_audio(self, audio_bytes):
        self.raw_audio_in.enqueue(audio_bytes)

    def add_translated_audio(self, audio_bytes):
        self.translated_audio_in.enqueue(audio_bytes)

    def add_video(self, video_bytes):
        frame = np.frombuffer(video_bytes, np.uint8).reshape((FRAME_WIDTH, FRAME_HEIGHT, 3))
        self.video_in.enqueue(frame)

    def process(self):
        global translated_audio_socket, video_socket
        next_time = time.perf_counter()
        try:
            while not self.raw_audio_in.closed and not self.translated_audio_in.closed and not self.video_in.closed:
                # TODO, simple passthrough for now
                audio_seg = self.translated_audio_in.dequeue(SAMPLE_READ_SIZE)
                if audio_seg:
                    translated_audio_socket.send(self.session_id, audio_seg)

                video_frame = self.video_in.dequeue()
                if video_frame is not None and getattr(video_frame, "size", 0) > 0:
                    video_socket.send(self.session_id, video_frame.tobytes(order="C"))

                # TODO, find a way to avoid this (execute loop as soon as any info is received)
                next_time += OUTPUT_PERIOD
                sleep_time = next_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_time = time.perf_counter()
        except Exception as e:
            print(f"Error in processing thread: {e}")
            raise
        finally:
            self.translated_audio_in.closed = True  # TODO, refactor to work the same way as the video queue
            self.video_in.close()

class SessionManager:
    sessions_dict: Dict[str, Session] = {}

    def ensure_session_exists(self, session_id:str):
        if session_id not in self.sessions_dict:
            print(f"[Lip-Sync->SessionManager] Creating session with id {session_id}")
            self.sessions_dict[session_id] = Session(session_id)

    def add_translated_audio_to_session(self, session_id, raw_bytes):
        self.ensure_session_exists(session_id)
        self.sessions_dict[session_id].add_translated_audio(raw_bytes)

    def add_raw_audio_to_session(self, session_id, raw_bytes):
        self.ensure_session_exists(session_id)
        self.sessions_dict[session_id].add_raw_audio(raw_bytes)

    def add_video_to_session(self, session_id, raw_bytes):
        self.ensure_session_exists(session_id)
        self.sessions_dict[session_id].add_video(raw_bytes)

def raw_audio_received(session_id, raw_bytes):
    global sessionManager
    sessionManager.add_raw_audio_to_session(session_id, raw_bytes)

def translated_audio_received(session_id, raw_bytes):
    global sessionManager
    sessionManager.add_translated_audio_to_session(session_id, raw_bytes)

def video_received(session_id, raw_bytes):
    global sessionManager
    sessionManager.add_video_to_session(session_id, raw_bytes)

if __name__ == "__main__":
    raw_audio_socket = CommunicationHelper("raw_audio_socket", LIP_SYNC_RAW_AUDIO_IN_PORT, None, raw_audio_received)
    translated_audio_socket = CommunicationHelper("translated_audio_socket", LIP_SYNC_TRANSLATED_AUDIO_IN_PORT, LIP_SYNC_AUDIO_OUT_PORT, translated_audio_received)
    video_socket = CommunicationHelper("video_socket", LIP_SYNC_VIDEO_IN_PORT, LIP_SYNC_VIDEO_OUT_PORT, video_received)

    try:
        print("[Lip-Sync] Creating session manager")
        sessionManager = SessionManager()

        while(raw_audio_socket.online and  translated_audio_socket.online and video_socket.online):
            time.sleep(1)
    finally:
        print("[Lip-Sync] Closing all processing")
        translated_audio_socket.close()
        video_socket.close()