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
from lib.queue.audio_queue import AudioQueue
from lib.queue.video_queue import VideoQueue

from config.video_config import *
from config.connection_config import *
from config.lipsync_config import *

from collections import deque

from helpers import *

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
        self.translated_audio_size_history = MaxAudioQueueSizeHistory()

        self.video_in = VideoQueue()
        self.face_positions = FaceDetectProcessor()

        self.received_data = threading.Event()

        threading.Thread(
            target=self.process,
            daemon=True,
        ).start()

    # raw audio is not used for now
    def add_raw_audio(self, audio_bytes):
        #self.raw_audio_in.enqueue(audio_bytes)
        #self.received_data.set()
        pass

    def add_translated_audio(self, audio_bytes):
        self.translated_audio_in.enqueue(audio_bytes)
        self.translated_audio_size_history.push(self.translated_audio_in.length())
        self.received_data.set()

    def add_video(self, video_bytes):
        frame = np.frombuffer(video_bytes, np.uint8).reshape((FRAME_HEIGHT,FRAME_WIDTH, 3))
        self.face_positions.enqueue(frame)
        self.video_in.enqueue(frame)
        self.received_data.set()

    def close(self):
        self.translated_audio_in.closed = True  # TODO, refactor to work the same way as the video queue
        self.raw_audio_in.closed = True  # TODO, refactor to work the same way as the video queue
        self.video_in.close()
        self.face_positions.close()

    def process(self):
        global translated_audio_socket, video_socket
        try:
            while True: 
                timeout = not self.received_data.wait(timeout=30) # waits for data to be received, also has a timeout

                if(self.raw_audio_in.closed or self.translated_audio_in.closed or self.video_in.closed):    # IMPORTANT, must be refactored if audio/video is eventually optional
                    break
                if(timeout):
                    continue # if woke up due to timeout, goes to the next loop iteration

                # TODO, simple passthrough for now
                audio_seg = self.translated_audio_in.dequeue(SAMPLE_READ_SIZE)
                if audio_seg:
                    translated_audio_socket.send(self.session_id, audio_seg)

                video_frame = self.video_in.dequeue()
                if video_frame is not None and getattr(video_frame, "size", 0) > 0:
                    # Make a writable copy
                    video_frame = video_frame.copy()
                    f = self.face_positions.dequeue()
                    if f is not None:
                        x1, y1, x2, y2 = f
                        cv2.rectangle(video_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    video_socket.send(self.session_id, video_frame.tobytes(order="C"))
        except Exception as e:
            print(f"Error in processing thread: {e}")
            raise
        finally:
            self.close()

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

        while(raw_audio_socket.online and translated_audio_socket.online and video_socket.online):
            time.sleep(1)
    finally:
        print("[Lip-Sync] Closing all processing")
        translated_audio_socket.close()
        video_socket.close()