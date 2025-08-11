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
from config.audio_config import *

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
        return self.max_deque[0] if self.max_deque else MINIMUM_AUDIO_BUFFER_SIZE_SECONDS * EXTERNAL_SAMPLERATE

class LipSyncManager:
    def __init__(self, audio_in: AudioQueue, video_in: VideoQueue):
        self.translated_audio = audio_in
        self.video = video_in
        self.audio_len_history = MaxAudioQueueSizeHistory()
        self.video_speed_change = 0

class Session:
    def __init__(self, session_id:str):
        self.closed = False
        self.session_id = session_id

        self.raw_audio_in = AudioQueue()

        self.translated_audio_in = AudioQueue()
        self.translated_audio_size_history = MaxAudioQueueSizeHistory()

        self.video_in = VideoQueue()
        self.face_positions = FaceDetectProcessor()

        self.received_data = threading.Event()

        self.audio_out = AudioQueue()
        self.video_out = VideoQueue()
        self.input_queue_primed = False

        threading.Thread(
            target=self.process,
            daemon=True,
        ).start()

        threading.Thread(
            target=self.send_audio_data,
            daemon=True,
        ).start()

        threading.Thread(
            target=self.send_video_data,
            daemon=True,
        ).start()

    # raw audio is not used for now
    def add_raw_audio(self, audio_bytes):
        #self.raw_audio_in.enqueue(audio_bytes)
        #self.received_data.set()
        pass

    def add_translated_audio(self, audio_bytes):
        self.translated_audio_in.enqueue(audio_bytes)
        self.translated_audio_size_history.push(len(self.translated_audio_in))
        self.received_data.set()

    def add_video(self, video_bytes):
        frame = np.frombuffer(video_bytes, np.uint8).reshape((FRAME_HEIGHT,FRAME_WIDTH, 3))
        self.face_positions.enqueue(frame)
        self.video_in.enqueue(frame)
        self.received_data.set()

    def close(self):
        self.closed = True
        self.translated_audio_in.closed = True  # TODO, refactor to work the same way as the video queue
        self.raw_audio_in.closed = True  # TODO, refactor to work the same way as the video queue
        self.video_in.close()
        self.face_positions.close()

    def process(self):
        try:
            while True: 
                timeout = not self.received_data.wait(timeout=30) # waits for data to be received, also has a timeout
                if(self.raw_audio_in.closed or self.translated_audio_in.closed or self.video_in.closed):    # IMPORTANT, must be refactored if audio/video is eventually optional
                    break
                if(timeout):
                    continue # if woke up due to timeout, goes to the next loop iteration

                # TODO, simple passthrough for now
                audio_seg = self.translated_audio_in.dequeue(N_AUDIO_CHUNK_SAMPLES)
                if audio_seg:
                    self.audio_out.enqueue(audio_seg)

                video_frame = self.video_in.dequeue()
                if video_frame is not None and getattr(video_frame, "size", 0) > 0:
                    # Make a writable copy
                    video_frame = video_frame.copy()
                    f = self.face_positions.dequeue()
                    if f is not None:
                        x1, y1, x2, y2 = f
                        cv2.rectangle(video_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    self.video_out.enqueue(video_frame)

        except Exception as e:
            print(f"Error in processing thread: {e}")
            raise
        finally:
            self.close()

    def send_audio_data(self):
        global translated_audio_socket
        while True: # waits for output queue to be filled before starting to return data
            time.sleep(AUDIO_CHUNK_DURATION)
            if len(self.audio_out)/EXTERNAL_SAMPLERATE >= 2*2*OUTPUT_QUEUE_SIZE_SECONDS: # 2*2 because audio_out is in bytes, but we are working with 16 bit (2 bytes) stereo (2 channels)
                break
        print("send_audio -> has data to start!")

        next_frame_time = time.perf_counter()
        i = 0
        while not self.closed:
            i +=1
            if(i==100):
                print(len(self.audio_out), len(self.video_out))
                i = 0
            audio = self.audio_out.dequeue(2*2*N_AUDIO_CHUNK_SAMPLES) #2*2 for the same reason as above
            if audio:
                translated_audio_socket.send(self.session_id, audio)

            # Schedule next frame time
            next_frame_time += AUDIO_CHUNK_DURATION
            sleep_time = next_frame_time - time.perf_counter()
            if len(self.audio_out)/EXTERNAL_SAMPLERATE >= 2*2*OUTPUT_QUEUE_SIZE_SECONDS:
                sleep_time-=0.1# slightly speed up if we are running late on the sends (queue is increasing)
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # We're running late — skip sleep to catch up
                next_frame_time = time.perf_counter()

    def send_video_data(self):
        global video_socket
        time_per_video_send = 1/FRAME_RATE
        while True: # waits for output queue to be filled before starting to return data
            time.sleep(time_per_video_send)
            if len(self.video_out)/FRAME_RATE >= OUTPUT_QUEUE_SIZE_SECONDS:
                break
        print("send_video -> has data to start!")
        next_frame_time = time.perf_counter()
        while not self.closed:
            frame = self.video_out.dequeue()
            if frame is not None and getattr(frame, "size", 0) > 0:
                frame_to_send = frame  # update only if we have a new frame

            video_socket.send(self.session_id, frame_to_send.tobytes(order="C"))

            # Schedule next frame time
            next_frame_time += time_per_video_send
            sleep_time = next_frame_time - time.perf_counter()
            if len(self.video_out)/FRAME_RATE > OUTPUT_QUEUE_SIZE_SECONDS:
                sleep_time-=0.1 # slightly speed up if we are running late on the sends (queue is increasing)
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # We're running late — skip sleep to catch up
                next_frame_time = time.perf_counter()

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
        raw_audio_socket.close()
        translated_audio_socket.close()
        video_socket.close()