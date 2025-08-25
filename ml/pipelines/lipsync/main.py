import os
import sys
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import threading
import numpy as np
import time

import cv2

from typing import Dict
from lib.communication.communication_helper import CommunicationHelper
from lib.communication.session_settings import SessionSettings
from lib.queue.audio_queue import AudioQueue
from lib.queue.video_queue import VideoQueue

from config.video_config import *
from config.connection_config import *
from config.lipsync_config import *
from config.audio_config import *

from helpers import *

# used to avoid calculating every time
VIDEO_INPUT_BUFFER_SIZE = MAX_LIPSYNC_MODEL_CHUNK_SECONDS*FRAME_RATE
VIDEO_OUTPUT_BUFFER_SIZE = OUTPUT_QUEUE_SIZE_SECONDS*FRAME_RATE
AUDIO_INPUT_BUFFER_SIZE = MAX_LIPSYNC_MODEL_CHUNK_SECONDS*INTERNAL_SAMPLERATE*2 # *2 because audio queue is in bytes, but we are working with 16 bit samples (2 bytes)
AUDIO_OUTPUT_BUFFER_SIZE = OUTPUT_QUEUE_SIZE_SECONDS*INTERNAL_SAMPLERATE*2

class Session:
    def __init__(self, session_id:str, settings:SessionSettings):
        self.closed = False
        self.session_id = session_id
        self.settings = settings

        self.raw_audio_in = AudioQueue()

        self.translated_audio_in = AudioQueue()

        self.video_in = VideoQueue()
        self.face_positions = FaceDetectProcessor()

        self.received_data = threading.Event()

        self.audio_out = AudioQueue()
        self.video_out = VideoQueue()

        self.empty_audio_chunk = self.generate_empty_audio()

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

    def generate_empty_audio(self):
        # === Enqueue matching silent audio ===
        # Duration of one frame in seconds
        frame_duration_sec = 1.0 / FRAME_RATE
        # Number of audio samples for that duration
        num_samples = int(round(frame_duration_sec * INTERNAL_SAMPLERATE))
        # Create silent audio (16-bit PCM, so 2 bytes per sample)
        return (np.zeros(num_samples, dtype=np.int16)).tobytes()
    
    # raw audio is not used for now
    def add_raw_audio(self, audio_bytes):
        #self.raw_audio_in.enqueue(audio_bytes)
        pass

    def add_translated_audio(self, audio_bytes):
        #if the lipsync doesnt need to be applied, fowards audio directly to the output
        if self.settings.enable_video and self.settings.enable_translation and self.settings.enable_lip_sync:
            self.translated_audio_in.enqueue(audio_bytes)
        else:
            self.audio_out.enqueue(audio_bytes)

    def add_video(self, video_bytes):
        # TODO, send blank frame to avoid video freezing on last frame sent if video is disabled mid-call
        if self.settings.enable_video: # only enqueue if the video is enabled
            frame = np.frombuffer(video_bytes, np.uint8).reshape((FRAME_HEIGHT,FRAME_WIDTH, 3))
            if(self.settings.enable_translation and self.settings.enable_lip_sync): # if lipsync must be applied, sends the video to the input queue
                self.face_positions.enqueue(frame)
                self.video_in.enqueue(frame)
                self.received_data.set()
            else:   # otherwise, bypasses the lipsync code entirely
                self.video_out.enqueue(frame)

    def update_settings(self, settings:SessionSettings):
        self.settings = settings

    def close(self):
        self.closed = True
        self.translated_audio_in.close()
        self.raw_audio_in.close()
        self.video_in.close()
        self.face_positions.close()


    def draw_debug_arrow(self, video_frame, x1, y1, x2, y2):
        # --- Rotation-aware indicator (arrow pointing outward) ---
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Define anchor points (on the box edges)
        pts = [
            (cx, y1),  # top center
            (x1, cy),  # left center
            (cx, y2),  # bottom center
            (x2, cy),  # right center
        ]

        # Pick the active point based on rotation
        idx = self.settings.camera_rotation % 4
        px, py = pts[idx]

        # Define outward offset for arrow start (so arrow points outward from (px, py))
        offset = 40
        if idx == 0:
            start = (px, py - offset)
        elif idx == 1:
            start = (px - offset, py)
        elif idx == 2:
            start = (px, py + offset)
        elif idx == 3:
            start = (px + offset, py)

        # Draw arrow pointing outward from the box
        cv2.arrowedLine(
            video_frame,
            (px, py), start,    # arrow from box edge → outward
            (0, 0, 255),        # red
            2,                  # thickness
            tipLength=0.4       # relative size of arrowhead
        )

    def process(self):
        try:
            while True: 
                timeout = not self.received_data.wait(timeout=30)  # waits for data to be received, also has a timeout
                if (self.raw_audio_in.closed or 
                    self.translated_audio_in.closed or 
                    self.video_in.closed or
                    self.settings.closed):
                    break
                if timeout:
                    continue  # if woke up due to timeout, goes to the next loop iteration
                
                available_audio_bytes = len(self.translated_audio_in)
                available_video_frames = len(self.video_in)
                if available_video_frames >= VIDEO_INPUT_BUFFER_SIZE:  # waits until there is enough video buffered on the input
                    available_audio_time = available_audio_bytes / (2 * INTERNAL_SAMPLERATE)
                    available_video_time = available_video_frames / FRAME_RATE

                    if available_audio_time > 0:
                        while available_audio_time > 0 and available_video_time > MIN_LIPSYNC_MODEL_CHUNK_SECONDS:
                            chunk_seconds = min(
                                available_video_time,
                                available_audio_time,
                                MAX_LIPSYNC_MODEL_CHUNK_SECONDS
                            )
                            
                            # If chunk is too short, pad it with silence
                            audio_padding_needed = False
                            if chunk_seconds < MIN_LIPSYNC_MODEL_CHUNK_SECONDS:
                                chunk_seconds = MIN_LIPSYNC_MODEL_CHUNK_SECONDS
                                audio_padding_needed = True

                            video_frames_to_process = int(round(chunk_seconds * FRAME_RATE))
                            audio_bytes_to_process = int(round(chunk_seconds * INTERNAL_SAMPLERATE)) * 2

                            video_frames_to_process = min(video_frames_to_process, available_video_frames)
                            audio_bytes_to_process = min(audio_bytes_to_process, available_audio_bytes)

                            video_for_lipsync = []
                            positions_for_lipsync = []
                            for _ in range(video_frames_to_process):
                                video_for_lipsync.append(self.video_in.dequeue().copy())
                                positions_for_lipsync.append(self.face_positions.dequeue())

                            audio_for_lipsync = self.translated_audio_in.dequeue(audio_bytes_to_process)
                            
                            # Pad audio if needed
                            if audio_padding_needed:
                                expected_samples = int(MIN_LIPSYNC_MODEL_CHUNK_SECONDS * INTERNAL_SAMPLERATE)
                                current_samples = len(audio_for_lipsync) // 2  # 16-bit PCM = 2 bytes/sample
                                if current_samples < expected_samples:
                                    pad_samples = expected_samples - current_samples
                                    silence = (np.zeros(pad_samples, dtype=np.int16)).tobytes()
                                    audio_for_lipsync += silence

                            synced_video = run_lipsync_from_frames(
                                video_for_lipsync,
                                audio_for_lipsync,
                                positions_for_lipsync,
                                self.settings.camera_rotation
                            )

                            for synced_video_frame in synced_video:
                                self.video_out.enqueue(synced_video_frame)

                            self.audio_out.enqueue(audio_for_lipsync)

                            # Update available times
                            available_video_frames = len(self.video_in)
                            available_audio_bytes = len(self.translated_audio_in)
                            available_audio_time = available_audio_bytes / (2 * INTERNAL_SAMPLERATE)
                            available_video_time = available_video_frames / FRAME_RATE

                    else:  # No translated audio yet → pass video through
                        video_frame = self.video_in.dequeue().copy()
                        f = self.face_positions.dequeue()  # still dequeue to keep queues in sync
                        if f is not None:
                            x1, y1, x2, y2 = f
                            cv2.rectangle(video_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            self.draw_debug_arrow(video_frame, x1,y1,x2,y2)

                        self.video_out.enqueue(video_frame)
                        self.audio_out.enqueue(self.empty_audio_chunk)

        except Exception as e:
            print(f"Error in processing thread: {e}")
            raise
        finally:
            self.close()

    def send_audio_data(self):
        global translated_audio_socket
        start_time = time.perf_counter()
        frame_index = 0
        while not self.closed:
            target_time = start_time + frame_index * AUDIO_CHUNK_DURATION
            frame_index += 1
            audio = self.audio_out.dequeue(INTERNAL_N_AUDIO_CHUNK_BYTES, timeout=AUDIO_CHUNK_DURATION/2) # blocks for half the send rate in the worst case
            if audio:
                translated_audio_socket.send(self.session_id, audio)

            sleep_time = target_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    def send_video_data(self):
        global video_socket
        time_per_video_send = 1/FRAME_RATE
        while True: # waits for output queue to be filled before starting to return data
            time.sleep(time_per_video_send)
            if len(self.video_out) >= VIDEO_OUTPUT_BUFFER_SIZE:
                break
        print("video output has enough data to start streaming")

        start_time = time.perf_counter()
        frame_index = 0

        while not self.closed:
            target_time = start_time + frame_index * time_per_video_send
            frame_index += 1

            frame = self.video_out.dequeue()
            if frame is not None and getattr(frame, "size", 0) > 0:
                video_socket.send(self.session_id, frame.tobytes(order="C"))

            sleep_time = target_time - time.perf_counter()

            # Slight speed-up if buffer is too full
            if len(self.video_out) > VIDEO_OUTPUT_BUFFER_SIZE:
                sleep_time *= 0.9

            if sleep_time > 0:
                time.sleep(sleep_time)

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self.lock = threading.Lock()

    def is_initialized(self, session_id, settings:SessionSettings|None = None):
        with self.lock:
            if session_id not in self.sessions:     # if session was not created yet
                if(settings is None):               # only create session on the first "receive_session_settings" call
                    return False
                self.sessions[session_id] = None    # stops other threads from trying to acces the session while its created below
            else:
                return self.sessions[session_id] is not None

        session = Session(session_id, settings)

        with self.lock:
            self.sessions[session_id] = session
        
        return True

    def add_translated_audio_to_session(self, session_id, raw_bytes):
        if self.is_initialized(session_id):
            self.sessions[session_id].add_translated_audio(raw_bytes)

    def add_raw_audio_to_session(self, session_id, raw_bytes):
        if self.is_initialized(session_id):
            self.sessions[session_id].add_raw_audio(raw_bytes)

    def add_video_to_session(self, session_id, raw_bytes):
        if self.is_initialized(session_id):
            self.sessions[session_id].add_video(raw_bytes)

    def close(self):
        for sess in self.sessions.values():
            sess.close()

    def receive_session_settings(self, session_id:str, settings:SessionSettings):
        if self.is_initialized(session_id, settings):   # on first call, this also initializes the session with the settings
            self.sessions[session_id].update_settings(settings)

if __name__ == "__main__":
    try:
        print("[Lip-Sync] Creating session manager")
        manager = SessionManager()

        translated_audio_socket = CommunicationHelper("translated_audio_socket", LIP_SYNC_TRANSLATED_AUDIO_IN_PORT, LIP_SYNC_AUDIO_OUT_PORT, manager.add_translated_audio_to_session, manager.receive_session_settings)
        raw_audio_socket = CommunicationHelper("raw_audio_socket", LIP_SYNC_RAW_AUDIO_IN_PORT, None, manager.add_raw_audio_to_session)
        video_socket = CommunicationHelper("video_socket", LIP_SYNC_VIDEO_IN_PORT, LIP_SYNC_VIDEO_OUT_PORT, manager.add_video_to_session)

        while(raw_audio_socket.online and translated_audio_socket.online and video_socket.online):
            time.sleep(1)
    finally:
        print("[Lip-Sync] Closing all processing")
        raw_audio_socket.close()
        translated_audio_socket.close()
        video_socket.close()
        manager.close()