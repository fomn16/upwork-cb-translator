from typing import Dict
from lib.communication.communication_helper import CommunicationHelper
from subprocess import Popen
from config.connection_config import *
from lib.communication.session_settings import SessionSettings

import copy


import os
import numpy as np
import subprocess
from datetime import datetime
from collections import defaultdict
import tempfile

# Buffers keyed by (session_id, base_filename)
video_buffers = defaultdict(list)
audio_buffers = defaultdict(list)
FILENAME = "debug_files"

def save_av_with_tempfiles(
    session_id,
    video_frames,
    audio_frames,
    fps,
    sample_rate,
    frame_size,
    base_filename=FILENAME,
):
    """
    Save video frames (BGR) + audio frames (PCM16-LE mono) into a single MP4 file.
    Uses temporary raw files for muxing.
    """
    os.makedirs(base_filename, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_filename = os.path.join(
        base_filename, f"{session_id}_{base_filename}_{timestamp}.mp4"
    )

    h, w = frame_size

    with tempfile.NamedTemporaryFile(delete=False, suffix=".raw") as vf, tempfile.NamedTemporaryFile(
        delete=False, suffix=".raw"
    ) as af:
        vpath, apath = vf.name, af.name

        # Write raw video
        for f in video_frames:
            vf.write(f.tobytes())

        # Write raw audio
        for a in audio_frames:
            af.write(a)

    # Now mux with ffmpeg
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            vpath,
            "-f",
            "s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-i",
            apath,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            out_filename,
        ],
        check=True,
    )

    os.remove(vpath)
    os.remove(apath)

    print(f"[{session_id}] Saved {out_filename}")
    return out_filename


def collect_video_frame(
    frame_bytes,
    session_id,
    fps=15,
    base_filename=FILENAME,
    duration=40,
    frame_size=(480, 640),
):
    """
    Collect a single video frame (BGR raw bytes).
    Saves when enough frames are collected AND audio is also available.
    """
    key = (session_id, base_filename)

    # Convert raw BGR bytes → NumPy array
    frame = np.frombuffer(frame_bytes, np.uint8).reshape(
        (frame_size[0], frame_size[1], 3)
    )
    video_buffers[key].append(frame)

    # Check if both audio and video have enough data
    if len(video_buffers[key]) >= fps * duration:
        maybe_save_av(session_id, fps, frame_size, base_filename)


def collect_audio_chunk(
    audio_bytes,
    session_id,
    sample_rate=16000,
    base_filename=FILENAME,
    duration=20,
    fps=15,
    frame_size=(480, 640),
):
    """
    Collect a chunk of PCM16-LE mono audio.
    Saves when enough audio is collected AND video is also available.
    """
    key = (session_id, base_filename)
    audio_buffers[key].append(audio_bytes)

    # Each second of audio = sample_rate * 2 bytes (int16 mono)
    bytes_per_second = sample_rate * 2
    total_bytes = sum(len(b) for b in audio_buffers[key])

    if total_bytes >= bytes_per_second * duration:
        maybe_save_av(session_id, fps, frame_size, base_filename)


def maybe_save_av(session_id, fps, frame_size, base_filename=FILENAME, sample_rate=16000):
    """
    Check if both audio and video buffers have enough data to save.
    """
    key = (session_id, base_filename)

    # Compute how many seconds of video we have
    video_seconds = len(video_buffers[key]) / fps
    audio_bytes = sum(len(b) for b in audio_buffers[key])
    audio_seconds = audio_bytes / (sample_rate * 2)

    # Only save if both have at least the target duration
    if video_seconds >= 1 and audio_seconds >= 1:
        save_av_with_tempfiles(
            session_id,
            video_buffers[key],
            audio_buffers[key],
            fps,
            sample_rate,
            frame_size,
            base_filename,
        )
        video_buffers[key].clear()
        audio_buffers[key].clear()

audio_out_pipes: Dict[str, Popen[bytes]]= {}
video_out_pipes: Dict[str, Popen[bytes]]= {}
session_settings:Dict[str, SessionSettings] = {}

def receive_synced_audio(session_id:str, audio_bytes:bytes):
    '''global audio_out_pipes
    if session_id not in audio_out_pipes:
        print(f"⚠️ [received_synced_audio]: session {session_id} not found.")
        return
    out_pipe = audio_out_pipes[session_id]
    try:
        out_pipe.stdin.write(audio_bytes)
        out_pipe.stdin.flush()'''
    collect_audio_chunk(audio_bytes, session_id)
    '''except BrokenPipeError:
        print("⚠️ FFmpeg-audio-out pipe closed")
        return'''

def receive_synced_video(session_id:str, video_bytes:bytes):
    '''global video_out_pipes
    if session_id not in video_out_pipes:
        print(f"⚠️ [receive_synced_video]: session {session_id} not found.")
        return
    out_pipe = video_out_pipes[session_id]
    try:
        out_pipe.stdin.write(video_bytes)'''
    collect_video_frame(video_bytes, session_id)
    '''except BrokenPipeError:
        print("⚠️ FFmpeg-video-out pipe closed")
        return'''
    
def receive_translated_audio(session_id:str, audio_bytes:bytes):
    lipsync_translated_audio_socket.send(session_id, audio_bytes)
 
lipsync_raw_audio_socket = CommunicationHelper("lipsync_raw_audio_socket", None, LIP_SYNC_RAW_AUDIO_IN_PORT, None)
lipsync_translated_audio_socket = CommunicationHelper("lipsync_translated_audio_socket", LIP_SYNC_AUDIO_OUT_PORT, LIP_SYNC_TRANSLATED_AUDIO_IN_PORT, receive_synced_audio)
lipsync_video_socket = CommunicationHelper("lipsync_video_socket", LIP_SYNC_VIDEO_OUT_PORT, LIP_SYNC_VIDEO_IN_PORT, receive_synced_video)
translate_socket = CommunicationHelper("audio_translate_socket", TRANSLATION_VOICE_CLONE_OUT_PORT, TRANSLATION_VOICE_CLONE_IN_PORT, receive_translated_audio)

# checks if settings for the session changed, and if so, broadcasts to other environments
def send_settings(session_id:str, settings:SessionSettings):
    if (session_id not in session_settings) or (session_settings[session_id] != settings):
        lipsync_translated_audio_socket.send_settings(session_id, settings)
        translate_socket.send_settings(session_id, settings)
    session_settings[session_id]=settings

def get_current_settings(session_id:str):
    if (session_id in session_settings):
        return copy.copy(session_settings[session_id])
    return None