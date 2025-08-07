from typing import Dict
from lib.communication_helper import CommunicationHelper
from subprocess import Popen
from config.connection_config import *

audio_out_pipes: Dict[int, Popen[bytes]]= {}
video_out_pipes: Dict[int, Popen[bytes]]= {}
def receive_synced_audio(session_id:int, audio_bytes:bytes):
    global audio_out_pipes
    if session_id not in audio_out_pipes:
        print(f"⚠️ [received_synced_audio]: session {session_id} not found.")
        return
    out_pipe = audio_out_pipes[session_id]
    try:
        out_pipe.stdin.write(audio_bytes)
        out_pipe.stdin.flush()
    except BrokenPipeError:
        print("⚠️ FFmpeg-audio-out pipe closed")
        return
    
def receive_synced_video(session_id:int, video_bytes:bytes):
    global video_out_pipes
    if session_id not in video_out_pipes:
        print(f"⚠️ [receive_synced_video]: session {session_id} not found.")
        return
    out_pipe = video_out_pipes[session_id]
    try:
        out_pipe.stdin.write(video_bytes)
    except BrokenPipeError:
        print("⚠️ FFmpeg-video-out pipe closed")
        return
    
lipsync_audio_socket = CommunicationHelper("lipsync_audio_socket", LIP_SYNC_AUDIO_OUT_PORT, LIP_SYNC_AUDIO_IN_PORT, receive_synced_audio)
lipsync_video_socket = CommunicationHelper("lipsync_video_socket", LIP_SYNC_VIDEO_OUT_PORT, LIP_SYNC_VIDEO_IN_PORT, receive_synced_video)