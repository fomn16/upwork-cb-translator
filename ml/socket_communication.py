from typing import Dict
from lib.communication.communication_helper import CommunicationHelper
from subprocess import Popen
from config.connection_config import *
from lib.communication.session_settings import SessionSettings

audio_out_pipes: Dict[str, Popen[bytes]]= {}
video_out_pipes: Dict[str, Popen[bytes]]= {}
session_settings:Dict[str, SessionSettings] = {}

def receive_synced_audio(session_id:str, audio_bytes:bytes):
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

def receive_synced_video(session_id:str, video_bytes:bytes):
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