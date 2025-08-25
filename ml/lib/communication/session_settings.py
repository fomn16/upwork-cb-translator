from dataclasses import dataclass
from .mediassoup_request import *

# stores all session settings that must be breadcast to the environments
@dataclass
class SessionSettings:
    enable_translation:bool
    enable_voice_clone:bool
    enable_lip_sync:bool
    enable_video:bool
    source_language:str
    target_language:str
    user_id:str|None
    closed: bool
    camera_rotation:int

    def __init__(self, video_request:VideoCaptureRequest|None = None, audio_request: TranslationRequest|None = None):
        # defaults
        self.enable_translation = True
        self.enable_voice_clone = True
        self.enable_lip_sync = True
        self.enable_video = True
        self.source_language = 'en'
        self.target_language = 'hin'
        self.user_id = None
        self.closed = False
        self.camera_rotation = 0

        #received on fist connection from mediassoup, prioritizing settings received in audio connection (less latency)
        if audio_request is not None:
            if audio_request.sourceLang is not None:
                self.source_language = audio_request.sourceLang

            if audio_request.targetLang is not None:
                self.target_language = audio_request.targetLang

            if audio_request.userId is not None:
                self.user_id = audio_request.userId

            if audio_request.cameraRotation is not None:
                self.camera_rotation = audio_request.cameraRotation

        elif video_request is not None:
            pass # For now, no settings are received by the video connection