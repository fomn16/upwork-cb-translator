from pydantic import BaseModel

class VideoCaptureRequest(BaseModel):
    payloadType: int = None
    codec: str = None
    clockRate: int = None
    rtpPort: int = None
    outputPort: int = None
    sessionId: str = None
    ssrc: int = None

class TranslationRequest(BaseModel):
    payloadType: int = None
    codec: str = None
    clockRate: int = None
    channels: int = None
    rtpPort: int = None
    outputPort: int = None
    ssrc: int = None
    sourceLang: str = None
    targetLang: str = None
    sessionId: str = None
    userId: str = None
    cameraRotation:int = None