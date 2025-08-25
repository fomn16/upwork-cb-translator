from pydantic import BaseModel

class VideoCaptureRequest(BaseModel):
    payloadType: int
    codec: str
    clockRate: int
    rtpPort: int
    outputPort: int
    sessionId: str
    ssrc: int

class TranslationRequest(BaseModel):
    payloadType: int
    codec: str
    clockRate: int
    channels: int
    rtpPort: int
    outputPort: int
    ssrc: int
    sourceLang: str
    targetLang: str
    sessionId: str
    userId: str
    cameraRotation:int