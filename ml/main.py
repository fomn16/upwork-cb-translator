import os
import tempfile
import subprocess
from subprocess import Popen
import threading
import numpy as np
from fastapi import FastAPI
import uvicorn
import uuid
import torch
import torchaudio

from config.video_config import *
from config.translation_config import *
from config.connection_config import *
from config.audio_config import *

from socket_communication import *

from lib.communication.mediassoup_request import *

# Initializes the file used to read input from the network
def write_sdp_file(payload_type, codec_name, clock_rate, channels, rtp_port):
    sdp_content = f"""v=0
    o=- 0 0 IN IP4 0.0.0.0
    s=Mediasoup Audio
    c=IN IP4 0.0.0.0
    t=0 0
    m=audio {rtp_port} RTP/AVP {payload_type}
    a=rtpmap:{payload_type} {codec_name}/{clock_rate}/{channels}
    a=recvonly
    """.strip()

    tmp_dir = tempfile.gettempdir()
    sdp_path = os.path.join(tmp_dir, f"audio_{int(os.getpid())}.sdp")

    with open(sdp_path, "w") as f:
        f.write(sdp_content)

    return sdp_path

# Creates pipe that reads the data from the provided SDP path
def run_ffmpeg_input(sdp_path):
    return subprocess.Popen(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "info",
            "-protocol_whitelist",
            "file,udp,rtp",
            "-fflags", "nobuffer",       # disable buffering
            "-flags", "low_delay",       # low-latency decoding
            "-probesize", "32",          # minimal probing
            "-analyzeduration", "0",     # no extra analysis delay
            "-flush_packets", "1",       # flush decoded audio immediately
            "-f",
            "sdp",
            "-i",
            sdp_path,
            "-c:a",
            "pcm_s16le",
            "-ar",
            str(EXTERNAL_SAMPLERATE),
            "-ac",
            "2",
            "-f",
            "s16le",  # raw PCM
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=EXTERNAL_N_AUDIO_CHUNK_SAMPLES
    )

# Creates pipe that writes to the destination RTP endpoint
def run_ffmpeg_output(target_ip: str, target_port: int, payload_type: int, ssrc: int):
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-f", "s16le",
        "-ar", "16000",          # input sample rate
        "-ac", "1",              # input channels (mono)
        "-i", "pipe:0",
        "-ar", str(EXTERNAL_SAMPLERATE),  # output sample rate
        "-ac", "2",              # output channels (stereo)
        "-c:a", "libopus",
        "-application", "lowdelay",
        "-frame_duration", "20",
        "-packet_loss", "0",
        "-b:a", "64k",
        "-payload_type", str(payload_type),
        "-ssrc", str(ssrc),
        "-f", "rtp",
        f"rtp://{target_ip}:{target_port}?pkt_size=1200&buffer_size=65536",
    ]
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=0
    )

# Logs FFmpeg errors
def print_ffmpeg_logs(proc, label):
    for line in iter(proc.stderr.readline, b""):
        text = line.decode(errors="ignore").strip()
        if "error" in text.lower():
            print(f"{label}: {text}")

def to_internal_format(
    seg: bytes,
    orig_rate: int,
    target_rate: int = 16000,
    num_ch: int = 2
) -> bytes:
    """
    Convert raw PCM16-LE bytes from FFmpeg to PCM16-LE mono at target_rate

    Args:
        seg: Raw PCM16-LE bytes (interleaved channels).
        orig_rate: Original sample rate (EXTERNAL_SAMPLERATE from FFmpeg).
        target_rate: Desired output sample rate (default 16kHz).
        num_ch: Number of channels in seg (default 2 for stereo).

    Returns:
        bytes: Processed PCM16-LE mono audio at target_rate.
    """
    # Convert bytes to torch tensor (int16)
    dtype_in = np.dtype('<i2')  # little-endian int16
    audio_np = np.frombuffer(seg, dtype=dtype_in)

    # Reshape to (channels, samples)
    audio_np = audio_np.reshape(-1, num_ch).T  # shape: (num_ch, num_samples)
    audio_tensor = torch.from_numpy(audio_np.astype(np.float32))

    # Normalize to [-1, 1]
    audio_tensor /= np.iinfo(np.int16).max

    # Downmix to mono if needed
    if num_ch > 1:
        audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

    # Resample if needed
    if orig_rate != target_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_rate, new_freq=target_rate
        )
        audio_tensor = resampler(audio_tensor)

    # Convert back to int16 PCM
    audio_tensor = (audio_tensor * np.iinfo(np.int16).max).clamp(
        min=np.iinfo(np.int16).min, max=np.iinfo(np.int16).max
    ).short()

    # Return as bytes (interleaved mono)
    return audio_tensor.squeeze(0).numpy().tobytes()

# Function that reads from the input pipe, processes audio, and enqueues to output
def pump_audio(
    ff_in: Popen,
    ff_out: Popen,
    segment_size: int,
    sdp_path: str,
    session_id: str,
):
    try:
        while True:
            seg = ff_in.stdout.read(segment_size)
            if not seg:
                print("empty chunk, stopping")
                break
            seg_converted = to_internal_format(seg, EXTERNAL_SAMPLERATE)
            #lipsync_raw_audio_socket.send(session_id, seg)
            translate_socket.send(session_id, seg_converted)
    finally:
        ff_in.stdout.close()
        ff_out.stdin.close()
        ff_in.wait()
        ff_out.wait()
        try:
            os.remove(sdp_path)
        except OSError:
            pass

# ----------------- FastAPI Server ----------------- #
app = FastAPI()

@app.post("/translation/initiate")
async def initiate_translation(data: TranslationRequest):
    global system, audio_out_pipes
    print("📥 Received translation initiation:", data.dict())

    # sends initial settings to all enviromnents
    send_settings(data.sessionId, SessionSettings(audio_request=data))

    sample_rate = data.clockRate

    # Sets up the read file from the rtp port provided by the client
    sdp_path = write_sdp_file(
        payload_type=data.payloadType,
        codec_name=data.codec,
        clock_rate=sample_rate,
        channels=data.channels,
        rtp_port=data.rtpPort,
    )

    ff_in = run_ffmpeg_input(sdp_path)
    ff_out = run_ffmpeg_output(MEDIASERVER_IP, data.outputPort, data.payloadType, data.ssrc)
    audio_out_pipes[data.sessionId] = ff_out

    # Create threads that log errors encountered by FFmpeg
    threading.Thread(
        target=print_ffmpeg_logs, args=(ff_in, "FFmpeg-IN"), daemon=True
    ).start()
    threading.Thread(
        target=print_ffmpeg_logs, args=(ff_out, "FFmpeg-OUT"), daemon=True
    ).start()
    
    # manually enter the language code here
    # Create thread to process audio
    threading.Thread(
        target=pump_audio,
        args=(
            ff_in,
            ff_out,
            EXTERNAL_N_AUDIO_CHUNK_BYTES,
            sdp_path,
            data.sessionId,
        ),
        daemon=True,
    ).start()

    return {"status": "Translation pipeline started"}


def write_video_sdp_file(payload_type, codec_name, clock_rate, rtp_port):
    sdp = (
        "v=0\n"
        "o=- 0 0 IN IP4 0.0.0.0\n"
        "s=Mediasoup Video\n"
        "c=IN IP4 0.0.0.0\n"
        "t=0 0\n"
        f"m=video {rtp_port} RTP/AVP {payload_type}\n"
        f"a=rtpmap:{payload_type} {codec_name}/{clock_rate}\n"
        "a=recvonly\n"
        "a=rtcp-mux\n"
    )
    fn = f"video_{uuid.uuid4().hex}.sdp"
    path = os.path.join(tempfile.gettempdir(), fn)
    with open(path, "w") as f:
        f.write(sdp)
    return path


def run_ffmpeg_video_pipe(sdp_path):
    print(f"Running FFmpeg with SDP path: {sdp_path}")
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "info",
        "-protocol_whitelist",
        "file,udp,rtp",
        "-flags", "low_delay",          # low-latency decoding
        "-probesize", "32",              # minimal probing
        "-analyzeduration", "0",         # no extra analysis delay
        "-flush_packets", "1",           # flush packets ASAP
        "-f",
        "sdp",
        "-i",
        sdp_path,
        "-an",  # no audio
        "-r", str(FRAME_RATE),  # Cap the frame rate (e.g., 15 fps)
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
        "pipe:1",
    ]
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=FRAME_WIDTH*FRAME_HEIGHT*3
    )

def run_ffmpeg_video_output(
    target_ip,
    target_port,
    payload_type: int,
    ssrc: int,
    bitrate=2500,  # kbps
):
    # FFmpeg command: read raw BGR frames from stdin, encode VP8, send RTP/RTCP
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-re",
        # Raw video from STDIN
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",  # input frame format (BGR uint8)
        "-s:v",
        f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
        "-r",
        str(FRAME_RATE),
        "-i",
        "pipe:0",
        "-an",
        # Encode as VP8
        "-c:v",
        "libvpx",
        # Convert to yuv420p for VP8 encoder compatibility
        "-pix_fmt",
        "yuv420p",

        # Low-latency libvpx settings
        "-deadline",
        "realtime",       # realtime mode
        "-cpu-used",
        "8",              # higher = faster, lower quality; try 5..8
        "-lag-in-frames",
        "0",              # no lookahead
        "-rc_lookahead",
        "0",              # no RC lookahead
        
        # Rate control
        "-b:v",
        f"{bitrate}k",
        "-maxrate",
        f"{bitrate}k",
        "-bufsize",
        f"{max(int(bitrate * 0.5), 100)}k",
        "-g",
        str(FRAME_RATE * 2), # send keyframe every 2 seconds
        "-threads",
        "4",
        # Output as RTP (video only)
        "-f",
        "rtp",
        "-payload_type",
        str(payload_type),
        "-ssrc",
        str(ssrc),
        f"rtp://{target_ip}:{target_port}"
        f"?localrtcpport=0"
        f"&rtcpport={target_port+1}"
        f"&pkt_size=1200"
        f"&buffer_size=65536"
    ]

    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

def foward_frames_for_processing(
    in_proc: Popen,
    out_proc: Popen,
    frame_width: int,
    frame_height: int,
    session_id: str
):
    global lipsync_video_socket
    frame_size = frame_width * frame_height * 3  # BGR24
    try:
        while True:
            raw_frame = in_proc.stdout.read(frame_size)
            if not raw_frame:
                print("📤 FFmpeg pipe ended")
                break
            lipsync_video_socket.send(session_id, raw_frame)
    except Exception as e:
        print(f"⚠️ Error in foward_frames_for_processing: {e}")
    finally:
        print(f"⚠️ closing FFmpeg video pipes for session {session_id}")
        try:
            in_proc.stdout.close()
            in_proc.stderr.close()
            in_proc.terminate()
            in_proc.wait(timeout=5)

            out_proc.stdout.close()
            out_proc.stderr.close()
            out_proc.terminate()
            out_proc.wait(timeout=5)
        except:
            pass
        print("✅ Frame fowarder stopped")

@app.post("/video/initiate")
async def initiate_video_capture(data: VideoCaptureRequest):
    global video_out_pipes
    print("📥 Received video capture initiation:", data.dict())

    sdp_path = write_video_sdp_file(
        payload_type=data.payloadType,
        codec_name=data.codec,
        clock_rate=data.clockRate,
        rtp_port=data.rtpPort,
    )

    ffmpeg_in = run_ffmpeg_video_pipe(sdp_path)

    print(f"🔄️ FFmpeg process started with PID {ffmpeg_in.pid}")

    ffmpeg_out = run_ffmpeg_video_output(
        MEDIASERVER_IP,  # Mediasoup plain transport IP
        data.outputPort,  # Mediasoup plain transport video port
        data.payloadType,
        data.ssrc
    )
    video_out_pipes[data.sessionId] = ffmpeg_out


    threading.Thread(
        target=print_ffmpeg_logs, args=(ffmpeg_in, "FFmpeg-video-in"), daemon=True
    ).start()

    threading.Thread(
        target=print_ffmpeg_logs, args=(ffmpeg_out, "FFmpeg-video-out"), daemon=True
    ).start()

    threading.Thread(
        target=foward_frames_for_processing,
        args=(
            ffmpeg_in,
            ffmpeg_out,
            FRAME_WIDTH,
            FRAME_HEIGHT,
            data.sessionId
        ),
        daemon=True,
    ).start()

    return {"status": "Frame-based video capture started."}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=MAIN_ENDPOINT_PORT, reload=False)
# %%
