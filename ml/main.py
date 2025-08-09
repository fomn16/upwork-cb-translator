import os
import tempfile
import subprocess
from subprocess import Popen
import threading
import time
import time
import numpy as np
from scipy import signal
from fastapi import FastAPI
from pydantic import BaseModel
import wave
import uvicorn
import uuid

import numpy as np

from config.video_config import *
from config.translation_config import *
from config.connection_config import *

from lipsync_communication import *

seamlessm4t = 0
if ENABLE_TRANSLATION:
    seamless_streaming = 1
else:
    seamless_streaming = 0

# from lib.streaming_translator_utils import SAMPLE_RATE, StatelessBytesTranslator
# translator1 = StatelessBytesTranslator(tgt_lang="hin")  # Hindi output
# start the voice clone
from multiprocessing.connection import Client

def request_voice_clone(audio):
    address = ("localhost", 6000)
    conn = Client(address, authkey=b"secret_vc")
    conn.send(audio)
    result = conn.recv()
    conn.close()
    return result

if seamless_streaming == 1:  # %%

    from lib.seamless.seamless_streaming_utils import (
        OutputSegments,
        reset_states,
        get_audio_bytes,
        build_streaming_system,
        bytes_to_float32_mono_array,
    )

    from seamless_communication.streaming.agents.seamless_streaming_s2st import (
        SeamlessStreamingS2STJointVADAgent,
    )
    from simuleval.data.segments import SpeechSegment, TextSegment

    agent_class = SeamlessStreamingS2STJointVADAgent
    tgt_lang = "hin"

    model_configs = dict(
        source_segment_size=320,
        device="cuda:0",
        dtype="fp16",
        min_starting_wait_w2vbert=192,
        decision_threshold=0.5,
        min_unit_chunk_size=50,
        no_early_stop=True,
        max_len_a=0,
        max_len_b=100,
        task="s2st",
        tgt_lang=tgt_lang,
        block_ngrams=True,
        detokenize_only=True,
    )

    system = build_streaming_system(model_configs, agent_class)
    print("✅ System ready.")


import pickle
def request_lipsync_in_worker(frame_buffer, audio_bytes, output_path):
    address = ('localhost', 6006)
    authkey = b'secret'

    with Client(address, authkey=authkey) as conn:
        data = pickle.dumps((frame_buffer, audio_bytes, output_path))
        conn.send_bytes(data)
        no_audio_path, final_path, error = conn.recv()
        if error:
            raise RuntimeError(f"Lip sync failed: {error}")
        return no_audio_path, final_path


import webrtcvad
import noisereduce as nr

vad = webrtcvad.Vad(1)  # Aggressiveness level

def float32_to_pcm16(audio_float):
    import numpy as np
    audio_int16 = np.clip(audio_float * 32767, -32768, 32767).astype(np.int16)
    return audio_int16.tobytes()

def is_voiced_float32(audio_float, sample_rate=16000, check_ms=300, frame_ms=30):
    pcm_bytes = float32_to_pcm16(audio_float)
    frame_size = int(sample_rate * frame_ms / 1000) * 2  # 2 bytes per sample
    max_bytes_to_check = int(sample_rate * check_ms / 1000) * 2

    for i in range(0, min(len(pcm_bytes), max_bytes_to_check), frame_size):
        frame = pcm_bytes[i:i + frame_size]
        if len(frame) < frame_size:
            break
        if vad.is_speech(frame, sample_rate):
            return True
    return False

def timed_is_voiced(float_audio, sample_rate=16000):
    start_time = time.perf_counter()
    result = is_voiced_float32(float_audio, sample_rate)
    elapsed_ms = (time.perf_counter() - start_time) * 1000  # milliseconds
    return result, elapsed_ms

def process_translation_chunk(
    audio_chunk: bytes,
    target_lang: str,
    system,
    system_states,
    output_queue,
    voice_clone_enabled: bool = False,
    request_voice_clone=None,
    tensor_to_bytes=None,
    resample_audio=None,
    save_to_wav=None,
    input_sr: int = 48000,
    target_sr: int = 16000,
    video_frames_storage=None,
    session_id=None
):
    # Convert bytes to float32 mono
    float_audio = bytes_to_float32_mono_array(audio_chunk, input_sr=input_sr, target_sr=target_sr)

    # Optional: Noise reduction
    clean_chunk = nr.reduce_noise(y=float_audio, sr=target_sr)

    # Handle stereo manually if upstream bytes_to_float32_mono_array doesn't already do it
    if clean_chunk.ndim == 2:
        clean_chunk = clean_chunk.mean(axis=0)

    # Clip extremely large chunks (safety for Seamless model)
    MAX_SAMPLES = target_sr * 2  # e.g., 2 seconds max
    if clean_chunk.shape[-1] > MAX_SAMPLES:
        clean_chunk = clean_chunk[-MAX_SAMPLES:]

    # Ensure valid float32 values
    clean_chunk = np.nan_to_num(clean_chunk).astype(np.float32)

    # Run VAD
    is_speech, vad_latency_ms = timed_is_voiced(clean_chunk, sample_rate=target_sr)
    print(f"VAD latency: {vad_latency_ms:.4f} s")

    if is_speech:
        try:
            input_segment = SpeechSegment(content=clean_chunk, sample_rate=target_sr)
            input_segment.tgt_lang = target_lang

            output_segments = OutputSegments(system.pushpop(input_segment, system_states))

            for seg in output_segments.segments:
                if isinstance(seg, SpeechSegment) and seg.sample_rate > 1:
                    print("✅ audio_segment")

                    if voice_clone_enabled:
                        assert request_voice_clone and tensor_to_bytes and resample_audio, \
                            "Voice cloning functions must be provided."

                        clone_tensor = request_voice_clone(seg.content)
                        cloned_audio_bytes = tensor_to_bytes(clone_tensor)
                        translated_audio_bytes = resample_audio(cloned_audio_bytes, 22050, 48000)

                        if save_to_wav:
                            save_to_wav(translated_audio_bytes)
                    else:
                        translated_audio_bytes = get_audio_bytes(seg.content, seg.sample_rate)

                        # Optional: Lip sync if frames exist
                        video_frames = video_frames_storage.pop(session_id, None)
                        lip_syn_enabled = False
                        if lip_syn_enabled and video_frames:
                            print(f"Lip sync started on {len(video_frames)} frames")
                            start_time = time.time()
                            _, final_vid = request_lipsync_in_worker(
                                video_frames,
                                translated_audio_bytes,
                                f"output_{time.time()}.mp4"
                            )
                            print(f"🕒 Lip sync time: {(time.time() - start_time):.4f}s")

                    output_queue.enqueue(translated_audio_bytes)

                elif isinstance(seg, TextSegment):
                    print(f"📝 Translated text: {seg.content}")

            # Handle utterance end
            if output_segments.finished:
                time.sleep(0.3)
                print("⏹️ Utterance ended. Resetting...")
                reset_states(system, system_states)

        except Exception as e:
            print(f"❗ Error during pushpop: {str(e)}. Resetting system state.")
            reset_states(system, system_states)
    else:
        print("🔇 No speech detected. Skipping...")
        reset_states(system, system_states)

# ----------------- Utilities ----------------- #
# Saves the bytes to a wav file on disk for debugging
def save_to_wav(audio_bytes: bytes, sample_rate=48000, num_channels=2, sample_width=2):
    os.makedirs("recordings", exist_ok=True)
    filename = f"recordings/output_{int(time.time() * 1000)}.wav"
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    print(f"💾 Saved audio segment to {filename}")


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
            "48000",
            "-ac",
            "2",
            "-f",
            "s16le",  # raw PCM
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=SAMPLE_READ_SIZE
    )

# Creates pipe that writes to the destination RTP endpoint
def run_ffmpeg_output(target_ip: str, target_port: int, payload_type: int, ssrc: int):
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-f",
        "s16le",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-i",
        "pipe:0",
        "-c:a",
        "libopus",
        "-application", "lowdelay",   # low-latency Opus mode
        "-frame_duration", "20",      # 20 ms frames (try 10 for even lower latency)
        "-packet_loss", "0",          # no extra buffering for PLC
        "-b:a", "64k",                 # bitrate (adjust as needed)
        "-payload_type",
        str(payload_type),
        "-ssrc",
        str(ssrc),
        "-f",
        "rtp",
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


# Resamples and converts mono to stereo
def resample_audio(audio_bytes, original_sr=16000, target_sr=48000):
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
    new_length = int(len(audio_data) * target_sr / original_sr)
    resampled = signal.resample(audio_data, new_length)
    resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
    stereo_data = np.column_stack((resampled, resampled)).flatten()
    return stereo_data.tobytes()


# Converts a numpy tensor to raw PCM bytes
def tensor_to_bytes(translated_wav):
    audio_np = np.clip(np.array(translated_wav, dtype=np.float32), -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    return audio_int16.tobytes()


# Function that reads from the input pipe, processes audio, and enqueues to output
def pump_audio(
    ff_in: Popen,
    ff_out: Popen,
    segment_size: int,
    sample_rate: int,
    sdp_path: str,
    system,
    system_states,
    target_lang,
    session_id: str,
):
    global lipsync_translated_audio_socket, lipsync_raw_audio_socket
    try:
        while True:
            seg = ff_in.stdout.read(segment_size)
            if not seg:
                print("empty chunk, stopping")
                break
            lipsync_translated_audio_socket.send(session_id, seg)
            lipsync_raw_audio_socket.send(session_id, seg)

            '''buf.extend(chunk)  # O(1) append

            while len(buf) >= segment_size:
                seg = bytes(buf[:segment_size])  # make immutable for sending
                del buf[:segment_size]           # remove from front in O(1)
                #focusing on setting up the lipsync pipeline
                if seamless_streaming == 1:
                    process_translation_chunk(
                        seg,
                        target_lang=target_lang,
                        system=system,
                        system_states=system_states,
                        output_queue=output_queue,
                        voice_clone_enabled=False,
                        request_voice_clone=request_voice_clone,
                        tensor_to_bytes=tensor_to_bytes,
                        resample_audio=resample_audio,
                        save_to_wav=save_to_wav,
                        video_frames_storage=video_frames_storage
                    )
                    #################################################
                else:
                    video_frames = video_frames_storage.pop(session_id, None)
                    print(
                        f"🟩 Processing video and audio segment for session {session_id}... {chunk_count}"
                        + (
                            f", video frames: {len(video_frames)}"
                            if video_frames is not None
                            else ", no video frames found."
                        )
                    )
                    output_queue.enqueue(seg)'''
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

class TranslationRequest(BaseModel):
    payloadType: int
    codec: str
    clockRate: int
    channels: int
    rtpPort: int
    outputPort: int
    ssrc: int
    targetLang: str
    sessionId: str

@app.post("/translation/initiate")
async def initiate_translation(data: TranslationRequest):
    global system, audio_out_pipes
    print("📥 Received translation initiation:", data.dict())
    sample_rate = data.clockRate

    if ENABLE_TRANSLATION:
        system_states = system.build_states()
    else:
        system_states = None
        system = None

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

    segment_size = int(sample_rate * 2 * 2 * 0.02) #20 ms segments

    # target_lang = data.targetLang
    target_lang = "eng"
    # manually enter the language code here
    # Create thread to process audio
    threading.Thread(
        target=pump_audio,
        args=(
            ff_in,
            ff_out,
            segment_size,
            sample_rate,
            sdp_path,
            system,
            system_states,
            target_lang,
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

class VideoCaptureRequest(BaseModel):
    payloadType: int
    codec: str
    clockRate: int
    rtpPort: int
    outputPort: int
    sessionId: str
    ssrc: int

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
