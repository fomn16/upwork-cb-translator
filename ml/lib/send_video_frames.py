import subprocess
from subprocess import Popen
import os
import tempfile
import time

video_frames_storage = {}
def send_frames_to_mediasoup(
    frame_generator,
    target_ip,
    target_port,
    payload_type: int,
    ssrc: int,
    width=640,
    height=480,
    fps=30,
    bitrate=2500, #bitrate/1000
):
    cmd = [
        "ffmpeg",
        "-re",
        "-f", "lavfi",
        "-i",
        f"testsrc=size=1280x720:rate={fps}", #send test pattern
        "-c:v", "libvpx",# send with VP8 encoding
        "-pix_fmt", "yuv420p",#send with yuv420p
        "-b:v", str(bitrate)+"k",# bitrate for send
        "-maxrate", str(bitrate)+"k", # max rate for send
        "-bufsize", str(2*bitrate)+"k",
        "-g", str(fps*2),  # keyframe every 2s at 30fps
        "-threads", "4",
        "-f", "rtp", #output as RTP
        "-payload_type", str(payload_type),
        "-ssrc", str(ssrc),
        f"rtp://{target_ip}:{target_port}?localrtcpport=0&rtcpport={target_port+1}&pkt_size=1200",
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )

    try:
        for frame in frame_generator:
            # frame must be HxWx3, dtype=uint8, BGR order
            proc.stdin.write(frame.tobytes())
            proc.stdin.flush()
    except BrokenPipeError:
        print("⚠️ FFmpeg video pipe closed, stopping.")
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait(timeout=5)



def frame_generator(session_id):
    while True:
        frames = video_frames_storage.get(session_id, [])
        if frames:
            frame = frames.pop(0)
            # print(f"📤 Yielding frame for session {session_id}")
            yield frame
        # else:
        #     # print(f"⏳ No frames for session {session_id}")
        #     # time.sleep(0.01)



def stream_video(video_path, target_ip, target_port, width=640, height=480, fps=15):
    print(f"Starting to stream video {video_path} to Mediasoup at {target_ip}:{target_port}...")
    cmd = [
        "ffmpeg",
        "-re",  # Read input at native frame rate
        "-i", video_path,  # Input video file
        "-c:v", "libvpx",  # Use VP8 to match Mediasoup router
        "-b:v", "1M",
        "-s", f"{width}x{height}",  # Scale to specified resolution
        "-r", str(fps),  # Set frame rate
        "-deadline", "realtime",
        "-cpu-used", "5",
        "-f", "rtp",
        f"rtp://{target_ip}:{25001}"
    ]
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)
    try:
        proc.wait()
    except Exception as e:
        print(f"⚠️ Error streaming video: {e}")
    finally:
        proc.terminate()
        print("✅ Video streaming stopped.")