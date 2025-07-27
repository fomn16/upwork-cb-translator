import subprocess
from subprocess import Popen
import os
import tempfile
import time

video_frames_storage = {}
def send_frames_to_mediasoup(frame_generator, target_ip, target_port, payload_type: int, ssrc: int, width=640, height=480, fps=15):
    cmd = [
        "ffmpeg",
        "-f",
        "rawvideo",  # Input format: raw video
        "-pix_fmt",
        "yuv420p",  # Pixel format for VP8
        "-s",
        f"{width}x{height}",  # Resolution (adjust as needed)
        "-r",
        f"{fps}",  # Frame rate (adjust as needed)
        "-i",
        "pipe:0",  # Input from stdin
        "-c:v",
        "libvpx",  # VP8 codec
        "-b:v",
        "1M",  # Bitrate (adjust as needed)
        "-payload_type",
        str(payload_type),
        "-ssrc",
        str(ssrc),
        "-f",
        "rtp",
        f"rtp://{target_ip}:{target_port}",
    ]
    proc =  subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    frame_interval = 1.0 / fps
    try:
        for frame in frame_generator:
            flat_frame = frame.tobytes()  # Convert 3D array to flat byte array
            proc.stdin.write(flat_frame)  # Write the flat frame to FFmpeg's stdin
            proc.stdin.flush()
            time.sleep(frame_interval)  # basic timing to avoid too much jitter on mediasoup
    except BrokenPipeError:
        print("⚠️ FFmpeg pipe closed, stopping sending.")
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")
    finally:
        proc.stdin.close()
        proc.wait()
        print("✅ FFmpeg process terminated.")



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