import subprocess
import sys
import numpy as np
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
    fps=15,
    bitrate=2500,  # kbps
):
    # FFmpeg command: read raw BGR frames from stdin, encode VP8, send RTP/RTCP
    cmd = [
        "ffmpeg",
        "-re",
        # Raw video from STDIN
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",  # input frame format (BGR uint8)
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        # Encode as VP8
        "-c:v",
        "libvpx",
        # Convert to yuv420p for VP8 encoder compatibility
        "-pix_fmt",
        "yuv420p",
        # Rate control
        "-b:v",
        f"{bitrate}k",
        "-maxrate",
        f"{bitrate}k",
        "-bufsize",
        f"{2*bitrate}k",
        "-g",
        str(fps * 2), # send keyframe every 2 seconds
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
        f"?localrtcpport=0&rtcpport={target_port+1}&pkt_size=1200",
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=10**7,
    )

    first_frame = True
    try:
        for frame in frame_generator:
            # Validate first frame
            if(first_frame):
                first_frame = False
                if (
                    not isinstance(frame, np.ndarray)
                    or frame.dtype != np.uint8
                    or frame.ndim != 3
                    or frame.shape[0] != height
                    or frame.shape[1] != width
                    or frame.shape[2] != 3
                ):
                    raise ValueError(
                        f"Frame must be numpy uint8 of shape ({height},{width},3) BGR."
                    )
            # Write raw bytes
            proc.stdin.write(frame.tobytes(order="C"))

    except BrokenPipeError:
        print("FFmpeg pipe closed (BrokenPipe).")
    finally:
        if proc.stdin:
            try:
                proc.stdin.close()
            except Exception:
                pass
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            proc.wait()

        if proc.returncode not in (0, None):
            try:
                err = proc.stderr.read().decode(errors="ignore")
                sys.stderr.write("\n[ffmpeg stderr]\n" + err[-4000:] + "\n")
            except Exception:
                pass

def frame_generator(session_id):
    while True:
        frames = video_frames_storage.get(session_id, [])
        if frames:
            frame = frames.pop(0)
            # print(f"📤 Yielding frame for session {session_id}")
            yield frame
        else:
            time.sleep(0.01)