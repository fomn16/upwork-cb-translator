import os
import numpy as np
from tqdm import tqdm
import lib.audio as audio
import argparse
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import threading
from queue import Queue
from threading import Thread
import torch
from torch.cuda.amp import autocast
import torchaudio

from pipelines.lipsync.wav2lip_model.wav2lip import Wav2Lip
from config.lipsync_config import *
from config.video_config import *

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='Inference code for lip-syncing videos using Wav2Lip models')
parser.add_argument('--checkpoint_path', type=str, default='/Users/apple/Downloads/Lip_Sync_Wav2Lip/checkpoints/wav2lip_Chinese.pth')
parser.add_argument('--face', type=str, default='/Users/apple/Downloads/Lip_Sync_Wav2Lip/demo.mp4')
parser.add_argument('--audio', type=str, default='/Users/apple/Downloads/Lip_Sync_Wav2Lip/demo.wav')
parser.add_argument('--outfile', type=str, default='/Users/apple/Downloads/Lip_Sync_Wav2Lip/result20.mp4')
parser.add_argument('--static', type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--fps', type=float, default=25)
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0])
parser.add_argument('--face_det_batch_size', type=int, default=16)
parser.add_argument('--wav2lip_batch_size', type=int, default=64)
parser.add_argument('--resize_factor', type=int, default=1)
parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1])
parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1])
parser.add_argument('--rotate', default=False, action='store_true')
parser.add_argument('--nosmooth', default=False, action='store_true')
args = parser.parse_args()
args.img_size = 96

if os.path.isfile(args.face) and args.face.split('.')[-1] in ['jpg', 'png', 'jpeg']:
    args.static = True

from ultralytics.utils import LOGGER
LOGGER.setLevel("ERROR")

def load_model(path):
    start = time.time()
    model = Wav2Lip()
    print(f"Loading checkpoint from: {path}")
    checkpoint = torch.load(path, map_location=device)
    s = checkpoint["state_dict"]
    new_s = {k.replace("module.", ""): v for k, v in s.items()}
    model.load_state_dict(new_s)
    model = model.to(device)
    model.eval()

    model = torch.compile(model)

    print(f"[load_model] Loaded in {time.time() - start:.2f}s")
    print("starting warmup")
    warmup_done = False
    if os.path.exists(CACHE_MEL) and os.path.exists(CACHE_FACE):
        warm_mel_full = torch.load(CACHE_MEL, map_location=device)
        warm_face_full = torch.load(CACHE_FACE, map_location=device)

        if warm_mel_full.shape[0] >= 15 and warm_face_full.shape[0] >= 15:
            print(f"[warmup] Using cached batch {tuple(warm_mel_full.shape)}, {tuple(warm_face_full.shape)}")

            with torch.inference_mode():
                for i in range(10):  # repeat 10 rounds
                    for bs in range(1, 16):  # 1..15
                        warm_mel = warm_mel_full[:bs].to(device)
                        warm_face = warm_face_full[:bs].to(device)
                        t = time.perf_counter()
                        _ = model(warm_mel, warm_face)
                        print(f"[warmup cached] round {i+1}, bs={bs}/15 complete in {time.perf_counter()-t:.4f}s")

                if device.startswith("cuda"):
                    torch.cuda.synchronize()

            print("[warmup] Done with cached 1–15 batch warmup")
            warmup_done = True
        else:
            print("[warmup] Cached batch found but not long enough (need at least 15). Skipping.")
    if not warmup_done:
        with torch.inference_mode():
            for i in range (10):
                for bs in range(1, 16):  # 1..15
                    dummy_mel = torch.randn(bs, 1, 80, 16, device=device, dtype=torch.float32).to('cuda:0')
                    dummy_face = torch.randn(bs, 6, 96, 96, device=device, dtype=torch.float32).to('cuda:0')
                    t = time.perf_counter()
                    _ = model(dummy_mel, dummy_face)
                    print(f"warmup {bs}/15 complete in {time.perf_counter()-t}s")
            torch.cuda.synchronize()

    print("[load_model] Warmup complete (compiled model ready for 1–15 batch sizes).")

    return model

model = load_model("wav2lip_gan.pth")

def clamp_coords(coords, frame_shape):
    """Clamp face coordinates to be within frame boundaries."""
    if coords is None:
        return None
    x1, y1, x2, y2 = map(int, coords)
    h, w = frame_shape[:2]
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)

def crop_and_resize_face(frame, coords, size):
    """Crop and resize the face region from the frame."""
    if coords is None:
        return None
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        return None
    face = frame[y1:y2, x1:x2]
    return cv2.resize(face, (size, size))

def mask_half_face(img_batch):
    """Mask the right half of each face image and concatenate with original."""
    img_masked = img_batch.copy()
    img_masked[:, img_batch.shape[2] // 2 :, :, :] = 0
    return np.concatenate((img_masked, img_batch), axis=3) / 255.0

def threaded_generator(generator, max_prefetch=2):
    """Run a generator in a background thread for prefetching."""
    queue = Queue(max_prefetch)

    def producer():
        for item in generator:
            queue.put(item)
        queue.put(None)  # Sentinel to signal end

    Thread(target=producer, daemon=True).start()

    while True:
        item = queue.get()
        if item is None:
            break
        yield item

def datagen(frames, mels, face_det_results, prefetch=True):
    """
    Generates batches of masked face images (pinned CPU tensors) and mel spectrograms (GPU) for inference.
    """
    def _generator():
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        for i, mel in enumerate(mels):
            idx = 0 if args.static else i % len(frames)
            frame = frames[idx]
            coords = clamp_coords(face_det_results[idx], frame.shape)

            if coords is None:
                frame_batch.append(frame)
                coords_batch.append(None)
            else:
                face = crop_and_resize_face(frame, coords, args.img_size)
                if face is None:
                    frame_batch.append(frame)
                    coords_batch.append(None)
                else:
                    img_batch.append(face)
                    mel_batch.append(mel)
                    frame_batch.append(frame)
                    coords_batch.append(coords)

            # ✅ yield as soon as we have a full batch
            if len(img_batch) >= len(mels):
                img_batch_np = np.asarray(img_batch, dtype=np.float32)
                img_masked = mask_half_face(img_batch_np)
                img_tensor = torch.from_numpy(
                    np.transpose(img_masked, (0, 3, 1, 2))
                ).float().pin_memory()

                yield img_tensor, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        # ✅ flush leftovers (if any)
        if img_batch:
            img_batch_np = np.asarray(img_batch, dtype=np.float32)
            img_masked = mask_half_face(img_batch_np)
            img_tensor = torch.from_numpy(
                np.transpose(img_masked, (0, 3, 1, 2))
            ).float().pin_memory()

            yield img_tensor, mel_batch, frame_batch, coords_batch

    return threaded_generator(_generator()) if prefetch else _generator()

_mel_filterbank = None
def get_mel_filterbank(n_mels=80, n_fft=1024, hop_length=256, sample_rate=16000):
    global _mel_filterbank
    if _mel_filterbank is None:
        mel_fb = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=1.0,
            norm="slaney",
            mel_scale="slaney"
        ).to(device)
        _mel_filterbank = mel_fb
    return _mel_filterbank

def preprocess_audio(audio_bytes: bytes) -> torch.Tensor:
    """
    Convert raw audio bytes to a mel spectrogram on GPU.
    Returns a torch.Tensor [n_mels, time] on GPU.
    """
    waveform_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    waveform_float32 = torch.from_numpy(waveform_int16.astype(np.float32) / 32768.0).to(device)

    # Ensure shape [1, time]
    if waveform_float32.dim() == 1:
        waveform_float32 = waveform_float32.unsqueeze(0)

    mel_transform = get_mel_filterbank()
    mel = mel_transform(waveform_float32)  # [1, n_mels, time]
    mel = torch.log(torch.clamp(mel, min=1e-5))  # log-mel
    return mel.squeeze(0)  # [n_mels, time]

def create_mel_chunks(
    mel: torch.Tensor, num_video_frames: int, step_size: int = 16
) -> list[torch.Tensor]:
    """
    Create mel spectrogram chunks aligned with video frames.
    Ensures the number of chunks == number of video frames,
    allowing small overlaps if necessary.

    Args:
        mel (torch.Tensor): Mel spectrogram of shape (n_mels, T).
        num_video_frames (int): Number of video frames (e.g., 15 for 1s at 15fps).
        step_size (int): Number of mel frames per chunk (default=16).

    Returns:
        list[torch.Tensor]: List of mel chunks, one per video frame.
    """
    mel_chunks = []
    mel_len = mel.shape[1]

    # Pad mel if too short
    if mel_len < step_size:
        mel = torch.nn.functional.pad(mel, (0, step_size - mel_len))
        mel_len = mel.shape[1]

    # Spread mel frames evenly across video frames
    mel_idx_multiplier = mel_len / num_video_frames

    for i in range(num_video_frames):
        start_idx = int(round(i * mel_idx_multiplier))
        end_idx = start_idx + step_size

        if end_idx > mel_len:
            # If we run past the end, just take the last step_size frames
            chunk = mel[:, -step_size:]
        else:
            chunk = mel[:, start_idx:end_idx]

        mel_chunks.append(chunk)

    return mel_chunks

def match_frames_to_mels(frames: list, mel_chunks: list, tolerance: int = 3) -> list:
    """
    Ensures the number of frames matches the number of mel chunks.
    Pads with the last frame or truncates as needed.
    Logs a warning if mismatch is larger than tolerance.
    """
    num_frames = len(frames)
    num_mels = len(mel_chunks)
    diff = num_mels - num_frames

    if abs(diff) > tolerance:
        print(
            f"[warning] Large mismatch between frames ({num_frames}) "
            f"and mel chunks ({num_mels}). Adjusting..."
        )

    if diff > 0:
        # More mels than frames → pad with last frame
        frames += [frames[-1]] * diff
    elif diff < 0:
        # More frames than mels → truncate
        frames = frames[:num_mels]

    return frames

def run_inference(gen, mel_chunks: list, save_for_warmup = False) -> list:
    frames_output = []
    frames_written = 0

    batch_size = len(mel_chunks)
    if batch_size == 0:
        raise ValueError("No mel chunks available for inference.")

    print(f"[stream] Running inference with batch size {batch_size}")

    for img_batch, mel_batch, frames, coords in tqdm(gen, total=1):
        # img_batch is pinned CPU tensor → move to GPU
        img_batch = img_batch.to(device, non_blocking=True)

        # mel_batch is list of GPU tensors → stack
        mel_batch = torch.stack(mel_batch, dim=0).unsqueeze(1)  # [B, 1, 80, step_size]

        if (mel_batch.shape[0] == 15 and save_for_warmup
            and not (os.path.exists(CACHE_MEL) and os.path.exists(CACHE_FACE))):
            torch.save(mel_batch.cpu(), CACHE_MEL)
            torch.save(img_batch.cpu(), CACHE_FACE)
            print(f"[cache] Saved warmup inputs → mel: {tuple(mel_batch.shape)}, face: {tuple(img_batch.shape)}")

        with torch.inference_mode():
            pred = model(mel_batch, img_batch)

        if isinstance(pred, torch.Tensor):
            pred = pred.float().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        else:
            pred = pred.astype(np.float32).transpose(0, 2, 3, 1) * 255.0

        for p, f, c in zip(pred, frames, coords):
            if c is None:
                frames_output.append(f)
                continue

            x1, y1, x2, y2 = map(int, c)
            h, w = f.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                frames_output.append(f)
                continue

            try:
                p_resized = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = cv2.addWeighted(
                    f[y1:y2, x1:x2], 0.2, p_resized, 0.8, 0
                )
            except Exception as e:
                print(f"Error applying lipsync: {e}")

            frames_output.append(f)
            frames_written += 1

    print(f"[stream] Inference complete. Frames written: {frames_written}")
    return frames_output


def rotate_frame_and_coords(frame, coords, rotation: int):
    """
    Rotate a frame and its face detection coords clockwise.
    rotation: 0=0°, 1=90° CW, 2=180°, 3=270° CW
    """
    if rotation == 0:
        return frame, coords

    h, w = frame.shape[:2]

    if rotation == 1:  # 90° CW
        frame_rot = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if coords is not None:
            x1, y1, x2, y2 = coords
            coords_rot = (h - y2, x1, h - y1, x2)
        else:
            coords_rot = None

    elif rotation == 2:  # 180°
        frame_rot = cv2.rotate(frame, cv2.ROTATE_180)
        if coords is not None:
            x1, y1, x2, y2 = coords
            coords_rot = (w - x2, h - y2, w - x1, h - y1)
        else:
            coords_rot = None

    elif rotation == 3:  # 270° CW (== 90° CCW)
        frame_rot = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if coords is not None:
            x1, y1, x2, y2 = coords
            coords_rot = (y1, w - x2, y2, w - x1)
        else:
            coords_rot = None

    return frame_rot, coords_rot


def unrotate_frame(frame, rotation: int):
    """
    Undo clockwise rotation on a frame.
    rotation: 0=0°, 1=90° CW, 2=180°, 3=270° CW
    """
    if rotation == 0:
        return frame
    elif rotation == 1:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == 2:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 3:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

def adjust_length_to_match(seq, target_len):
    """
    Adjusts a list (frames or mel chunks) to match target_len.
    If seq is shorter, repeats elements at equal spacing.
    If seq is longer, removes elements at equal spacing.
    """
    current_len = len(seq)
    if current_len == target_len:
        return seq

    adjusted = []
    if current_len < target_len:
        # Need to duplicate some elements
        repeats = target_len - current_len
        # Spread duplicates evenly
        indices_to_duplicate = set(
            round(i * current_len / repeats) for i in range(repeats)
        )
        dup_count = 0
        for i, item in enumerate(seq):
            adjusted.append(item)
            if i in indices_to_duplicate:
                adjusted.append(item)
                dup_count += 1
        # If still short (rounding), pad with last element
        while len(adjusted) < target_len:
            adjusted.append(seq[-1])
    else:
        # Need to drop some elements
        drops = current_len - target_len
        indices_to_drop = set(
            round(i * current_len / drops) for i in range(drops)
        )
        for i, item in enumerate(seq):
            if i not in indices_to_drop:
                adjusted.append(item)

    return adjusted

def run_lipsync_from_frames(frame_buffer, audio_bytes, face_detect, rotation:int, session_id, save_for_warmup = False):
    if not frame_buffer or not audio_bytes:
        print("WARNING: Empty frames or audio. Skipping lipsync.")
        return frame_buffer

    print(f"[stream {session_id}] Received {len(frame_buffer)} frames and {len(audio_bytes)} samples")

    # 1. Audio preprocessing (GPU)
    audio_start = time.time()
    mel_gpu = preprocess_audio(audio_bytes)  # [n_mels, time] on GPU
    print(f"[stream {session_id}] mel_gpu = {len(mel_gpu)}")
    #print(f"[stream] Audio processed in {time.time() - audio_start:.2f}s")

    # 2. Mel chunk creation (GPU)
    mel_chunks = create_mel_chunks(mel_gpu, len(frame_buffer))
    print(f"[stream {session_id}] Mel chunks: {len(mel_chunks)}")

    # Skip if mel chunk too short for model
    if mel_chunks[0].shape[1] < 3:
        print(f"[stream {session_id}] Skipping lipsync: mel chunk too short")
        return frame_buffer

    if rotation != 0:
        rotated_frames, rotated_coords = [], []
        for f, c in zip(frame_buffer, face_detect):
            f_rot, c_rot = rotate_frame_and_coords(f, c, rotation)
            rotated_frames.append(f_rot)
            rotated_coords.append(c_rot)
        frame_buffer, face_detect = rotated_frames, rotated_coords

    # 3. Prepare generator (mel stays on GPU)
    gen = datagen(frame_buffer, mel_chunks, face_detect)

    # 4. Run inference (mixed precision)
    output_frames =  run_inference(gen, mel_chunks, save_for_warmup)
    print(f"[stream {session_id}] Returning {len(output_frames)} frames")
    if rotation != 0:
        output_frames = [unrotate_frame(f, rotation) for f in output_frames]

    # inserting original frames in same positions in which no face detection was found
    for i, detect in enumerate(face_detect):
        if detect == None:
            output_frames.insert(i,frame_buffer[i])

    # 5. Adjusting number of returned frames in case its different from the number of received frames
    if len(output_frames) != len(frame_buffer):
        print(
            f"[stream {session_id}] Adjusting output frames "
            f"({len(output_frames)} → {len(frame_buffer)})"
        )
        output_frames = adjust_length_to_match(output_frames, len(frame_buffer))

    return output_frames

print("Loading MediaPipe Face Detector (GPU)...")

# Path to the downloaded .task model
MODEL_PATH = "blaze_face_short_range.tflite" #https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/index#models

# Create GPU-enabled MediaPipe Face Detector
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH,
    delegate=python.BaseOptions.Delegate.GPU  # Force GPU usage
)

options = vision.FaceDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    min_detection_confidence=0.5
)

face_detector_instance = vision.FaceDetector.create_from_options(options)
face_detector_lock = threading.Lock()

print("MediaPipe Face Detector (GPU) loaded.")

# Warm-up
dummy_img = np.random.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy_img)
_ = face_detector_instance.detect(mp_image)
print("Warm-up done.")

def face_detect_once(frame):
    """
    Detect exactly one face in the frame using MediaPipe Face Detection (GPU).
    Returns [x1, y1, x2, y2] if exactly one face is found, otherwise None.
    """
    global face_detector_instance, face_detector_lock

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    with face_detector_lock:
        with torch.inference_mode():
            result = face_detector_instance.detect(mp_image)

    if result is None or len(result.detections) != 1:
        return None

    detection = result.detections[0]
    bbox = detection.bounding_box

    # Convert to pixel coordinates
    x1 = int(bbox.origin_x)
    y1 = int(bbox.origin_y)
    x2 = int(bbox.origin_x + bbox.width)
    y2 = int(bbox.origin_y + bbox.height)

    # Apply padding if args.pads exists
    try:
        pady1, pady2, padx1, padx2 = args.pads
        y1 = max(0, y1 - pady1)
        y2 = min(frame.shape[0], y2 + pady2)
        x1 = max(0, x1 - padx1)
        x2 = min(frame.shape[1], x2 + padx2)
    except NameError:
        pass

    return [x1, y1, x2, y2]

class FaceDetectProcessor:
    def __init__(self):
        self.lock = threading.Lock()
        self.in_queue = []  # stores frames for processing
        self.out_queue = [] # stores face locations found in frames
        self.position_to_output = None
        self.closed = False

        # variables for controlling the "process for every n-th frame" behavior
        self.current_input_count = 0
        self.current_output_count = 0

        self.input_semaphore = threading.Semaphore(0)

        threading.Thread(
            target=self.process,
            daemon=True,
        ).start()
    
    def close(self):
        with self.lock:
            self.closed = True

    def enqueue(self, frame):
        with self.lock:
            if self.current_input_count == 0:
                self.in_queue.append(frame)
                self.input_semaphore.release()
            self.current_input_count = (self.current_input_count + 1)%FACE_DETECT_FRAME_SKIP

    def dequeue(self):
        with self.lock:
            if(self.out_queue):
                if self.current_output_count == 0:
                    self.position_to_output = self.out_queue.pop(0)
                self.current_output_count = (self.current_output_count + 1)%FACE_DETECT_FRAME_SKIP
            return self.position_to_output

    def process(self):
        try:
            while True:
                timeout = not self.input_semaphore.acquire(timeout=30) # this blocks the thread until a frame is received

                with self.lock:
                    if self.closed:
                        break
                    if timeout:
                        continue
                    frame_to_process = self.in_queue.pop(0)

                face_location = face_detect_once(frame_to_process)

                with self.lock:
                    self.out_queue.append(face_location)
        finally:
            self.close()


# VAD utils
_silero_vad_model, _silero_vad_utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    onnx=False,
    trust_repo=True
)
_silero_vad_model = _silero_vad_model.to(device)

def detect_speech_in_bytes(
    audio_bytes: bytes,
    sampling_rate: int = 16000,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 500,
) -> float | None:
    """
    Detect the amount of time between end of audio and first detected speech.
    """
    global _silero_vad_model, _silero_vad_utils, device
    get_speech_timestamps, _, _, _, _ = _silero_vad_utils
    
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    wav = torch.from_numpy(audio_np / 32768.0).to(device)

    # --- Run VAD ---
    with torch.inference_mode():
        speech_timestamps = get_speech_timestamps(
            wav,
            _silero_vad_model,
            sampling_rate=sampling_rate,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
        )

    if not speech_timestamps:
        return None

    clip_length_sec = len(audio_np) / float(sampling_rate)
    first_speech_start_sec = speech_timestamps[0]["start"] / float(sampling_rate)
    return clip_length_sec - first_speech_start_sec