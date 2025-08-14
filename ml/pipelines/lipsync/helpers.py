import os
import torch

#for lip sync
import numpy as np
import subprocess
from tqdm import tqdm
from models import Wav2Lip
import lib.audio as audio
import argparse
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import threading
import queue
import torchaudio
import io

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

def face_detect_many(images):
    start_time = time.time()
    results = []
    last_box = None

    for i, img in enumerate(images):
        if i % 10 == 0:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yolo_result = face_detector_instance.predict(img_rgb, verbose=False)
            boxes = yolo_result[0].boxes.xyxy

            if boxes is None or len(boxes) == 0:
                last_box = None
                results.append(None)
                continue

            x1, y1, x2, y2 = boxes[0].int().tolist()
            pady1, pady2, padx1, padx2 = args.pads
            y1 = max(0, y1 - pady1)
            y2 = min(img.shape[0], y2 + pady2)
            x1 = max(0, x1 - padx1)
            x2 = min(img.shape[1], x2 + padx2)
            last_box = [x1, y1, x2, y2]

        results.append(last_box)

    return results

def load_model(path):
    start = time.time()
    model = Wav2Lip()
    print("Loading checkpoint from: {}".format(path))
    checkpoint = torch.load(path, map_location=device)
    s = checkpoint["state_dict"]
    new_s = {k.replace('module.', ''): v for k, v in s.items()}
    model.load_state_dict(new_s)
    model = model.to(device)
    model.eval()
    print(f"[load_model] Loaded in {time.time() - start:.2f}s")
    return model

model = load_model("wav2lip_Chinese.pth")

# ✅ MODIFIED datagen (preserve original frame if no face)
def datagen(frames, mels, face_det_results):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    #print(len(frames), len(mels))
    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame = frames[idx]
        face_coords = face_det_results[idx]

        if face_coords is None:
            # No face detected: keep frame, insert None for coords
            frame_batch.append(frame)
            coords_batch.append(None)
            continue

        x1, y1, x2, y2 = face_coords
        face = frame[y1:y2, x1:x2]
        face = cv2.resize(face, (args.img_size, args.img_size))
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame)
        coords_batch.append((x1, y1, x2, y2))

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
            img_masked = img_batch.copy()
            img_masked[:, args.img_size // 2:] = 0
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch = np.asarray(img_batch, dtype=np.float32) / 255.0
        mel_batch = np.asarray(mel_batch, dtype=np.float32)
        img_masked = img_batch.copy()
        img_masked[:, args.img_size // 2:, :, :] = 0
        img_combined = np.empty((img_batch.shape[0], img_batch.shape[1], img_batch.shape[2], img_batch.shape[3] * 2), dtype=np.float32)
        img_combined[:, :, :, :img_batch.shape[3]] = img_masked
        img_combined[:, :, :, img_batch.shape[3]:] = img_batch
        mel_batch = mel_batch[..., np.newaxis]
        yield img_combined, mel_batch, frame_batch, coords_batch

def video_writer_worker(write_queue, output_path, frame_size, fps):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    while True:
        frame = write_queue.get()
        if frame is None:
            break
        writer.write(frame)

    writer.release()


def run_lipsync_from_frames(frame_buffer: list, audio_bytes: bytes, face_detect: list):
    full_frames = frame_buffer
    frames_output = []
    print(f"[stream] Received {len(full_frames)} frames")

    # === 1. Audio Preprocessing ===
    audio_start = time.time()
    waveform_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    waveform_float32 = waveform_int16.astype(np.float32) / 32768.0
    mel = audio.melspectrogram(waveform_float32)
    print(f"[stream] Audio processed in {time.time() - audio_start:.2f}s")

    # === 2. Mel Chunks Creation ===
    mel_chunks, mel_step_size = [], 16
    mel_idx_multiplier = 80. / FRAME_RATE
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > mel.shape[1]:
            mel_chunks.append(mel[:, -mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx:start_idx + mel_step_size])
        i += 1
    print(f"[stream] Mel chunks: {len(mel_chunks)}")

    # === 3. Match mel_chunks and frames ===
    print(f"mel chunks: {len(mel_chunks)}, fullframes: {len(full_frames)}")
    if len(mel_chunks) > len(full_frames):
        full_frames += [full_frames[-1]] * (len(mel_chunks) - len(full_frames))
    else:
        full_frames = full_frames[:len(mel_chunks)]

    # === 4. Prepare datagen ===
    gen = datagen(full_frames.copy(), mel_chunks, face_detect)

    # === 6. Run Inference ===
    frames_written = 0
    args.wav2lip_batch_size = min(len(mel_chunks), args.wav2lip_batch_size or 32)
    if args.wav2lip_batch_size == 0:
        raise ValueError("No mel chunks available.")

    for i, (img_batch, mel_batch, frames, coords) in enumerate(
        tqdm(gen, total=int(np.ceil(len(mel_chunks) / args.wav2lip_batch_size)))
    ):
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            if c is None:
                frames_output.append(f)
                continue
            x1, y1, x2, y2 = c
            if x2 <= x1 or y2 <= y1:
                frames_output.append(f)
                continue
            try:
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = cv2.addWeighted(f[y1:y2, x1:x2], 0.2, p, 0.8, 0)
            except Exception as e:
                print(f"[warning] Error applying lipsync: {e}")
            frames_output.append(f)
            frames_written += 1

    # === 7. Fallback if no frames written ===
    if frames_written == 0:
        print("⚠️ No processed frames. Writing fallback video with original frames.")
        for f in full_frames:
            frames_output.append(f)
    print(f"[stream] Lip-synced video created")
    return frames_output

'''  
with Listener(address, authkey=authkey) as listener:
    while True:
        print("[Audio Worker] Waiting for a job...")
        with listener.accept() as conn:
            print("[Audio Worker] Job accepted")
            try:
                data = conn.recv_bytes()
                frame_buffer, audio_path, output_path = pickle.loads(data)
                print(f"frame_size: {len(frame_buffer)}")

                print("[Audio Worker] Running lip sync...")
                no_audio_path, final_path = run_lipsync_from_frames(
                    frame_buffer=frame_buffer,
                    audio_bytes=audio_path,
                    output_path=output_path,
                    with_audio=True
                )
                conn.send((no_audio_path, final_path, None))
                print("[Audio Worker] Job completed")

            except Exception as e:
                print(f"[Audio Worker] Error: {e}")
                conn.send((None, None, str(e)))
'''

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