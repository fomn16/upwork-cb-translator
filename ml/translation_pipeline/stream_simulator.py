import wave
import time
from typing import Callable
import numpy as np
import librosa
import threading
import os

class StreamSimulator:
    def __init__(
        self,
        audio_dir: str,
        received: Callable[[bytes], None],
        end_of_clip: Callable[[None], None],
        chunk_size: int = 940,
        target_rate: int = 16000,
        between_clip_delay: float = 5
    ):
        """
        :param audio_dir:     path to folder containing input WAV files
        :param received:     callback (bytes), function to call passing in bytes of audio
        :param received:     callback (receives no params) function to call when an audio clip finishes streaming
        :param chunk_size:   # samples PER chunk at target_rate
        :param target_rate:  output sample rate (Hz), here 16 kHz
        :param between_clip_delay:  float that says how long to wait (seconds) between clips being sent
        """
        self.audio_dir = audio_dir
        self.received = received
        self.finished = False
        self.chunk_size = chunk_size
        self.target_rate = target_rate
        self._thread = None  # Thread object
        self._stop_event = threading.Event()  # Event to signal stopping

        self.between_clip_delay = between_clip_delay
        self.end_of_clip = end_of_clip

    def start(self):
        """Start the simulation in the current thread."""
        audio_extensions = (".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a")
        for filename in os.listdir(self.audio_dir):
            # Check if the file has an audio extension
            if filename.lower().endswith(audio_extensions):
                with wave.open(self.audio_dir+filename, 'rb') as wf:
                    orig_rate = wf.getframerate()
                    num_ch = wf.getnchannels()
                    sampwidth = wf.getsampwidth()  # bytes per sample
                    # how many input-frames to read so that after resampling
                    # we get chunk_size frames at target_rate:
                    in_chunk = int(round(self.chunk_size * orig_rate / self.target_rate))
                    delay = self.chunk_size / self.target_rate
                    # dtype for incoming PCM:
                    dtype_in = np.dtype(f'<i{sampwidth}')
                    print(f"started streaming {filename}")

                    while not self._stop_event.is_set():  # Check if stop is requested
                        raw = wf.readframes(in_chunk)
                        if not raw:
                            break

                        # 1) bytes → int array
                        audio = np.frombuffer(raw, dtype=dtype_in)

                        # 2) down-mix to mono if needed
                        if num_ch > 1:
                            audio = audio.reshape(-1, num_ch).mean(axis=1).astype(dtype_in)

                        # 3) normalize to float32 ∈ [-1,1]
                        audio_f = audio.astype(np.float32) / np.iinfo(dtype_in).max

                        # 4) resample to 16 kHz if needed
                        if orig_rate != self.target_rate:
                            audio_f = librosa.resample(
                                audio_f, orig_sr=orig_rate, target_sr=self.target_rate
                            )

                        # 5) back to PCM16-LE bytes
                        audio_i16 = (audio_f * np.iinfo(np.int16).max).astype(np.int16)
                        chunk_bytes = audio_i16.tobytes()

                        # 6) deliver
                        self.received(chunk_bytes)

                        # 7) wait real-time
                        time.sleep(delay)
                    self.end_of_clip() # send signal saying current audio clip finished

                    # send in_chunk frames of 0 (empty audio) and wait the necessary amount of time
                    empty_audio = np.zeros(in_chunk, dtype=np.int16)  # Create empty audio
                    empty_bytes = empty_audio.tobytes()  # Convert to bytes
                    start_delay = time.perf_counter()
                    while(time.perf_counter() - start_delay < self.between_clip_delay):
                        self.received(empty_bytes)  # Deliver empty audio
                        time.sleep(delay)
        self.finished = True

    def start_in_thread(self):
        """Start the simulation in a separate thread."""
        self._thread = threading.Thread(target=self.start, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the simulation."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()