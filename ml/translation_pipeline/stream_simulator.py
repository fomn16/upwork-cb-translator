import wave
import time
from typing import Callable
import numpy as np
import librosa
import threading


class StreamSimulator:
    def __init__(
        self,
        wav_path: str,
        received: Callable[[bytes], None],
        chunk_size: int = 940,
        target_rate: int = 16000,
    ):
        """
        :param wav_path:     path to input WAV
        :param received:     callback(bytes, is_final_chunk)
        :param chunk_size:   # samples PER chunk at target_rate
        :param target_rate:  output sample rate (Hz), here 16 kHz
        """
        self.wav_path = wav_path
        self.received = received
        self.finished = False
        self.chunk_size = chunk_size
        self.target_rate = target_rate
        self._thread = None  # Thread object
        self._stop_event = threading.Event()  # Event to signal stopping

    def start(self):
        """Start the simulation in the current thread."""
        with wave.open(self.wav_path, 'rb') as wf:
            orig_rate = wf.getframerate()
            num_ch = wf.getnchannels()
            sampwidth = wf.getsampwidth()  # bytes per sample
            # how many input-frames to read so that after resampling
            # we get chunk_size frames at target_rate:
            in_chunk = int(round(self.chunk_size * orig_rate / self.target_rate))
            delay = self.chunk_size / self.target_rate
            # dtype for incoming PCM:
            dtype_in = np.dtype(f'<i{sampwidth}')

            while not self._stop_event.is_set():  # Check if stop is requested
                raw = wf.readframes(in_chunk)
                if not raw:
                    # signal end of stream
                    self.finished = True
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

    def start_in_thread(self):
        """Start the simulation in a separate thread."""
        self._thread = threading.Thread(target=self.start, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the simulation."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()