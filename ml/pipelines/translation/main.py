import threading
import time
import torch
import torchaudio
import numpy as np
import re
from multiprocessing.connection import Client
from scipy import signal
from scipy.signal import butter, lfilter
import noisereduce as nr
import socket
import json
import pickle

import whisper_utils as wms
from sentence_splitter import SentenceSplitter

import os
import sys
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.communication.communication_helper import CommunicationHelper
from lib.communication.session_settings import SessionSettings
from lib.queue.audio_queue import AudioQueue

from config.connection_config import *
from config.audio_config import *
from config.translation_config import *
from bench import *

whisper_sessions = {}

def process_voice_embedding_chunk(
    seg,
    session_id,
    chunk_count,
    audio_buffer,
    tensor_buffer,
    embedding_counter,
    speaker_id,  # 🔸 current speaker_id passed in
    save_audio,
    request_voice_embed,
    bytes_to_float32_mono_array,
    bandpass_filter,
    nr,
    is_silent
):
    audio_buffer.append(seg)
    #print(session_id, chunk_count)
    if chunk_count == 50 and PROCESS_VOICE_EMBEDDING_EVERY_50_CHUNKS:
        os.makedirs(VOICE_EMBEDDING_CHUNK_RECORDINGS_LOCATION, exist_ok=True)
        print("🟡 Received 50 chunks – checking for silence")

        for chunk in audio_buffer:
            try:
                mono_tensor = bytes_to_float32_mono_array(chunk)
                mono_tensor = bandpass_filter(mono_tensor, 16000)
                mono_tensor = nr.reduce_noise(y=mono_tensor, sr=16000)
                if not is_silent(mono_tensor):
                    tensor_buffer.append(mono_tensor)
            except Exception as e:
                print(f"⚠️ Failed to process chunk: {e}")

        if len(tensor_buffer) == 0:
            print("❌ All chunks were silent. Skipping embedding.")
            audio_buffer.clear()
            tensor_buffer.clear()
            return 0, embedding_counter, speaker_id  # Return unchanged speaker_id

        # ✅ Do embedding
        tensor_buffer = [torch.tensor(x) if isinstance(x, np.ndarray) else x for x in tensor_buffer]
        stitched_tensor = torch.cat(tensor_buffer, dim=-1).to(torch.float32)

        saved_path = save_audio(stitched_tensor, sample_rate=16000, prefix=f"{session_id}")
        print(saved_path)
        #new_speaker_id = f"{session_id}"
        #emb_time = time.time()
        request_voice_embed(saved_path, speaker_id)
        #print(f"Time taken to request voice embedding: {time.time() - emb_time}")
        print(f"✅ Voice embedding done for {speaker_id}")
        embedding_counter += 1

        # try:
        #     os.remove(saved_path)
        #     print(f"🧹 Deleted temp audio file: {saved_path}")
        # except Exception as e:
        #     print(f"⚠️ Failed to delete temp audio file: {e}")

        audio_buffer.clear()
        tensor_buffer.clear()
        return 0, embedding_counter, speaker_id  # ✅ Return updated speaker_id

    # Not enough chunks yet
    return chunk_count, embedding_counter, speaker_id

def save_audio(audio_tensor, sample_rate=INTERNAL_SAMPLERATE, prefix="embedding"):
    # Remove anything after '@'
    if "@" in prefix:
        prefix = prefix.split("@")[0]

    filename = f"{prefix}_{time.time()}.wav"
    os.makedirs("recordings", exist_ok=True)
    relative_path = os.path.join("recordings", filename)
    full_path = os.path.abspath(relative_path)  # 🔁 Convert to absolute path

    # Ensure correct shape and dtype for torchaudio
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    if audio_tensor.dtype != torch.float32:
        audio_tensor = audio_tensor.to(torch.float32)

    torchaudio.save(full_path, audio_tensor.cpu(), sample_rate)
    return full_path

def sanitize_speaker_id(speaker_id):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", speaker_id)

def request_voice_embed(audio_path, speaker_id="Utkarsh_hindi"):
    """
    Sends a voice embedding request using an audio file path.

    Returns True on success, None on failure.
    """
    safe_speaker_id = sanitize_speaker_id(speaker_id)
    conn = Client(('0.0.0.0', VOICE_CLONE_SERVER_PORT), authkey=b'secret_vc')
    conn.send({
        "mode": "embed",
        "audio_path": audio_path,
        "speaker_id": safe_speaker_id
    })
    result = conn.recv()
    conn.close()
    return result

def bytes_to_float32_mono_array(audio_bytes: bytes, input_sr=INTERNAL_SAMPLERATE, target_sr=INTERNAL_SAMPLERATE) -> np.ndarray:
    # 1. Decode stereo int16 bytes to numpy array
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

    # Print min/max and energy before normalization
    #print(f"🔊 Min: {mono_np.min()}, Max: {mono_np.max()}, Mean: {mono_np.mean():.4f}")

    # 3. Convert to float32 [-1.0, 1.0]
    waveform = torch.tensor(audio_np, dtype=torch.float32) / 32768.0
    waveform = waveform.unsqueeze(0)  # (1, N)

    # 4. Resample
    resampled = torchaudio.functional.resample(waveform, orig_freq=input_sr, new_freq=target_sr)
    return resampled.squeeze(0).numpy()

def bandpass_filter(waveform: np.ndarray, sample_rate: int, lowcut: float = 300.0, highcut: float = 3400.0, order: int = 5) -> np.ndarray:
    """
    Apply a bandpass filter to isolate voice frequencies (300 Hz to 3400 Hz).

    Args:
        waveform (np.ndarray): The mono audio waveform as a 1D float32 NumPy array.
        sample_rate (int): The sampling rate of the audio in Hz.
        lowcut (float): The lower cutoff frequency in Hz (default: 300 Hz).
        highcut (float): The upper cutoff frequency in Hz (default: 3400 Hz).
        order (int): The order of the Butterworth filter (default: 5).

    Returns:
        np.ndarray: The filtered audio waveform as a 1D float32 NumPy array.
    """
    # Calculate the Nyquist frequency (half the sampling rate)
    nyquist = 0.5 * sample_rate

    # Normalize cutoff frequencies to the Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design the Butterworth bandpass filter
    b, a = butter(order, [low, high], btype='band')

    # Apply the filter to the waveform
    filtered_waveform = lfilter(b, a, waveform)

    return filtered_waveform

def is_silent(audio_tensor, threshold_db=-45.0, zcr_threshold=0.12):
    if isinstance(audio_tensor, torch.Tensor):
        audio_tensor = audio_tensor.detach().cpu().numpy()

    # 1. RMS in dB
    rms = np.sqrt(np.mean(np.square(audio_tensor)))
    db = 20 * np.log10(rms + 1e-6)
    #print(f"🔊 RMS dB: {db:.2f}")

    # 2. Quick Zero-Crossing Rate
    zero_crossings = np.sum(np.diff(np.sign(audio_tensor)) != 0)
    zcr = zero_crossings / len(audio_tensor)
    #print(f"🔄 ZCR: {zcr:.4f}")

    # Determine silence or noise
    return db < threshold_db and zcr > zcr_threshold

def convert_audio_format(audio_chunk: bytes, input_sr: int = INTERNAL_SAMPLERATE, target_sr: int = INTERNAL_SAMPLERATE):
    """
    Convert stereo PCM audio from 48kHz to 16kHz mono format for Whisper.
    
    Args:
        audio_chunk: Raw PCM int16 stereo audio bytes at input_sr
        input_sr: Input sample rate (default: 48000)
        target_sr: Target sample rate for Whisper (default: 16000)
    
    Returns:
        numpy.ndarray: Mono float32 audio array at target sample rate
    """
    try:
        # Convert bytes to numpy array (assuming int16 stereo)
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        
        # Convert to float32 and normalize to [-1, 1]
        mono_float = audio_data.astype(np.float32) / 32768.0
        
        # Resample if needed
        if input_sr != target_sr:
            # Simple resampling using scipy.signal.resample
            target_length = int(len(mono_float) * target_sr / input_sr)
            mono_float = signal.resample(mono_float, target_length)
        
        return mono_float.astype(np.float32)
    except Exception as e:
        print(f"❗ Error in audio format conversion: {e}")
        # Return silence if conversion fails
        duration_seconds = len(audio_chunk) / (input_sr * 2 * 2)  # stereo int16
        target_samples = int(duration_seconds * target_sr)
        return np.zeros(target_samples, dtype=np.float32)
    
def is_valid_english(text: str, min_len: int = 2) -> bool:
    """
    Returns True if text contains mostly English letters and is above min_len.
    """
    if not text:
        return False
    text_stripped = text.strip()

    # Skip if too short or just punctuation
    if len(text_stripped) < min_len or re.fullmatch(r"[^\w]*", text_stripped):
        return False

    # Check ratio of English letters to all characters
    english_chars = re.findall(r"[A-Za-z]", text_stripped)
    ratio = len(english_chars) / len(text_stripped)
    return ratio > 0.5  # at least 50% English letters

def request_translation_tensor(
    host: str,
    port: int,
    text: str,
    tgt_lang: str,
    timeout: float = 10.0
):
    """
    Send text to Seamless server and receive translated speech as a PyTorch tensor.

    :param host: Server hostname or IP
    :param port: Server port
    :param text: Source text to translate
    :param tgt_lang: Target language code (e.g., 'hin' for Hindi)
    :param timeout: Socket timeout in seconds
    :return: (audio_tensor, sample_rate) or (None, None) on failure
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((host, port))

            # Prepare request
            req = json.dumps({"text": text, "tgt_lang": tgt_lang})
            s.sendall(req.encode())

            # Receive length prefix
            length_bytes = s.recv(4)
            if not length_bytes:
                print("[Client] No length prefix received")
                return None, None

            length = int.from_bytes(length_bytes, "big")

            # Receive payload
            data = b""
            while len(data) < length:
                packet = s.recv(length - len(data))
                if not packet:
                    print("[Client] Connection closed before all data received")
                    return None, None
                data += packet

            # Deserialize
            payload = pickle.loads(data)

            audio_tensor = payload["audio_tensor"]
            sample_rate = payload["sample_rate"]
            text_output = payload["text_output"]

            #print(text_output)

            if not isinstance(audio_tensor, torch.Tensor):
                audio_tensor = torch.tensor(audio_tensor, dtype=torch.float32)

            #print(f"[Client] Received tensor shape: {audio_tensor.shape}, sample rate: {sample_rate}")
            return audio_tensor, sample_rate, text_output

    except Exception as e:
        print(f"[Client] Error: {e}")
        return None, None, None
    
def request_voice_clone(audio_tensor, speaker_id="user_1", sample_rate=INTERNAL_SAMPLERATE):

    # Ensure it's CPU and numpy
    if isinstance(audio_tensor, torch.Tensor):
        audio_tensor = audio_tensor.detach().cpu().numpy()
    safe_speaker_id = sanitize_speaker_id(speaker_id)

    conn = Client(("0.0.0.0", VOICE_CLONE_SERVER_PORT), authkey=b"secret_vc")
    conn.send({
        "mode": "clone",
        "audio": audio_tensor,
        "sample_rate": sample_rate,
        "speaker_id": safe_speaker_id,
    })
    result = conn.recv()
    conn.close()
    return result

def tensor_to_bytes(translated_wav):
    audio_np = np.clip(np.array(translated_wav, dtype=np.float32), -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    return audio_int16.tobytes()

def resample_audio(audio_bytes, original_sr=INTERNAL_SAMPLERATE, target_sr=INTERNAL_SAMPLERATE):
    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
    new_length = int(len(audio_data) * target_sr / original_sr)
    resampled = signal.resample(audio_data, new_length)
    resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
    return resampled.tobytes()

def get_raw_audio_pcm(samples, sr: int, target_sr: int = INTERNAL_SAMPLERATE, stereo: bool = True):
    import torchaudio

    # Convert input to float32 tensor
    if isinstance(samples, list):
        samples = torch.tensor(samples, dtype=torch.float32)
    elif isinstance(samples, np.ndarray):
        samples = torch.from_numpy(samples).float()

    # Mono to [1, N]
    if samples.dim() == 1:
        samples = samples.unsqueeze(0)

    # Convert to stereo if needed
    if stereo and samples.size(0) == 1:
        samples = samples.repeat(2, 1)

    # Resample if needed
    if sr != target_sr:
        samples = torchaudio.functional.resample(samples, orig_freq=sr, new_freq=target_sr)

    # Clamp to [-1, 1]
    samples = torch.clamp(samples, -1.0, 1.0)

    return samples, target_sr  # shape: [2, N]

def process_translation_chunk_whisper(
    audio_chunk: bytes,
    output_function: None, #output_queue=None,
    input_sr: int = INTERNAL_SAMPLERATE,
    target_sr: int = INTERNAL_SAMPLERATE,
    target_lang: str = None,
    voice_clone_enabled: bool = False,
    session_id=None,
    speaker_id:str = None
):
    """
    Process an already-chunked 1s stereo audio segment through Whisper streaming ASR.

    :param audio_chunk: Raw PCM int16 stereo audio bytes at input_sr (default 48kHz).
    :param output_queue: Optional queue to push transcribed text into.
    :param input_sr: Input sample rate of the audio chunk
    :param target_sr: Target sample rate for Whisper processing
    """
    session_whisper = whisper_sessions.get(session_id)
    
    try:
        #start_time = time.time()
        
        # Convert audio format for Whisper processing
        audio_float = convert_audio_format(audio_chunk, input_sr, target_sr)
        
        # Insert audio chunk into Whisper online processor
        if session_whisper and session_whisper["online"]:
            #insert_audio_chunk_bench = time.perf_counter()
            n_bytes = len(audio_float)
            log_to_server('audio', 'whisper_insert_audio', session_id, n_bytes)
            session_whisper["online"].insert_audio_chunk(audio_float)
            #add_time_and_print(time.perf_counter()-insert_audio_chunk_bench, 'insert_audio_chunk')
            #print("seg reached here")

            
            # Process the audio chunk
            #process_iter_bench = time.perf_counter()
            log_to_server('audio', 'whisper_process_iter', session_id, n_bytes)
            result = session_whisper["online"].process_iter()
            #add_time_and_print(time.perf_counter()-process_iter_bench, 'process_iter')

            #print(result)
            # Extract transcribed text if available

            #log_to_server('text', 'whisper_process_iter', session_id, len(result.split()))
            transcribed_text = wms.output_transcript(result)
            if transcribed_text and is_valid_english(transcribed_text):
                log_to_server('text', 'whisper_process_iter_post', session_id, len(transcribed_text.split()))
                print(f"🎤 Transcribed: {transcribed_text}", flush=True)

                #translation_time = time.perf_counter()
                audio_tensor, sr, text_output = request_translation_tensor(
                    host="0.0.0.0",
                    port=SEAMLESS_T2S_SERVER_PORT,
                    text=transcribed_text,
                    tgt_lang=target_lang
                )
                log_to_server('text', 'request_translation_tensor', session_id, len(text_output.split()))

                #translation_time = time.perf_counter() - translation_time
                #add_time_and_print(translation_time, 'request_translation_tensor')
                #print(f"seamless time = {translation_time}")

                if audio_tensor is not None:
                    print(f"session id :{session_id}, text_output: {text_output}")
                    if voice_clone_enabled:
                        # 🔍 Check if embedding exists
                        speaker_path = os.path.join(
                            VOICE_EMBEDDINGS_LOCATION,
                            f"{speaker_id.replace('@', '_').replace('.', '_')}.pth"
                        )
                        if not os.path.exists(speaker_path):
                            print(f"⚠️ Speaker embedding not found for '{speaker_id}'. Using default.")
                            speaker_id = "Ayan"
                        else:
                            speaker_id = speaker_id

                        print(f"Cloning voice for {speaker_id}")
                        #request_voice_clone_bench = time.perf_counter()
                        clone_tensor = request_voice_clone(audio_tensor, speaker_id=speaker_id, sample_rate=sr)
                        #add_time_and_print(time.perf_counter() - request_voice_clone_bench, 'request_voice_clone')

                        cloned_audio_bytes = tensor_to_bytes(clone_tensor)
                        translated_audio_bytes = resample_audio(cloned_audio_bytes, 22050, INTERNAL_SAMPLERATE)

                        log_to_server('audio', 'translated_audio_bytes', session_id, len(translated_audio_bytes))
                    else:
                        samples, sr = get_raw_audio_pcm(audio_tensor, sr, stereo=False)
                        translated_audio_bytes = (samples.numpy() * 32767.0).astype(np.int16).T.tobytes()
                    output_function(translated_audio_bytes)

                    #processing_time = time.time() - start_time
                    #print(f"⏱️ Whisper processing time: {processing_time:.4f}s")
        else:
            print("❗ Whisper online processor not initialized")
            output_function(audio_chunk)

    except Exception as e:
        print(f"❗ Exception in Whisper processing: {e}")
        # Pass through original audio on error
        if output_function:
            output_function(audio_chunk)

def initialize_whisper_for_session(session_id, lang:str):
    """Initialize Whisper ASR for a given session."""

    #segmenter = SentenceSplitter(lang=tgt_lang, use_gpu=True)  # use this tokenizer if results from the default one are not satisfactory
    class WhisperArgs:
        def __init__(self):
            self.language = lang # this is the source language. target is either the same (if task == transcript) or english (task==translate)
            self.min_chunk_size = 0.7
            self.vac = True
            self.model = "medium"
            self.task = "translate"
            self.model_cache_dir = None
            self.model_dir = None
            self.backend = "faster-whisper"
            self.vac_chunk_size = 0.05
            self.vad = False
            self.buffer_trimming = "segment"
            self.buffer_trimming_sec = 15
            self.log_level = "WARNING"

    args = WhisperArgs()
    wms.set_logging(args, wms.logger)

    print(f"Initializing Whisper ASR for session {session_id}...")
    whisper_asr, whisper_online = wms.asr_factory(args, logfile=sys.stderr)

    # Warm-up
    dummy_audio = np.zeros(int(16000), dtype=np.float32)
    whisper_asr.transcribe(dummy_audio)

    whisper_sessions[session_id] = {
        "asr": whisper_asr,
        "online": whisper_online
    }
    print(f"✅ Whisper ASR ready for session {session_id}")

CHUNK_SIZE_BYTES = INTERNAL_SAMPLERATE*2*TRANSLATION_CHUNK_SECONDS

class Session:
    def __init__(self, session_id, audio_socket, settings: SessionSettings):
        self.session_id = session_id
        self.received_data = threading.Event()

        self.audio_in = AudioQueue()
        self.audio_socket = audio_socket

        self.settings = settings

        initialize_whisper_for_session(session_id, lang=self.settings.source_language)
        
        threading.Thread(
            target=self.process,
            daemon=True,
        ).start()

    def update_settings(self, settings: SessionSettings):
        self.settings = settings

    def output(self, audio_bytes):
        log_to_server("audio",f"translation sent", self.session_id, len(audio_bytes))
        self.audio_socket.send(self.session_id, audio_bytes)

    def process(self):
        try:
            audio_buffer = []  # list of (bytes) chunks
            tensor_buffer = []  # list of tensors to stitch
            chunk_count = 0
            embedding_counter = 0
            while True: 
                timeout = not self.received_data.wait(timeout=30) # waits for data to be received, also has a timeout
                if(self.audio_in.closed or self.settings.closed):
                    break
                if(timeout):
                    continue

                if(len(self.audio_in)>=CHUNK_SIZE_BYTES):
                    #bench_time = time.perf_counter()
                    chunk_count += 1
                    seg = self.audio_in.dequeue(CHUNK_SIZE_BYTES)
                    log_to_server('audio', 'translate_raw_received', self.session_id, len(seg))
                    #####################################################################################
                    if self.settings.enable_voice_clone:
                        # 🔁 Inside your while loop:
                        #vc_time = time.perf_counter()
                        chunk_count, embedding_counter, speaker_id = process_voice_embedding_chunk(
                            seg=seg,
                            session_id=self.session_id,
                            chunk_count=chunk_count,
                            audio_buffer=audio_buffer,
                            tensor_buffer=tensor_buffer,
                            embedding_counter=embedding_counter,
                            speaker_id=self.settings.user_id,  # ✅ Pass current speaker_id
                            save_audio=save_audio,
                            request_voice_embed=request_voice_embed,
                            bytes_to_float32_mono_array=bytes_to_float32_mono_array,
                            bandpass_filter=bandpass_filter,
                            nr=nr,
                            is_silent=is_silent
                        )

                        #add_time_and_print(time.perf_counter()-vc_time, 'process_voice_embedding_chunk')
                    ##############################################################

                    if self.settings.enable_translation:
                        #t_time = time.perf_counter()
                        process_translation_chunk_whisper(
                            seg,                            # audio chunk bytes
                            self.output,#output_queue,      # function that outputs audio bytes # (previously, your queue to receive transcriptions)
                            input_sr=INTERNAL_SAMPLERATE,   # input sample rate of the audio chunk
                            target_sr=INTERNAL_SAMPLERATE,  # sample rate expected by Whisper
                            target_lang = self.settings.target_language,
                            voice_clone_enabled = self.settings.enable_voice_clone,
                            session_id=self.session_id,
                            speaker_id=self.settings.user_id
                        )

                        #add_time_and_print(time.perf_counter()-t_time, 'process_translation_chunk_whisper')
                    else: # if not translating, pass audio through
                        self.output(seg)
                    #add_time_and_print(time.perf_counter()-bench_time, 'translation_process')

        except Exception as e:
            print(f"[translate]: Error in processing thread: {e}")
            raise
        finally:
            self.close()

    def close(self):
        self.settings.closed = True

    def translate(self, audio_bytes):
        self.audio_in.enqueue(audio_bytes)
        self.received_data.set()

class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.audio_socket = None
        self.lock = threading.Lock()

    def is_initialized(self, session_id, settings:SessionSettings|None = None):
        with self.lock:
            if session_id not in self.sessions:     # if session was not created yet
                if(settings is None):               # only create session on the first "receive_session_settings" call
                    return False
                self.sessions[session_id] = None    # stops other threads from trying to acces the session while its created below
            else:
                return self.sessions[session_id] is not None

        session = Session(session_id, self.audio_socket, settings)

        with self.lock:
            self.sessions[session_id] = session
        
        return True

    def translate(self, session_id, audio_bytes):
        if(self.audio_socket == None):
            return
        if self.is_initialized(session_id):
            self.sessions[session_id].translate(audio_bytes)

    def close(self):
        for sess in self.sessions.values():
            sess.close()

    def receive_session_settings(self, session_id:str, settings:SessionSettings):
        if self.is_initialized(session_id, settings):   # on first call, this also initializes the session with the settings
            self.sessions[session_id].update_settings(settings)

if __name__ == "__main__":
    try:
        print("[Translate] Creating session manager")
        sessionManager = SessionManager()
        audio_socket = CommunicationHelper("audio_translate_socket", TRANSLATION_VOICE_CLONE_IN_PORT, TRANSLATION_VOICE_CLONE_OUT_PORT, sessionManager.translate, sessionManager.receive_session_settings)
        sessionManager.audio_socket = audio_socket

        while(audio_socket.online):
            time.sleep(1)
    finally:
        print("[Translate] Closing all processing")
        sessionManager.close()
        audio_socket.close()