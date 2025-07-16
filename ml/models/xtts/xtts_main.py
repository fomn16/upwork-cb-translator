from pathlib import Path
import scipy
import numpy as np
import time
from TTS.vc.models.openvoice import OpenVoice
from TTS.api import TTS
import os

def save_to_wav(wav):
    path = "testOutputs/"
    os.makedirs(path, exist_ok=True)
    path +=f"wav{time.time()}.wav"

    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    wav_norm = wav_norm.astype(np.int16)
    scipy.io.wavfile.write(path, vc_model.config.audio.output_sample_rate, wav_norm)

print("Downloading model...")
tts = TTS(model_name="voice_conversion_models/multilingual/multi-dataset/openvoice_v2").cuda()
vc_model: OpenVoice = tts.voice_converter.vc_model

print("Computing and caching speaker latents...")
vc_model.voice_conversion("source_voice.wav","source_voice.wav", speaker_id="MySpeaker1", voice_dir="./voice_embeddings")

print('Starting benchmark...')
total_time_start = time.perf_counter()
full_total = []
folder = Path("./benchmark_audios")
for wav_path in folder.glob("*.wav"):
    audio = vc_model.load_audio(str(wav_path))
    full_time = time.perf_counter()
    converted_wav = vc_model.voice_conversion(audio, speaker_id="MySpeaker1", voice_dir="./voice_embeddings")
    full_time=time.perf_counter()-full_time
    full_total.append(full_time)
    print(f"generated conversion in ${full_time}")
    save_to_wav(converted_wav)
print(f'total:{time.perf_counter() - total_time_start}')
print(f"avrg: {np.average(full_total)}")