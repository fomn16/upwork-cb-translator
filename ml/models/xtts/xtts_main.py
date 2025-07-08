from pathlib import Path
import scipy
import numpy as np
import time
from cusom_implementation.vc.models.freevc import FreeVC
from cusom_implementation.api import TTS

def save_to_wav(wav):
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    wav_norm = wav_norm.astype(np.int16)
    scipy.io.wavfile.write(f"testOutputs/wav{time.time()}.wav", vc_model.config.audio.output_sample_rate, wav_norm)

print("Downloading model...")
tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).cuda()
vc_model: FreeVC = tts.voice_converter.vc_model # type: ignore

print("Computing speaker latents...")
speaker_embedding = vc_model.get_speaker_embedding("voice-profile.wav")

print('Starting benchmark...')
total_time_start = time.perf_counter()
full_total = []
folder = Path("./benchmark_audios")
for wav_path in folder.glob("*.wav"):
    audio = vc_model.new_load_audio(str(wav_path))
    full_time = time.perf_counter()
    converted_wav = vc_model.voice_conversion_preprocessed(audio,speaker_embedding)
    full_time=time.perf_counter()-full_time
    full_total.append(full_time)
    print(f"generated conversion in ${full_time}")
    save_to_wav(converted_wav)
print(f'total:{time.perf_counter() - total_time_start}')
print(f"avrg: {np.average(full_total)}")