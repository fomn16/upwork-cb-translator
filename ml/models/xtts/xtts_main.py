from pathlib import Path
import numpy as np
import time

from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan

import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import librosa
import soundfile as sf

from datasets import load_dataset

def save_to_wav(wav, name):
    sf.write(f"testOutputs/speecht5->bn->{name}.wav", wav.numpy(), samplerate=16000)

print("Downloading model...")

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

voice_folder = Path("./testing_voices")
voices = list(voice_folder.glob("*.mp3")) + list(voice_folder.glob("*.wav"))

for voice_path in voices:
    name = voice_path.stem.split("_")[0].split("-")[0].split(".")[0]

    print("Computing speaker latents...")
    
    xvec_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb"
    )
    waveform, sr = torchaudio.load(voice_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
    waveform = waveform.mean(dim=0, keepdim=True)
    with torch.no_grad():
        embeddings = xvec_model.encode_batch(waveform)
    speaker_embedding = embeddings.squeeze().unsqueeze(0)

    #speaker_embedding = np.load("cmu_us_bdl_arctic-wav-arctic_a0009.npy")
    #speaker_embedding = torch.tensor(speaker_embedding).unsqueeze(0)

    print('Starting benchmark...')
    total_time_start = time.perf_counter()
    full_total = []
    folder = Path("./benchmark_audios")
    for wav_path in folder.glob("*.mp3"):
        audio, _ = librosa.load(wav_path, sr=16000)
        full_time = time.perf_counter()

        inputs = processor(audio=audio, sampling_rate=16000, return_tensors="pt")
        speech = model.generate_speech(inputs["input_values"], speaker_embedding, vocoder=vocoder)

        full_time=time.perf_counter()-full_time
        full_total.append(full_time)
        print(f"generated conversion in ${full_time}")
        save_to_wav(speech, name)

print(f'total:{time.perf_counter() - total_time_start}')
print(f"avrg: {np.average(full_total)}")