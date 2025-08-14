import torch
import torchaudio
import numpy as np
from whisper_online import FasterWhisperASR, OnlineASRProcessor
from transformers import MarianMTModel, MarianTokenizer
from TTS.tts.configs.xtts_config import XttsConfig
from custom_implementations.TTS.tts.models.xtts import Xtts
from huggingface_hub import snapshot_download
from pipeline import TranscribePipe, TranslatePipe, TTSPipe, AudioOutPipe

input_language = "en"
output_language = "hi"
voiceFile = "felipe.wav"

################# adding hindi support to num2words (used by XTTS) using indic-num2words ################
from pipelines.translation.custom_implementations.hin_nums2words import num_to_word as indic_num2words
from num2words.base import Num2Word_Base
from num2words import num2words, CONVERTER_CLASSES

class Num2Word_HI(Num2Word_Base):
    def to_cardinal(self, number):
        return indic_num2words(number, lang='hi')
    def to_ordinal(self, number):
        # indic-num2words doesn't have ordinal support, we might need to implement our own
        return f"{indic_num2words(number, lang='hi')}वाँ"
    
CONVERTER_CLASSES['hi'] = Num2Word_HI()

def postprocess(wav, target_rate=16000, orig_rate=24000):
    if isinstance(wav, list):
        wav = torch.cat(wav, dim=0)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    wav = wav.detach().cpu()
    if orig_rate != target_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_rate, new_freq=target_rate)
        wav = resampler(wav)
    wav = wav.numpy()
    wav = np.clip(wav, -1, 1)
    wav = (wav * 32767).astype(np.int16)
    return wav.tobytes()

def run_pipeline(conn):
    print("loading models")
    asr = FasterWhisperASR(input_language, "medium")
    asr.use_vad()
    transcription_processor = OnlineASRProcessor(asr)
    transcription_processor.init()

    translator_model_name = f"Helsinki-NLP/opus-mt-{input_language}-{output_language}"
    translator_tokenizer = MarianTokenizer.from_pretrained(translator_model_name)
    translator_model = MarianMTModel.from_pretrained(translator_model_name)

    tts_model_dir = snapshot_download(repo_id="Abhinay45/XTTS-Hindi-finetuned")
    tts_config = XttsConfig()
    tts_config.load_json(tts_model_dir + "/config.json")
    tts_model = Xtts.init_from_config(tts_config)
    tts_model.load_checkpoint(tts_config, checkpoint_dir=tts_model_dir, use_deepspeed=True)
    if torch.cuda.is_available():
        tts_model.cuda()
    gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(audio_path=["testing_voices/" + voiceFile])

    # Build pipeline
    def audio_chunk_received(audioBytes: bytes):
        transcription_processor.insert_audio_chunk(audioBytes)

    def transcription_iteration():
        _, _, transcription = transcription_processor.process_iter()
        return transcription if transcription else None

    def finish_transcription():
        _, _, transcription = transcription_processor.finish()
        return transcription if transcription else None

    transcribe_pipe = TranscribePipe(audio_chunk_received, transcription_iteration, finish_transcription)

    def translate_iteration(input_text: str) -> str:
        inputs = translator_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        translated = translator_model.generate(**inputs)
        translated_text = ' '.join([translator_tokenizer.decode(t, skip_special_tokens=True) for t in translated])
        print(f'{input_text} -> {translated_text}')
        return translated_text

    translate_pipe = TranslatePipe(translate_iteration)

    def tts_iteration(input_text: str, send):
        chunks = tts_model.inference_stream(input_text, "hi", gpt_cond_latent, speaker_embedding)
        for chunk in chunks:
            send(postprocess(chunk, target_rate=16000, orig_rate=24000))

    tts_pipe = TTSPipe(tts_iteration)

    def audio_out_iteration(output_bytes: bytes):
        conn.send(output_bytes)

    output_pipe = AudioOutPipe(audio_out_iteration)

    transcribe_pipe.to(translate_pipe).to(tts_pipe).to(output_pipe)
    transcribe_pipe.open()

    # Main loop: receive commands from parent
    print("model loading complete")
    while True:
        msg = conn.recv()
        if msg == "close":
            break
        elif isinstance(msg, bytes):
            transcribe_pipe.receive(msg)

    conn.close()