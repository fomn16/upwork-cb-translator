# transcription
import torch
from whisper_online import *
from stream_simulator import StreamSimulator
from benchmark import Benchmark
import os

#translation
from transformers import MarianMTModel, MarianTokenizer

# TTS
from TTS.tts.configs.xtts_config import XttsConfig
from custom_implementations.TTS.tts.models.xtts import Xtts
from huggingface_hub import snapshot_download
import torchaudio

# Pipeline
from pipeline import *
import time

input_language = "en"
output_language = "hi"

voiceFile = "arnold_original.mp3"
audioFile = "156550__acclivity__a-dream-within-a-dream.wav"

# Helper functions
def postprocess(wav): # taken from the coqui streaming example code
    if isinstance(wav, list):
        wav = torch.cat(wav, dim=0)
    wav = wav.clone().detach().cpu().numpy()
    wav = wav[None, : int(wav.shape[0])]
    wav = np.clip(wav, -1, 1)
    wav = (wav * 32767).astype(np.int16)
    return wav.tobytes()

def save_to_wav(wav):
    path = "testOutputs/"
    os.makedirs(path, exist_ok=True)
    path +=f"wav{time.time()}.wav"
    
    audio_np = np.frombuffer(wav, dtype=np.int16)
    audio_tensor = torch.from_numpy(audio_np).unsqueeze(0) # Unsqueeze to add channel dimension (1, N)
    torchaudio.save(path, audio_tensor, sample_rate=24000)

# initializing transcriptor
asr = FasterWhisperASR(input_language, "medium")  #options: tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,large-v3-turbo
asr.use_vad()
transcription_processor = OnlineASRProcessor(asr)
transcription_processor.init()

# initializing translator
translator_model_name = f"Helsinki-NLP/opus-mt-{input_language}-{output_language}"
translator_tokenizer = MarianTokenizer.from_pretrained(translator_model_name)
translator_model = MarianMTModel.from_pretrained(translator_model_name)

# initializing TTS
tts_model_dir = snapshot_download(repo_id="Abhinay45/XTTS-Hindi-finetuned")
tts_config = XttsConfig()
tts_config.load_json(tts_model_dir + "/config.json")
tts_model = Xtts.init_from_config(tts_config)
tts_model.load_checkpoint(tts_config, checkpoint_dir=tts_model_dir, use_deepspeed=True)
if torch.cuda.is_available():
    tts_model.cuda()
gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(audio_path=["testing_voices/" + voiceFile])

# declaring objects that will handle benchmarks
transcript_bench = Benchmark("transcript")
translate_bench = Benchmark("translate", buff_size=10)
tts_bench = Benchmark("tts", buff_size=10)

#### Building the pipeline ####

### Transcription Pipe ####
# function called every time an audio chunk is received
def audio_chunk_received(audioBytes: bytes):
    transcription_processor.insert_audio_chunk(audioBytes)
    
def transcription_iteration():
    global transcript_bench
    with transcript_bench:
        _,_,transcription = transcription_processor.process_iter()
    return transcription if transcription != "" else None

# function called when audio is finished, before closing the pipeline
def finish_transcription():
    global transcript_bench
    with transcript_bench:
        _,_,transcription = transcription_processor.finish()
    return transcription if transcription != "" else None
    # transcription_processor.init() # must do this if the processor will be reused

transcipt_pipe = TranscribePipe(audio_chunk_received, transcription_iteration, finish_transcription)

#### Translation Pipe ####
def translate_iteration(input:str) -> str:
    global translate_bench
    with translate_bench:
        inputs = translator_tokenizer(input, return_tensors="pt", padding=True, truncation=True)
        translated = translator_model.generate(**inputs)
        translated_text = ' '.join([translator_tokenizer.decode(t, skip_special_tokens=True) for t in translated])
    
    print(f'{input} -> {translated_text}')
    return translated_text

translate_pipe = TranslatePipe(translate_iteration)

#### TTS Pipe ####
def tts_iteration(input: str, send:Callable[[bytes], None]):
    global tts_bench
    with tts_bench:
        chunks = tts_model.inference_stream(
            input,
            "hi",
            gpt_cond_latent,
            speaker_embedding
        )
        for chunk in chunks:
            send(postprocess(chunk))

tts_pipe = TTSPipe(tts_iteration)

out_wav = bytearray()
def audio_out_iteration(input: bytes):
    global out_wav
    out_wav.extend(input)
output_pipe = AudioOutPipe(audio_out_iteration)

#### Connecting pipe sections ####
transcipt_pipe.to(translate_pipe).to(tts_pipe).to(output_pipe)
transcipt_pipe.open()

def to_pipeline(input: bytes):
    transcipt_pipe.receive(input)

# simulating received audio stream
simulator = StreamSimulator("benchmark_audios/" + audioFile, to_pipeline)
simulator.start_in_thread()
    
while not simulator.finished:
    time.sleep(1)

latency = time.perf_counter()

transcipt_pipe.close()

print(f'latency after end of stream = {time.perf_counter() - latency}')

save_to_wav(out_wav)

simulator.stop()

transcript_bench.show()
translate_bench.show()
tts_bench.show()