from whisper_online import *
from stream_simulator import StreamSimulator
from benchmark import Benchmark

# initializing transcriptor
asr = FasterWhisperASR("en", "large-v2")  #options: tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,large-v3-turbo
asr.use_vad()
transcription_processor = OnlineASRProcessor(asr)
transcription_processor.init()

# function called every time an audio chunk is received
def audio_chunk_received(audioBytes: bytes):
    transcription_processor.insert_audio_chunk(audioBytes)

# declaring object that will handle benchmarks
whisper_benchmark = Benchmark("whisper")

# simulating received audio stream
simulator = StreamSimulator('benchmark_audios/156550__acclivity__a-dream-within-a-dream.wav', audio_chunk_received)
simulator.start_in_thread()

while not simulator.finished:
    with whisper_benchmark:
        _,_,transcription = transcription_processor.process_iter()
    if(transcription != ''):
        print(transcription)
print(transcription_processor.finish())
#transcription_processor.init() # if the processor will be reused


simulator.stop()
whisper_benchmark.show()