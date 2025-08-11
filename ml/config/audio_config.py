EXTERNAL_SAMPLERATE = 48000     # samplerate in mediasoup
AUDIO_CHUNK_DURATION = 0.02 # seconds of audio sent to mediassoup each time data is returned
N_AUDIO_CHUNK_SAMPLES = int(AUDIO_CHUNK_DURATION*EXTERNAL_SAMPLERATE)