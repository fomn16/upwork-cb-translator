IS_PROD = False
if IS_PROD:
    MEDIASERVER_IP = "10.10.0.82"
else:
    MEDIASERVER_IP = "127.0.0.1"

SAMPLE_READ_SIZE = 4096  # minimum number of bytes read from the audio buffers/arrays
OUTPUT_PERIOD = 0.02  # defines frequency at which output is written to the network

MAIN_ENDPOINT_PORT=2002

LIP_SYNC_RAW_AUDIO_IN_PORT = 6005
LIP_SYNC_TRANSLATED_AUDIO_IN_PORT = 6006
LIP_SYNC_AUDIO_OUT_PORT = 6007
LIP_SYNC_VIDEO_IN_PORT = 6008
LIP_SYNC_VIDEO_OUT_PORT = 6009