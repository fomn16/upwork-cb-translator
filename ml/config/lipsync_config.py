MINIMUM_AUDIO_BUFFER_SIZE_SECONDS = 0.1     # minimum size for input buffer
AUDIO_QUEUE_HISTORY_SIZE = 100              # store max length of the audio queue after the last 100 appends
MAX_VIDEO_SPEED_CHANGE = 0.15               # video cannot be sped up or slowed down more than 15%
FACE_DETECT_FRAME_SKIP = 5                  # runs face detection once every 15 received frames

MAX_LIPSYNC_MODEL_CHUNK_SECONDS = 1         # maximum ammount of time passed in each call to the lipsync model
MIN_LIPSYNC_MODEL_CHUNK_SECONDS = 0.2       # maximum ammount of time passed in each call to the lipsync model
OUTPUT_QUEUE_SIZE_SECONDS = 1               # size of the output queue, should be greater than the time it takes to process one lipsync chunk