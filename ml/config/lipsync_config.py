from ml.config.audio_config import INTERNAL_SAMPLERATE

MINIMUM_AUDIO_BUFFER_SIZE_SECONDS = 0.1     # minimum size for input buffer
AUDIO_QUEUE_HISTORY_SIZE = 100              # store max length of the audio queue after the last 100 appends
MAX_VIDEO_SPEED_CHANGE = 0.15               # video cannot be sped up or slowed down more than 15%
FACE_DETECT_FRAME_SKIP = 5                  # runs face detection once every 15 received frames

MAX_LIPSYNC_MODEL_CHUNK_SECONDS = 1         # maximum ammount of time passed in each call to the lipsync model
MIN_LIPSYNC_MODEL_CHUNK_SECONDS = 0.2       # minimum ammount of time passed in each call to the lipsync model, only used at the end of a lipsync block call
OUTPUT_QUEUE_SIZE_SECONDS = 1               # size of the output queue, should be greater than the time it takes to process one lipsync chunk

DELAY_ESTIMATOR_ADJUSTMENT_SPEED = 0.01                     # must be between 0 and 1, with a higher value the video input queue size changes faster.
DELAY_ESTIMATOR_VAD_AUDIO_SAMPLES = INTERNAL_SAMPLERATE*2*2 # number of samples sent to the VAD, here is 2 seconds at 16 bits
DELAY_ESTIMATOR_TRANSLATOR_SILENCE_WAIT_TIME = 2            # the end of the translation is detected when no translation is received for 2 seconds
DELAY_ESTIMATOR_SLEEP_TIME = 30                             # time in seconds spend in the sleep state after a delay estimate is computed
DELAY_ESTIMATOR_MAX_VIDEO_DELAY_SECONDS = 10                # detected delays greater than this will be ignored