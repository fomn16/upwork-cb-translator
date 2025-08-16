ENABLE_TRANSLATION = True
ENABLE_VOICE_CLONE = True
TRANSLATION_CHUNK_SECONDS = 1

VOICE_EMBEDDINGS_LOCATION = "/home/test/docker_image/Docker_voice_clone/voice_embeddings"
VOICE_EMBEDDING_CHUNK_RECORDINGS_LOCATION = "/home/cb-translator/ml/recordings"
PROCESS_VOICE_EMBEDDING_EVERY_50_CHUNKS = False # enables/disables voice embedding reprocessing. If enabled again, i reccomend analysing the GPU memory usage