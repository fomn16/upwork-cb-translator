to run, first do

pip install -r requirements.txt

then download blaze_face_short_range from https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/index#models
and download the wav2lip .pth file. both files can stay in the same folder as main.

then run main.py, which will start a process listening on the ports configured in connection_config.py

if you get the following error

self._runner = _TaskRunner.create(graph_config, packet_callback)
RuntimeError: Service "kGpuService", required by node mediapipe_tasks_vision_face_detector_facedetectorgraph__mediapipe_tasks_core_inferencesubgraph__inferencecalculator__
mediapipe_tasks_vision_face_detector_facedetectorgraph__mediapipe_tasks_core_inferencesubgraph__InferenceCalculator, was not provided and cannot be created: ; 
RET_CHECK failure (mediapipe/gpu/gl_context_egl.cc:77) display != EGL_NO_DISPLAYeglGetDisplay() returned error 0x3000

you might need to run the following to add a few env variables before runing main:

export DISPLAY=:0
export EGL_PLATFORM=surfaceless