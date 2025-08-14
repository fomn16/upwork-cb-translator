#!/bin/bash

# I used this script during development in order to run all the pieces of the translation environment (including lauching the browser tabs)
# feel free to use as an exemple for execution, or to modify.

CONDA_PATH="/opt/miniconda3/etc/profile.d/conda.sh"
PIDS=()

# Permanent Chrome profile directory
PROFILE_DIR="$HOME/.chrome_script_profile"

# Create it if it doesn't exist
mkdir -p "$PROFILE_DIR"

cleanup() {
    echo "Stopping all terminals and Chrome..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    pkill -f "$PROFILE_DIR"  # Kill only Chrome processes using our script's profile
    exit 0
}

trap cleanup SIGINT

# Launch Konsole windows
konsole --hold -e bash -c "cd backend && yarn start:dev" &
PIDS+=($!)

sleep 0.1

konsole --hold -e bash -c "cd frontend && yarn dev" &
PIDS+=($!)

sleep 0.1

konsole --hold -e bash -c "source $CONDA_PATH && conda activate ml2 && cd ml && python main.py" &
PIDS+=($!)

sleep 0.1

konsole --hold -e bash -c "source $CONDA_PATH && conda activate lipsync && cd ml/pipelines/lipsync && export DISPLAY=:0 && export EGL_PLATFORM=surfaceless && python main.py" &
PIDS+=($!)

sleep 0.1

konsole --hold -e bash -c "source $CONDA_PATH && conda activate voice_clone && cd ml/pipelines/voice_clone && python main.py" &
PIDS+=($!)

sleep 0.1

konsole --hold -e bash -c "source $CONDA_PATH && conda activate seamless_t2s && cd ml/pipelines/seamless_t2s && python main.py" &
PIDS+=($!)

sleep 0.1

konsole --hold -e bash -c "source $CONDA_PATH && conda activate translation && cd ml/pipelines/translation && python main.py" &
PIDS+=($!)

sleep 1

# Open Chrome with persistent profile
google-chrome-stable --user-data-dir="$PROFILE_DIR" "http://localhost:5173/" --new-window &
google-chrome-stable --user-data-dir="$PROFILE_DIR" --incognito "http://localhost:5173/" &

wait