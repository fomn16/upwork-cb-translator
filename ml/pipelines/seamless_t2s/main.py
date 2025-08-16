import socket
import json
import torch
import pickle
from seamless_communication.inference import Translator

import time

import os
import sys
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.connection_config import *

# Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"running on {device}")
model_name = "seamlessM4T_v2_large"
vocoder_name = "vocoder_v2" if model_name == "seamlessM4T_v2_large" else "vocoder_36langs"
translator = Translator(model_name, vocoder_name, device=device, dtype=torch.float16)

# this takes a while, but in my tests it cut the inference time by more than half
import torch._dynamo
torch._dynamo.config.cache_size_limit = 32  # higher compilation cache = more RAM usage, smaller response time

# compiling model
if hasattr(translator.model, "text_encoder") and translator.model.text_encoder is not None:
    print("compiling text_encoder")
    translator.model.text_encoder = torch.compile(translator.model.text_encoder)

if hasattr(translator.model, "speech_encoder") and translator.model.speech_encoder is not None:
    print("compiling speech_encoder")
    translator.model.speech_encoder = torch.compile(translator.model.speech_encoder)

if hasattr(translator.model, "t2u_model") and translator.model.t2u_model is not None:
    print("compiling t2u_model")
    translator.model.t2u_model = torch.compile(translator.model.t2u_model)

if hasattr(translator.model, "u2u_model") and translator.model.u2u_model is not None:
    print("compiling u2u_model")
    translator.model.u2u_model = torch.compile(translator.model.u2u_model)

# Compile the vocoder if it exists
if hasattr(translator, "vocoder") and translator.vocoder is not None:
    print("compiling vocoder")
    translator.vocoder = torch.compile(translator.vocoder)

print('starting warmup, this can take up to 3.5 minutes.')
warmup_phrases = [
    # Very short (1–3 words)
    "Hello.",
    "Yes.",
    "No.",
    "Okay.",
    "Thank you.",
    "Good morning.",
    "Good night.",
    "See you.",
    "Alright.",
    "Sure.",

    # Short (4–8 words)
    "How are you today?",
    "I am fine, thank you.",
    "What time is it now?",
    "Please call me later.",
    "I need some help.",
    "Can you hear me?",
    "Turn on the lights.",
    "Close the door, please.",
    "Where is the station?",
    "I love this place.",

    # Medium (9–15 words)
    "Could you please tell me how to get to the nearest hospital?",
    "I will be there in about fifteen minutes, maybe a little longer.",
    "The weather today is sunny with a chance of light showers later.",
    "Please remember to bring your passport and boarding pass to the gate.",
    "I think we should try that new restaurant downtown for dinner tonight.",
    "Do you know if the library is open on Sundays during the summer?",
    "I have been learning English for three years and I really enjoy it.",
    "The quick brown fox jumps over the lazy dog every single morning.",
    "My phone battery is almost dead, can I borrow your charger please?",
    "We need to finish this project before the deadline next Friday afternoon.",

    # Long (16–30 words)
    "When I was a child, I used to spend my summers at my grandparents' house in the countryside, surrounded by fields and forests.",
    "Technology has advanced rapidly over the past decade, changing the way we communicate, work, and even think about the world around us.",
    "If you ever find yourself lost in a big city, the best thing to do is to stay calm and ask a local for directions.",
    "Learning a new language can be challenging, but it is also one of the most rewarding experiences you can have in your lifetime.",
    "The conference will cover topics such as artificial intelligence, renewable energy, and the future of space exploration over the next fifty years.",
    "Many people believe that breakfast is the most important meal of the day, providing the energy needed to start the morning right.",
    "In the middle of the night, I heard a strange noise coming from the kitchen, but when I checked, nothing was there.",
    "Traveling to different countries allows you to experience new cultures, try unique foods, and meet people from all walks of life.",
    "The old man told us a fascinating story about his adventures at sea during the war, full of danger, courage, and unexpected friendships.",
    "After months of preparation, the team finally launched their new product, hoping it would revolutionize the way people interact with technology.",

    # Very long (31–50 words)
    "Over the past few years, scientists have made incredible progress in the field of renewable energy, developing new technologies that harness the power of the sun, wind, and water to generate electricity in a cleaner and more sustainable way than ever before.",
    "As the sun set over the horizon, painting the sky in shades of orange and pink, the travelers gathered around the campfire, sharing stories of their journeys, their struggles, and the dreams that kept them moving forward despite the challenges they faced.",
    "In a world that is becoming increasingly interconnected, the ability to communicate effectively across cultures and languages is more important than ever, opening doors to new opportunities, fostering understanding, and building bridges between people from vastly different backgrounds and experiences."
]

total = 100
goal = 20
iteration = 0

while total > goal:
    a = time.perf_counter()
    for idx, phrase in enumerate(warmup_phrases, start=1):
        # Print progress on the same line
        print(
            f"Iteration {iteration}, sentence {idx}/{len(warmup_phrases)}: {phrase[:60]}...",
            end="\r",
            flush=True,
        )
        with torch.inference_mode():
            translator.predict(
                input=phrase,
                task_str="t2st",
                tgt_lang="hin",
                src_lang="eng"
            )
    total = time.perf_counter() - a
    iteration += 1
    # After finishing one full pass, print the timing result on a new line
    print(f"\n[Iteration {iteration}] warming up until the model takes {goal}s. Currently = {total:.2f}s")

print("translator loaded")

host = '0.0.0.0'
port = SEAMLESS_T2S_SERVER_PORT

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((host, port))
    s.listen()
    print(f"[Seamless Server] Listening on {host}:{port}")

    while True:
        conn, addr = s.accept()
        with conn:
            print(f"[Seamless Server] Connected by {addr}")

            try:
                # Receive request JSON
                data = conn.recv(4096).decode()
                req = json.loads(data)
                input_text = req["text"]
                tgt_lang = req["tgt_lang"]
            except Exception as e:
                print(f"[ERROR] Invalid request: {e}")
                conn.sendall(pickle.dumps({"error": "Invalid request"}))
                continue

            # Translation
            text_output, speech_output = translator.predict(
                input=input_text,
                task_str="t2st",
                tgt_lang=tgt_lang,
                src_lang="eng"
            )

            audio_tensor = speech_output.audio_wavs[0][0].to(torch.float32).cpu()
            sample_rate = speech_output.sample_rate

            # Package into dict
            # Package into dict including text output
            payload = {
                "sample_rate": sample_rate,
                "audio_tensor": audio_tensor,
                "text_output": str(text_output)  # <-- add text output
            }

            # Serialize with pickle
            data_bytes = pickle.dumps(payload)

            # Send length + payload
            conn.sendall(len(data_bytes).to_bytes(4, "big"))
            conn.sendall(data_bytes)

            print(f"[Seamless Server] Sent tensor with shape {audio_tensor.shape} @ {sample_rate} Hz")
            print(f"[Seamless Server] Text Output: {text_output}")
