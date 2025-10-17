import requests
from threading import Thread
total_times = {}

def add_time_and_print(time, name):
    global total_times
    n_calls = 1
    if(name in total_times):
        time += total_times[name][0]
        n_calls += total_times[name][1]
    total_times[name] = (time, n_calls)

    print(f"{name}: total time = {time}, calls = {n_calls}")

def log_to_server(type, location, session, quantity):
    def _send():
        try:
            requests.post(
                "http://127.0.0.1:4567/message",
                json={"type":type, "location": location, "session": session, "quantity": quantity},
                timeout=0.5
            )
        except Exception as e:
            print(f"Logging failed: {e}")
    Thread(target=_send, daemon=True).start()