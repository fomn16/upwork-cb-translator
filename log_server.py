from flask import Flask, request
import time
from datetime import datetime
from threading import Lock, Thread
import signal
import sys
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as pyo

app = Flask(__name__)

data_store = {}
data_lock = Lock()

plt.ion()
fig, ax_dict = None, {}

# --------------------------------------------------
# Flask endpoint
# --------------------------------------------------
@app.route("/message", methods=["POST"])
def receive_message():
    data = request.get_json()
    location = data.get("location", "")
    session = data.get("session", "")
    quantity = float(data.get("quantity", 0))
    event_type = data.get("type", "default")

    key = (event_type, location, session)
    current_time = datetime.now()

    with data_lock:
        if key not in data_store:
            data_store[key] = {
                "times": [],
                "quantities": [],
                "cumulative": 0.0,
                "type": event_type,
            }
        entry = data_store[key]
        entry["cumulative"] += quantity
        entry["times"].append(current_time)
        entry["quantities"].append(entry["cumulative"])

    return ("OK", 200)


# --------------------------------------------------
# Live plot loop (Matplotlib)
# --------------------------------------------------
def plot_loop():
    global fig, ax_dict
    while True:
        with data_lock:
            event_types = sorted(
                set(entry["type"] for entry in data_store.values())
            )
            n_types = len(event_types) if event_types else 1

            # Create subplots as needed
            if fig is None or len(ax_dict) != n_types:
                fig, axes = plt.subplots(
                    n_types, 1, figsize=(9, 4 * n_types), squeeze=False
                )
                ax_dict = {t: axes[i, 0] for i, t in enumerate(event_types or ["default"])}

            for event_type in event_types:
                ax = ax_dict[event_type]
                ax.clear()
                ax.set_title(f"Type: {event_type}")
                ax.set_xlabel("Time")
                ax.set_ylabel("Cumulative Quantity")

                for (etype, location, session), data in data_store.items():
                    if etype == event_type:
                        ax.plot(
                            data["times"],
                            data["quantities"],
                            label=f"{location}-{session}",
                        )

                ax.legend()
                fig.autofmt_xdate()

        plt.pause(0.01)
        time.sleep(1)


# --------------------------------------------------
# Save interactive HTML on exit
# --------------------------------------------------
def save_interactive_html(*args):
    """Save all plots to one interactive HTML file on exit."""
    print("\nSaving final plot to 'final_plot.html' ...")

    with data_lock:
        if not data_store:
            print("No data recorded.")
            sys.exit(0)

        # Group traces by event type
        fig_dict = {}
        for (etype, location, session), data in data_store.items():
            trace = go.Scatter(
                x=data["times"],
                y=data["quantities"],
                mode="lines+markers",
                name=f"{location}-{session}",
            )
            if etype not in fig_dict:
                fig_dict[etype] = []
            fig_dict[etype].append(trace)

        # One subplot per type
        rows = len(fig_dict)
        all_figs = []
        for idx, (etype, traces) in enumerate(fig_dict.items(), 1):
            subfig = go.Figure(traces)
            subfig.update_layout(
                title=f"Type: {etype}",
                xaxis_title="Time",
                yaxis_title="Cumulative Quantity",
            )
            all_figs.append(subfig)

        # Combine all figures into an HTML page
        html_parts = []
        for subfig in all_figs:
            html_parts.append(pyo.plot(subfig, include_plotlyjs=False, output_type="div"))

        html_page = (
            "<html><head>"
            "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>"
            "</head><body>"
            + "<hr>".join(html_parts)
            + "</body></html>"
        )

        with open("final_plot.html", "w", encoding="utf-8") as f:
            f.write(html_page)

        print("Saved interactive file as 'final_plot.html'.")

    sys.exit(0)


signal.signal(signal.SIGINT, save_interactive_html)
signal.signal(signal.SIGTERM, save_interactive_html)


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    server_thread = Thread(
        target=lambda: app.run(
            host="127.0.0.1", port=4567, debug=False, use_reloader=False
        ),
        daemon=True,
    )
    server_thread.start()

    print("Server running. Plotting live updates...")
    plot_loop()