import serial
import csv
import threading
import time
from datetime import datetime
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

PORT = "/dev/cu.usbserial-0001"
BAUD_RATE = 115200
OUTPUT_FILE = "open5.csv"
MAX_POINTS = 1000
RECORD_DURATION = 10  # seconds — change this to adjust recording length
NUM_CHANNELS = 4

COLORS = ["#00ff88", "#ff4444", "#4488ff", "#ffaa00"]
LABELS = ["CH1", "CH2", "CH3", "CH4"]

# shared data — one buffer per channel
voltage_buffers = [deque(maxlen=MAX_POINTS) for _ in range(NUM_CHANNELS)]
time_buffer = deque(maxlen=MAX_POINTS)
start_time = None
running = True
sample_count = 0

def serial_reader(ser, writer, csvfile):
    global start_time, running, sample_count
    start_time = time.time()
    while running:
        try:
            line = ser.readline().decode("utf-8").strip()
            if line:
                try:
                    values = line.split(",")
                    if len(values) == NUM_CHANNELS:
                        voltages = [float(v) for v in values]
                        timestamp = datetime.now().isoformat()
                        elapsed = time.time() - start_time
                        time_buffer.append(elapsed)
                        for i, v in enumerate(voltages):
                            voltage_buffers[i].append(v)
                        writer.writerow([timestamp] + voltages)
                        csvfile.flush()
                        sample_count += 1
                except ValueError:
                    pass
        except Exception:
            break

def main():
    global running

    print(f"Opening {PORT} at {BAUD_RATE} baud...")

    csvfile = open(OUTPUT_FILE, "w", newline="")
    writer = csv.writer(csvfile)
    writer.writerow(["timestamp"] + LABELS)  # header: timestamp, CH1, CH2, CH3, CH4

    ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
    print(f"Recording to {OUTPUT_FILE} — auto stops after {RECORD_DURATION}s")

    thread = threading.Thread(target=serial_reader, args=(ser, writer, csvfile), daemon=True)
    thread.start()

    # ── plot setup ──────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 7), facecolor="#0d0d0d")
    fig.canvas.manager.set_window_title("EMG Live Monitor — 4 Channels")

    gs = GridSpec(NUM_CHANNELS, 1, figure=fig, hspace=0.4)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(NUM_CHANNELS)]
    line_plots = []

    for i, ax in enumerate(axes):
        ax.set_facecolor("#0d0d0d")
        for spine in ax.spines.values():
            spine.set_color("#333333")
        ax.tick_params(colors="#888888", labelsize=7)
        ax.set_ylim(-0.1, 5.5)
        ax.set_ylabel(LABELS[i], fontsize=9, color=COLORS[i])
        ax.grid(True, color="#1e1e1e", linewidth=0.8)
        if i < NUM_CHANNELS - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (s)", fontsize=9, color="#888888")
        line, = ax.plot([], [], color=COLORS[i], linewidth=0.8, alpha=0.9)
        line_plots.append(line)

    timer_text = axes[0].text(
        0.01, 0.85, "⏺  00:00",
        transform=axes[0].transAxes,
        fontsize=11, color="#ff4444",
        fontfamily="monospace", va="top", ha="left"
    )

    sample_text = axes[0].text(
        0.99, 0.85, "0 samples",
        transform=axes[0].transAxes,
        fontsize=9, color="#555555",
        fontfamily="monospace", va="top", ha="right"
    )

    countdown_text = axes[0].text(
        0.5, 0.85, f"stops in {RECORD_DURATION}s",
        transform=axes[0].transAxes,
        fontsize=9, color="#888888",
        fontfamily="monospace", va="top", ha="center"
    )

    def update(frame):
        global running

        if not time_buffer:
            return line_plots + [timer_text, sample_text, countdown_text]

        t = list(time_buffer)

        for i, ax in enumerate(axes):
            v = list(voltage_buffers[i])
            line_plots[i].set_data(t, v)
            ax.set_xlim(max(0, t[-1] - 10), t[-1] + 0.5)

        if start_time:
            elapsed = time.time() - start_time
            elapsed_int = int(elapsed)
            remaining = max(0, RECORD_DURATION - elapsed_int)

            mins, secs = divmod(elapsed_int, 60)
            timer_text.set_text(f"⏺  {mins:02d}:{secs:02d}")

            if remaining > 0:
                countdown_text.set_text(f"stops in {remaining}s")
                countdown_text.set_color("#888888")
            else:
                countdown_text.set_text("recording complete")
                countdown_text.set_color("#00ff88")
                timer_text.set_color("#00ff88")

            if elapsed >= RECORD_DURATION and running:
                running = False
                plt.close(fig)

        sample_text.set_text(f"{sample_count:,} samples")

        return line_plots + [timer_text, sample_text, countdown_text]

    ani = animation.FuncAnimation(
        fig, update, interval=50, blit=True, cache_frame_data=False
    )

    plt.tight_layout()
    try:
        plt.show()
    finally:
        running = False
        ser.close()
        csvfile.close()
        print(f"\nStopped. {sample_count} samples saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()