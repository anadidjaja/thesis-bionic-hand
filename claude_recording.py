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
OUTPUT_FILE = "1zzzzz.csv"
MAX_POINTS = 1000
RECORD_DURATION = 500 # seconds — change this to adjust recording length

# shared data
voltage_buffer = deque(maxlen=MAX_POINTS)
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
                    voltage = float(line)
                    timestamp = datetime.now().isoformat()
                    elapsed = time.time() - start_time
                    voltage_buffer.append(voltage)
                    time_buffer.append(elapsed)
                    writer.writerow([timestamp, voltage])
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
    writer.writerow(["timestamp", "voltage"])

    ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
    print(f"Recording to {OUTPUT_FILE} — auto stops after {RECORD_DURATION}s")

    thread = threading.Thread(target=serial_reader, args=(ser, writer, csvfile), daemon=True)
    thread.start()

    # ── plot setup ──────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(12, 5), facecolor="#0d0d0d")
    fig.canvas.manager.set_window_title("EMG Live Monitor")

    gs = GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor("#0d0d0d")

    for spine in ax.spines.values():
        spine.set_color("#333333")

    ax.tick_params(colors="#888888")
    ax.yaxis.label.set_color("#888888")
    ax.xaxis.label.set_color("#888888")
    ax.set_ylim(-0.1, 5.5)
    ax.set_ylabel("Voltage (V)", fontsize=10, color="#888888")
    ax.set_xlabel("Time (s)", fontsize=10, color="#888888")
    ax.grid(True, color="#1e1e1e", linewidth=0.8)

    line_plot, = ax.plot([], [], color="#00ff88", linewidth=1.0, alpha=0.9)

    # recording timer — top left
    timer_text = ax.text(
        0.01, 0.96, "⏺  00:00",
        transform=ax.transAxes,
        fontsize=13,
        color="#ff4444",
        fontfamily="monospace",
        va="top", ha="left"
    )

    # sample counter — top right
    sample_text = ax.text(
        0.99, 0.96, "0 samples",
        transform=ax.transAxes,
        fontsize=10,
        color="#555555",
        fontfamily="monospace",
        va="top", ha="right"
    )

    # countdown — top center
    countdown_text = ax.text(
        0.5, 0.96, f"stops in {RECORD_DURATION}s",
        transform=ax.transAxes,
        fontsize=10,
        color="#888888",
        fontfamily="monospace",
        va="top", ha="center"
    )

    def update(frame):
        global running

        if not time_buffer:
            return line_plot, timer_text, sample_text, countdown_text

        t = list(time_buffer)
        v = list(voltage_buffer)

        line_plot.set_data(t, v)
        ax.set_xlim(max(0, t[-1] - 10), t[-1] + 0.5)

        if start_time:
            elapsed = time.time() - start_time
            elapsed_int = int(elapsed)
            remaining = max(0, RECORD_DURATION - elapsed_int)

            # update recording timer
            mins, secs = divmod(elapsed_int, 60)
            timer_text.set_text(f"⏺  {mins:02d}:{secs:02d}")

            # update countdown
            if remaining > 0:
                countdown_text.set_text(f"stops in {remaining}s")
                countdown_text.set_color("#888888")
            else:
                countdown_text.set_text("recording complete")
                countdown_text.set_color("#00ff88")
                timer_text.set_color("#00ff88")  # turn timer green when done

            # auto stop after RECORD_DURATION seconds
            if elapsed >= RECORD_DURATION and running:
                running = False
                plt.close(fig)

        sample_text.set_text(f"{sample_count:,} samples")

        return line_plot, timer_text, sample_text, countdown_text

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