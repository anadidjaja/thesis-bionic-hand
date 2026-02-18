import os
import serial
import time
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import re

PORT = "/dev/cu.usbserial-0001"
BAUD = 115200
DURATION = 60   # seconds
DEBUG = True

OUTPUT_DIR = "buka_tutup_telapak/ulnaris"   
# OUTPUT_DIR = "angkat_lengan"  
os.makedirs(OUTPUT_DIR, exist_ok=True)

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)
ser.reset_input_buffer()

start_time = time.time()
start_perf = time.perf_counter()
times_s, ecg = [], []
total_lines = 0
parsed_lines = 0
skipped_lines = 0

plt.ion()
fig, ax = plt.subplots()
line_plot, = ax.plot([], [])
ax.set_xlabel("Time (s)")
ax.set_ylabel("ECG Value")
ax.set_title("Recorded ECG (Live)")
ax.xaxis.set_major_locator(MultipleLocator(1))
time_text = ax.text(
    0.02,
    0.95,
    "Time: 0.00 s",
    transform=ax.transAxes,
    va="top",
    ha="left",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
)

def update_plot():
    if not times_s:
        return
    line_plot.set_data(times_s, ecg)
    time_text.set_text(f"Time: {times_s[-1]:.2f} s")
    ax.relim()
    ax.autoscale_view(scalex=False)
    ax.set_xlim(times_s[0], times_s[-1] + 5)
    fig.canvas.draw()
    fig.canvas.flush_events()

csv_path = os.path.join(OUTPUT_DIR, "ulnaris.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time_ms", "device_time_ms", "ecg", "upper", "lower"])

    while time.time() - start_time < DURATION:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue
        total_lines += 1

        try:
            # Arduino output often isn't exactly "t,v,u,l".
            # Extract numbers so formats like "ECG: 512" or "123, 456" still parse.
            nums = re.findall(r"-?\d+(?:\.\d+)?", line)
            host_time_ms = int((time.perf_counter() - start_perf) * 1000)
            device_time_ms = ""

            if len(nums) == 4:
                device_time_ms, v, u, l = map(int, map(float, nums))
            elif len(nums) == 3:
                v, u, l = map(int, map(float, nums))
            elif len(nums) == 2:
                v, u = map(int, map(float, nums))
                l = ""
            elif len(nums) == 1:
                v = int(float(nums[0]))
                u, l = "", ""
            else:
                raise ValueError("no numeric fields")

            writer.writerow([host_time_ms, device_time_ms, v, u, l])
            times_s.append(host_time_ms / 1000.0)
            ecg.append(v)
            parsed_lines += 1
            if parsed_lines % 10 == 0:
                update_plot()
        except ValueError:
            skipped_lines += 1
            if DEBUG and skipped_lines <= 20:
                print(f"Skipped line (unparsed): {line!r}")
            continue

ser.close()

if DEBUG:
    print(f"Read {total_lines} non-empty lines; parsed {parsed_lines}; skipped {skipped_lines}.")

plt.ioff()
update_plot()
plt.show()
