import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

# -----------------------
# Parameters (safe defaults for your setup)
# -----------------------
LOW_HZ = 20.0
HIGH_HZ = 120.0
WINDOW_SEC = 0.5       # 500 ms windows
OVERLAP = 0.5          # 50% overlap


# -----------------------
# CSV utilities
# -----------------------
def read_csv(path, time_col, value_col):
    times = []
    values = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if time_col not in fieldnames or value_col not in fieldnames:
            raise ValueError(
                "CSV columns not found. "
                f"Requested time_col={time_col!r}, value_col={value_col!r}. "
                f"Available columns: {fieldnames}"
            )
        for row in reader:
            try:
                times.append(float(row[time_col]))
                values.append(float(row[value_col]))
            except (KeyError, ValueError):
                continue
    return np.array(times), np.array(values)


def estimate_fs(time_ms):
    diffs = np.diff(time_ms)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        raise ValueError("Cannot estimate sampling rate.")
    return 1000.0 / np.median(diffs)


# -----------------------
# Signal processing
# -----------------------
def bandpass_filter(x, fs, low, high, order=4):
    nyq = fs / 2.0
    if high >= nyq or low >= high:
        raise ValueError("Invalid band for Nyquist.")
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)


def extract_features(signal, fs):
    freqs, psd = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=len(signal),
        noverlap=0
    )

    power = np.sum(psd)
    mean_freq = np.sum(freqs * psd) / power

    cumulative = np.cumsum(psd)
    median_freq = freqs[np.searchsorted(cumulative, power / 2)]

    rms = np.sqrt(np.mean(signal ** 2))

    return rms, mean_freq, median_freq, power


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="sEMG processing pipeline")
    parser.add_argument("input", help="Raw CSV input")
    parser.add_argument("output", help="Processed CSV output")
    parser.add_argument("--time-col", default="time_ms")
    parser.add_argument("--value-col", default="ecg")
    parser.add_argument("--low-hz", type=float, default=LOW_HZ)
    parser.add_argument("--high-hz", type=float, default=HIGH_HZ)
    args = parser.parse_args()

    times, values = read_csv(args.input, args.time_col, args.value_col)

    fs = estimate_fs(times)
    print(f"Estimated Fs: {fs:.2f} Hz")

    # Remove DC
    values = values - np.mean(values)

    # Time-domain EMG filter (auto-adjust if sampling rate is low)
    nyq = fs / 2.0
    low = args.low_hz
    high = args.high_hz
    if high >= nyq:
        high = max(nyq * 0.9, 0.5)
        print(f"Adjusted high cutoff to {high:.2f} Hz (Nyquist {nyq:.2f} Hz)")
    if low >= high:
        low = max(min(0.5, high * 0.2), 0.1)
        print(f"Adjusted low cutoff to {low:.2f} Hz to keep a valid band")

    try:
        filtered = bandpass_filter(values, fs, low, high)
    except ValueError:
        print("Bandpass invalid for this sampling rate; skipping filter.")
        filtered = values

    # Windowing
    win_len = int(WINDOW_SEC * fs)
    step = int(win_len * (1 - OVERLAP))

    rows = []
    for start in range(0, len(filtered) - win_len, step):
        segment = filtered[start:start + win_len]
        t_center = times[start + win_len // 2] / 1000.0

        rms, mnf, mdf, power = extract_features(segment, fs)
        rows.append([t_center, rms, mnf, mdf, power])

    # Save processed CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "rms", "mean_freq", "median_freq", "band_power"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} windows to {args.output}")

    if not rows:
        print("No windows to plot.")
        return

    # Plot features over time
    arr = np.array(rows)
    t = arr[:, 0]
    rms = arr[:, 1]
    mnf = arr[:, 2]
    mdf = arr[:, 3]
    power = arr[:, 4]

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
    axes[0].plot(t, rms, color="tab:blue")
    axes[0].set_ylabel("RMS")
    axes[1].plot(t, mnf, color="tab:orange")
    axes[1].set_ylabel("Mean Freq (Hz)")
    axes[2].plot(t, mdf, color="tab:green")
    axes[2].set_ylabel("Median Freq (Hz)")
    axes[3].plot(t, power, color="tab:red")
    axes[3].set_ylabel("Band Power")
    axes[3].set_xlabel("Time (s)")

    fig.suptitle("sEMG Features Over Time")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
