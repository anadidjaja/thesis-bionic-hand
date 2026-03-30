import csv
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

path = "readings.csv"  # Default path to the CSV file

def read_csv(path):
    times = []
    ecg = []
    base_time = None
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("timestamp", "")
            v = row.get("voltage", "")
            if t == "" or v == "":
                continue
            try:
                value = float(v)
            except ValueError:
                continue
            if "T" in t:
                try:
                    dt = datetime.fromisoformat(t)
                except ValueError:
                    continue
                if base_time is None:
                    base_time = dt
                times.append((dt - base_time).total_seconds() * 1000.0)
            else:
                try:
                    times.append(float(t))
                except ValueError:
                    continue
            ecg.append(value)
    return times, ecg

def main():
    parser = argparse.ArgumentParser(description="Plot ECG CSV")
    parser.add_argument("path", nargs="?", default=path, help="CSV file path")
    args = parser.parse_args()

    times, ecg = read_csv(args.path)
    if not times:
        raise SystemExit("No valid data found in CSV.")

    plt.plot(times, ecg)
    plt.xlabel("Time (ms)")
    plt.ylabel("ECG Value")
    plt.title("ECG from CSV")
    plt.show()

if __name__ == "__main__":
    main()
