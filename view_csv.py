import csv
import argparse
import matplotlib.pyplot as plt

path = "buka_tutup_telapak/ulnaris/ulnaris.csv"

def read_csv(path):
    times = []
    ecg = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("time_ms", "")
            v = row.get("ecg", "")
            if t == "" or v == "":
                continue
            try:
                times.append(float(t))
                ecg.append(float(v))
            except ValueError:
                continue
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
