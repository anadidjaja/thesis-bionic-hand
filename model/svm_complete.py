#!/usr/bin/env python3
"""SVM with raw per-channel normalization + 50 Hz notch + windowed features."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, pstdev

try:
    import numpy as np
    from scipy.signal import iirnotch, filtfilt
except Exception as e:  # pragma: no cover
    print("Missing dependency: scipy/numpy. Install with: pip install scipy numpy", file=sys.stderr)
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    print("Missing dependency: matplotlib. Install with: pip install matplotlib", file=sys.stderr)
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
except Exception as e:  # pragma: no cover
    print("Missing dependency: scikit-learn. Install with: pip install scikit-learn", file=sys.stderr)
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)


CHANNELS = ["CH1", "CH2", "CH3", "CH4"]
TIMESTAMP_COL = "timestamp"
CUT_SECONDS = 1.0
SAMPLE_RATE_HZ = 1000
WINDOW_MS = 200
STEP_MS = 100
WINDOW_SAMPLES = int(SAMPLE_RATE_HZ * WINDOW_MS / 1000)
STEP_SAMPLES = max(1, int(SAMPLE_RATE_HZ * STEP_MS / 1000))
MA_WINDOW = 5
SMOOTH_ALPHA = 0.2
FEATURE_NAMES = ["mav", "rms", "zcr", "wl", "ssc"]

NOTCH_HZ = 50.0
NOTCH_Q = 30.0


def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def apply_notch(signal_vals):
    if len(signal_vals) < 3:
        return signal_vals
    b, a = iirnotch(w0=NOTCH_HZ, Q=NOTCH_Q, fs=SAMPLE_RATE_HZ)
    return filtfilt(b, a, signal_vals)


def normalize_raw(ch_vals):
    arr = np.asarray(ch_vals, dtype=float)
    mu = arr.mean()
    sigma = arr.std()
    if sigma == 0:
        return arr.tolist()
    return ((arr - mu) / sigma).tolist()


def moving_average(ch_vals, window_size=MA_WINDOW):
    if window_size <= 1 or len(ch_vals) < window_size:
        return ch_vals
    arr = np.asarray(ch_vals, dtype=float)
    kernel = np.ones(window_size, dtype=float) / window_size
    smoothed = np.convolve(arr, kernel, mode="same")
    return smoothed.tolist()


def exponential_smooth(ch_vals, alpha=SMOOTH_ALPHA):
    if not ch_vals:
        return ch_vals
    smoothed = [float(ch_vals[0])]
    for x in ch_vals[1:]:
        smoothed.append(alpha * float(x) + (1.0 - alpha) * smoothed[-1])
    return smoothed


def preprocess_channel(ch_vals, use_ma, use_smooth):
    processed = ch_vals
    if use_ma:
        processed = moving_average(processed)
    if use_smooth:
        processed = exponential_smooth(processed)
    return processed


def window_features(vals, selected_features):
    n = len(vals)
    if n == 0:
        return None
    mav = sum(abs(v) for v in vals) / n
    rms = (sum(v * v for v in vals) / n) ** 0.5
    zcr = 0
    for i in range(1, n):
        if vals[i - 1] == 0:
            continue
        if (vals[i - 1] > 0 and vals[i] < 0) or (vals[i - 1] < 0 and vals[i] > 0):
            zcr += 1
    wl = 0.0
    for i in range(1, n):
        wl += abs(vals[i] - vals[i - 1])
    ssc = 0
    for i in range(1, n - 1):
        diff1 = vals[i] - vals[i - 1]
        diff2 = vals[i] - vals[i + 1]
        if diff1 * diff2 > 0:
            ssc += 1
    metric_map = {
        "mav": mav,
        "rms": rms,
        "zcr": zcr,
        "wl": wl,
        "ssc": ssc,
    }
    return [metric_map[name] for name in selected_features]


def extract_features(csv_path: Path, use_ma: bool, use_smooth: bool, selected_features):
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None
        for col in [TIMESTAMP_COL] + CHANNELS:
            if col not in reader.fieldnames:
                print(f"Skipping {csv_path}: missing column {col}", file=sys.stderr)
                return None

        timestamps = []
        channel_values = [[] for _ in CHANNELS]

        t0 = None
        for row in reader:
            ts_raw = row.get(TIMESTAMP_COL, "")
            if not ts_raw:
                continue
            try:
                ts = parse_iso(ts_raw)
            except Exception:
                continue
            if t0 is None:
                t0 = ts
            if ts < t0 + timedelta(seconds=CUT_SECONDS):
                continue

            vals = []
            bad = False
            for ch in CHANNELS:
                try:
                    vals.append(float(row[ch]))
                except Exception:
                    bad = True
                    break
            if bad:
                continue

            timestamps.append(ts)
            for idx, v in enumerate(vals):
                channel_values[idx].append(v)

    if not timestamps:
        print(f"Skipping {csv_path}: no data after 1s trim", file=sys.stderr)
        return None

    channel_values = [preprocess_channel(ch, use_ma, use_smooth) for ch in channel_values]
    channel_values = [normalize_raw(ch) for ch in channel_values]

    # Notch filter per channel
    filtered = []
    for ch_vals in channel_values:
        arr = np.asarray(ch_vals, dtype=float)
        arr_f = apply_notch(arr)
        filtered.append(arr_f.tolist())
    channel_values = filtered

    features = []
    for ch_vals in channel_values:
        if len(ch_vals) < WINDOW_SAMPLES:
            print(f"Skipping {csv_path}: not enough samples for windowing", file=sys.stderr)
            return None
        win_feats = []
        for start in range(0, len(ch_vals) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
            window = ch_vals[start:start + WINDOW_SAMPLES]
            wf = window_features(window, selected_features)
            if wf is not None:
                win_feats.append(wf)
        if not win_feats:
            print(f"Skipping {csv_path}: no windows after trim", file=sys.stderr)
            return None
        for k in range(len(win_feats[0])):
            col = [w[k] for w in win_feats]
            features.append(mean(col))
            features.append(pstdev(col))

    features.append(len(timestamps))
    features.append((timestamps[-1] - timestamps[0]).total_seconds())
    return features


def load_split(split_dir: Path, use_ma: bool, use_smooth: bool, selected_features):
    X, y = [], []
    for class_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        label = class_dir.name
        for csv_path in sorted(class_dir.glob("*.csv")):
            feat = extract_features(csv_path, use_ma, use_smooth, selected_features)
            if feat is None:
                continue
            X.append(feat)
            y.append(label)
    return X, y


def plot_confusion_heatmap(cm, labels, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="SVM with raw normalization + notch + windowed features")
    ap.add_argument("--data", default="data", help="Training root folder")
    ap.add_argument("--validation", default="validation", help="Validation root folder")
    ap.add_argument("--test", default="test", help="Test root folder")
    ap.add_argument("--kernel", default="rbf", choices=["linear", "rbf", "poly", "sigmoid"], help="SVM kernel")
    ap.add_argument("--C", default=1.0, type=float, help="SVM C parameter")
    ap.add_argument("--ma", action="store_true", help="Apply moving average before feature extraction")
    ap.add_argument("--smooth", action="store_true", help="Apply exponential smoothing before feature extraction")
    ap.add_argument(
        "--features",
        nargs="*",
        choices=FEATURE_NAMES,
        default=None,
        help="Selected window features. Default uses all: mav rms zcr wl ssc",
    )
    args = ap.parse_args()

    selected_features = args.features if args.features else FEATURE_NAMES

    data_dir = Path(args.data)
    val_dir = Path(args.validation)
    test_dir = Path(args.test)

    for d in [data_dir, val_dir, test_dir]:
        if not d.exists():
            print(f"Missing directory: {d}", file=sys.stderr)
            sys.exit(1)

    X_train, y_train = load_split(data_dir, args.ma, args.smooth, selected_features)
    X_val, y_val = load_split(val_dir, args.ma, args.smooth, selected_features)
    X_test, y_test = load_split(test_dir, args.ma, args.smooth, selected_features)

    if not X_train or not X_val or not X_test:
        print("One or more splits have no usable samples after trimming.", file=sys.stderr)
        sys.exit(1)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel=args.kernel, C=args.C)),
    ])
    model.fit(X_train, y_train)

    def evaluate(name, X, y):
        pred = model.predict(X)
        acc = accuracy_score(y, pred)
        p, r, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
        pred_counts = Counter(pred)
        true_counts = Counter(y)
        print(f"\n{name} accuracy: {acc:.4f}")
        print(f"{name} macro P/R/F1: {p:.4f} / {r:.4f} / {f1:.4f}")
        print(f"{name} prediction counts:")
        for label in sorted(set(list(true_counts.keys()) + list(pred_counts.keys()))):
            print(f"  {label}: predicted={pred_counts.get(label, 0)} true={true_counts.get(label, 0)}")
        print(f"{name} confusion matrix:")
        labels = sorted(set(y + list(pred)))
        print(f"Labels (rows=true, cols=pred): {labels}")
        cm = confusion_matrix(y, pred, labels=labels)
        print(cm)
        plot_confusion_heatmap(cm, labels, f"{name} confusion matrix")
        print(f"{name} classification report:")
        print(classification_report(y, pred))

    evaluate("Validation", X_val, y_val)
    evaluate("Test", X_test, y_test)


if __name__ == "__main__":
    main()
