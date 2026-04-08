#!/usr/bin/env python3
"""Plot SVM decision boundaries in PCA space for selected pipelines."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, median, pstdev

try:
    import numpy as np
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    print("Missing dependency: numpy/matplotlib. Install with: pip install numpy matplotlib", file=sys.stderr)
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
except Exception as e:  # pragma: no cover
    print("Missing dependency: scikit-learn. Install with: pip install scikit-learn", file=sys.stderr)
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)


CHANNELS = ["CH1", "CH2", "CH3", "CH4"]
TIMESTAMP_COL = "timestamp"
CUT_SECONDS = 1.0

# Params matching normalized_model/svm_complete.py
SAMPLE_RATE_HZ = 1000
WINDOW_MS = 200
STEP_MS = 100
WINDOW_SAMPLES = int(SAMPLE_RATE_HZ * WINDOW_MS / 1000)
STEP_SAMPLES = max(1, int(SAMPLE_RATE_HZ * STEP_MS / 1000))
NOTCH_HZ = 50.0
NOTCH_Q = 30.0


def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def extract_features_simple(csv_path: Path):
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None
        for col in [TIMESTAMP_COL] + CHANNELS:
            if col not in reader.fieldnames:
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
        return None

    features = []
    for ch_vals in channel_values:
        if not ch_vals:
            return None
        features.extend([mean(ch_vals), pstdev(ch_vals), min(ch_vals), max(ch_vals), median(ch_vals)])
    features.append(len(timestamps))
    features.append((timestamps[-1] - timestamps[0]).total_seconds())
    return features


def normalize_raw(ch_vals):
    arr = np.asarray(ch_vals, dtype=float)
    mu = arr.mean()
    sigma = arr.std()
    if sigma == 0:
        return arr.tolist()
    return ((arr - mu) / sigma).tolist()


def apply_notch(signal_vals):
    try:
        from scipy.signal import iirnotch, filtfilt
    except Exception as e:  # pragma: no cover
        print("Missing dependency: scipy. Install with: pip install scipy", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        sys.exit(1)
    if len(signal_vals) < 3:
        return signal_vals
    b, a = iirnotch(w0=NOTCH_HZ, Q=NOTCH_Q, fs=SAMPLE_RATE_HZ)
    return filtfilt(b, a, signal_vals)


def window_features(vals):
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
    return [mav, rms, zcr, wl, ssc]


def extract_features_complete_norm(csv_path: Path):
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None
        for col in [TIMESTAMP_COL] + CHANNELS:
            if col not in reader.fieldnames:
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
        return None

    channel_values = [normalize_raw(ch) for ch in channel_values]
    filtered = []
    for ch_vals in channel_values:
        arr = np.asarray(ch_vals, dtype=float)
        arr_f = apply_notch(arr)
        filtered.append(arr_f.tolist())
    channel_values = filtered

    features = []
    for ch_vals in channel_values:
        if len(ch_vals) < WINDOW_SAMPLES:
            return None
        win_feats = []
        for start in range(0, len(ch_vals) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
            window = ch_vals[start:start + WINDOW_SAMPLES]
            wf = window_features(window)
            if wf is not None:
                win_feats.append(wf)
        if not win_feats:
            return None
        for k in range(len(win_feats[0])):
            col = [w[k] for w in win_feats]
            features.append(mean(col))
            features.append(pstdev(col))

    features.append(len(timestamps))
    features.append((timestamps[-1] - timestamps[0]).total_seconds())
    return features


def load_split(split_dir: Path, mode: str):
    X, y = [], []
    for class_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        label = class_dir.name
        for csv_path in sorted(class_dir.glob("*.csv")):
            if mode == "simple":
                feat = extract_features_simple(csv_path)
            else:
                feat = extract_features_complete_norm(csv_path)
            if feat is None:
                continue
            X.append(feat)
            y.append(label)
    return X, y


def main():
    ap = argparse.ArgumentParser(description="Plot SVM decision boundaries in PCA space")
    ap.add_argument("--mode", choices=["simple", "complete_norm"], required=True,
                    help="simple = not_normalized_model/svm_simple.py, complete_norm = normalized_model/svm_complete.py")
    ap.add_argument("--data", default="data", help="Training root folder")
    ap.add_argument("--validation", default="validation", help="Validation root folder")
    ap.add_argument("--test", default="test", help="Test root folder")
    ap.add_argument("--kernel", default="rbf", choices=["linear", "rbf", "poly", "sigmoid"])
    ap.add_argument("--C", default=1.0, type=float)
    ap.add_argument("--grid", default=250, type=int, help="Grid resolution for decision boundary")
    ap.add_argument("--no-boundary", action="store_true", help="Disable decision boundary plot")
    ap.add_argument("--show-support", action="store_true", help="Show support vectors")
    args = ap.parse_args()

    data_dir = Path(args.data)
    val_dir = Path(args.validation)
    test_dir = Path(args.test)
    for d in [data_dir, val_dir, test_dir]:
        if not d.exists():
            print(f"Missing directory: {d}", file=sys.stderr)
            sys.exit(1)

    X_train, y_train = load_split(data_dir, args.mode)
    X_val, y_val = load_split(val_dir, args.mode)
    X_test, y_test = load_split(test_dir, args.mode)

    if not X_train:
        print("No training samples found after processing.", file=sys.stderr)
        sys.exit(1)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val) if X_val else []
    X_test_s = scaler.transform(X_test) if X_test else []

    svm = SVC(kernel=args.kernel, C=args.C)
    svm.fit(X_train_s, y_train)

    pca = PCA(n_components=2, random_state=42)
    X_train_p = pca.fit_transform(X_train_s)
    X_val_p = pca.transform(X_val_s) if len(X_val_s) else np.empty((0, 2))
    X_test_p = pca.transform(X_test_s) if len(X_test_s) else np.empty((0, 2))

    classes = sorted(set(y_train + y_val + y_test))
    color_map = {c: plt.cm.tab10(i % 10) for i, c in enumerate(classes)}

    plt.figure(figsize=(10, 7))
    if not args.no_boundary:
        all_p = np.vstack([X_train_p, X_val_p, X_test_p]) if len(X_val_p) or len(X_test_p) else X_train_p
        x_min, x_max = all_p[:, 0].min() - 1.0, all_p[:, 0].max() + 1.0
        y_min, y_max = all_p[:, 1].min() - 1.0, all_p[:, 1].max() + 1.0
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, args.grid),
            np.linspace(y_min, y_max, args.grid),
        )
        grid_p = np.c_[xx.ravel(), yy.ravel()]
        grid_s = pca.inverse_transform(grid_p)
        zz = svm.predict(grid_s).reshape(xx.shape)
        # Plot decision regions
        for cls in classes:
            mask = (zz == cls)
            plt.contourf(xx, yy, mask, levels=[0.5, 1.5], alpha=0.08, colors=[color_map[cls]])

    def scatter(points, labels, marker, name):
        for cls in classes:
            idx = [i for i, y in enumerate(labels) if y == cls]
            if not idx:
                continue
            pts = points[idx]
            plt.scatter(pts[:, 0], pts[:, 1], s=28, marker=marker, color=color_map[cls], label=f"{name}:{cls}")

    scatter(X_train_p, y_train, "o", "train")
    if len(X_val_p):
        scatter(X_val_p, y_val, "^", "val")
    if len(X_test_p):
        scatter(X_test_p, y_test, "s", "test")

    if args.show_support:
        sv_p = pca.transform(svm.support_vectors_)
        plt.scatter(sv_p[:, 0], sv_p[:, 1], s=90, facecolors="none", edgecolors="k", linewidths=1.2, label="support")

    plt.title(f"SVM PCA view ({args.mode}, kernel={args.kernel})")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
