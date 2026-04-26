#!/usr/bin/env python3
"""Visualize MLP architecture and weights for selected NN pipelines."""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean

try:
    import numpy as np
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    print("Missing dependency: numpy/matplotlib. Install with: pip install numpy matplotlib", file=sys.stderr)
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
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


def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def normalize_raw(ch_vals):
    arr = np.asarray(ch_vals, dtype=float)
    mu = arr.mean()
    sigma = arr.std()
    if sigma == 0:
        return arr.tolist()
    return ((arr - mu) / sigma).tolist()


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


def load_csv_raw_windows(csv_path: Path, normalize: bool):
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

    if normalize:
        channel_values = [normalize_raw(ch) for ch in channel_values]

    if any(len(ch) < WINDOW_SAMPLES for ch in channel_values):
        return None

    windows = []
    for start in range(0, len(channel_values[0]) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
        feats = []
        for ch_vals in channel_values:
            window = ch_vals[start:start + WINDOW_SAMPLES]
            feats.extend(window)
        windows.append(feats)
    return windows if windows else None


def load_csv_feature_windows(csv_path: Path):
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

    if any(len(ch) < WINDOW_SAMPLES for ch in channel_values):
        return None

    windows = []
    for start in range(0, len(channel_values[0]) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
        feats = []
        for ch_vals in channel_values:
            window = ch_vals[start:start + WINDOW_SAMPLES]
            wf = window_features(window)
            if wf is None:
                feats = None
                break
            feats.extend(wf)
        if feats is not None:
            windows.append(feats)
    return windows if windows else None


def load_split(split_dir: Path, mode: str):
    X, y = [], []
    for class_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        label = class_dir.name
        for csv_path in sorted(class_dir.glob("*.csv")):
            if mode == "simple_notnorm":
                windows = load_csv_raw_windows(csv_path, normalize=False)
            elif mode == "simple_normraw":
                windows = load_csv_raw_windows(csv_path, normalize=True)
            else:  # complete_norm
                windows = load_csv_feature_windows(csv_path)
            if windows is None:
                continue
            for w in windows:
                X.append(w)
                y.append(label)
    return X, y


def plot_architecture(layer_sizes, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    x = 0.1
    for i, size in enumerate(layer_sizes):
        ax.add_patch(plt.Rectangle((x, 0.25), 0.18, 0.5, fill=False, lw=1.5))
        ax.text(x + 0.09, 0.5, f"{size}", ha="center", va="center", fontsize=10)
        ax.text(x + 0.09, 0.2, f"Layer {i}", ha="center", va="center", fontsize=8)
        x += 0.22
    ax.set_title(title)
    plt.tight_layout()


def plot_weights(coefs, title):
    n = len(coefs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for i, w in enumerate(coefs):
        ax = axes[i]
        im = ax.imshow(w, aspect="auto", cmap="coolwarm")
        ax.set_title(f"Weights L{i}→L{i+1}")
        ax.set_xlabel("Units (next layer)")
        ax.set_ylabel("Units (prev layer)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(title)
    plt.tight_layout()


def main():
    ap = argparse.ArgumentParser(description="Visualize MLP architecture and weights")
    ap.add_argument("--mode", choices=["simple_notnorm", "complete_norm"], required=True,
                    help="simple_notnorm = not_normalized_model/nn_simple.py, complete_norm = normalized_model/nn_complete.py")
    ap.add_argument("--data", default="data", help="Training root folder")
    ap.add_argument("--hidden", default="128,64", help="Hidden layer sizes, comma-separated")
    ap.add_argument("--max-iter", default=300, type=int, help="Max training iterations")
    ap.add_argument("--alpha", default=0.0001, type=float, help="L2 regularization")
    args = ap.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"Missing directory: {data_dir}", file=sys.stderr)
        sys.exit(1)

    X_train, y_train = load_split(data_dir, args.mode)
    if not X_train:
        print("No training samples found after processing.", file=sys.stderr)
        sys.exit(1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    hidden_sizes = tuple(int(x.strip()) for x in args.hidden.split(",") if x.strip())
    model = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        max_iter=args.max_iter,
        alpha=args.alpha,
        random_state=42,
    )
    model.fit(X_train, y_train)

    layer_sizes = [X_train.shape[1], *hidden_sizes, len(model.classes_)]
    plot_architecture(layer_sizes, f"MLP Architecture ({args.mode})")
    plot_weights(model.coefs_, f"MLP Weights ({args.mode})")
    plt.show()


if __name__ == "__main__":
    main()
