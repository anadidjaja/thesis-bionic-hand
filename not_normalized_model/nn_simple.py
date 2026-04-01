#!/usr/bin/env python3
"""Window-level MLP using raw samples (no feature extraction/filtering).

Uses 200 ms windows, 100 ms step at 1000 Hz, after trimming first 1s.
Each window is flattened (CH1..CH4 concatenated) as input.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

try:
    import numpy as np
except Exception as e:  # pragma: no cover
    print("Missing dependency: numpy. Install with: pip install numpy", file=sys.stderr)
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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


def extract_windows(csv_path: Path):
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

    if any(len(ch) < WINDOW_SAMPLES for ch in channel_values):
        print(f"Skipping {csv_path}: not enough samples for windowing", file=sys.stderr)
        return None

    windows = []
    for start in range(0, len(channel_values[0]) - WINDOW_SAMPLES + 1, STEP_SAMPLES):
        feats = []
        for ch_vals in channel_values:
            window = ch_vals[start:start + WINDOW_SAMPLES]
            feats.extend(window)
        windows.append(feats)

    if not windows:
        print(f"Skipping {csv_path}: no windows after trim", file=sys.stderr)
        return None

    return windows


def load_split(split_dir: Path):
    X, y, file_ids = [], [], []
    file_to_label = {}
    for class_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        label = class_dir.name
        for csv_path in sorted(class_dir.glob("*.csv")):
            windows = extract_windows(csv_path)
            if windows is None:
                continue
            file_id = str(csv_path)
            file_to_label[file_id] = label
            for w in windows:
                X.append(w)
                y.append(label)
                file_ids.append(file_id)
    return X, y, file_ids, file_to_label


def aggregate_predictions(file_ids, pred_labels):
    votes = defaultdict(list)
    for fid, pred in zip(file_ids, pred_labels):
        votes[fid].append(pred)
    file_pred = {}
    for fid, preds in votes.items():
        most_common = Counter(preds).most_common(1)[0][0]
        file_pred[fid] = most_common
    return file_pred


def main():
    ap = argparse.ArgumentParser(description="Window-level MLP on raw samples with file-level aggregation")
    ap.add_argument("--data", default="data", help="Training root folder")
    ap.add_argument("--validation", default="validation", help="Validation root folder")
    ap.add_argument("--test", default="test", help="Test root folder")
    ap.add_argument("--hidden", default="128,64", help="Hidden layer sizes, comma-separated")
    ap.add_argument("--max-iter", default=300, type=int, help="Max training iterations")
    ap.add_argument("--alpha", default=0.0001, type=float, help="L2 regularization")
    args = ap.parse_args()

    data_dir = Path(args.data)
    val_dir = Path(args.validation)
    test_dir = Path(args.test)

    for d in [data_dir, val_dir, test_dir]:
        if not d.exists():
            print(f"Missing directory: {d}", file=sys.stderr)
            sys.exit(1)

    X_train, y_train, _, _ = load_split(data_dir)
    X_val, y_val, val_file_ids, val_file_labels = load_split(val_dir)
    X_test, y_test, test_file_ids, test_file_labels = load_split(test_dir)

    if not X_train or not X_val or not X_test:
        print("One or more splits have no usable samples after trimming.", file=sys.stderr)
        sys.exit(1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    hidden_sizes = tuple(int(x.strip()) for x in args.hidden.split(",") if x.strip())
    model = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        max_iter=args.max_iter,
        alpha=args.alpha,
        random_state=42,
    )
    model.fit(X_train, y_train)

    def evaluate(name, X, y, file_ids, file_labels):
        pred = model.predict(X)
        file_pred = aggregate_predictions(file_ids, pred)
        file_true = {fid: file_labels[fid] for fid in file_pred.keys()}
        y_file_true = [file_true[fid] for fid in sorted(file_pred.keys())]
        y_file_pred = [file_pred[fid] for fid in sorted(file_pred.keys())]
        acc = accuracy_score(y_file_true, y_file_pred)
        print(f"\n{name} file-level accuracy: {acc:.4f}")
        print(f"{name} confusion matrix (file-level):")
        print(confusion_matrix(y_file_true, y_file_pred))
        print(f"{name} classification report (file-level):")
        print(classification_report(y_file_true, y_file_pred))

    evaluate("Validation", X_val, y_val, val_file_ids, val_file_labels)
    evaluate("Test", X_test, y_test, test_file_ids, test_file_labels)


if __name__ == "__main__":
    main()
