#!/usr/bin/env python3
"""Random Forest with raw per-channel normalization + selectable per-channel features."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path

try:
    import numpy as np
except Exception as e:  # pragma: no cover
    print("Missing dependency: numpy. Install with: pip install numpy", file=sys.stderr)
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from scipy.stats import f_oneway, ttest_ind
except Exception as e:  # pragma: no cover
    print("Missing dependency: scipy. Install with: pip install scipy", file=sys.stderr)
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    print("Missing dependency: matplotlib. Install with: pip install matplotlib", file=sys.stderr)
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
except Exception as e:  # pragma: no cover
    print("Missing dependency: scikit-learn. Install with: pip install scikit-learn", file=sys.stderr)
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)


CHANNELS = ["CH1", "CH2", "CH3", "CH4"]
TIMESTAMP_COL = "timestamp"
CUT_SECONDS = 1.0
MA_WINDOW = 5
SMOOTH_ALPHA = 0.2
FEATURE_NAMES = ["mav", "rms", "zcr", "wl", "ssc"]
ALL_FEATURE_KEYS = [f"{ch}_{feat.upper()}" for ch in CHANNELS for feat in FEATURE_NAMES]


def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


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


def channel_features(vals, selected_features):
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


def normalize_feature_key(feature_key: str):
    key = feature_key.strip().upper()
    parts = key.split("_", 1)
    if len(parts) != 2:
        return None
    ch, feat = parts
    if ch not in CHANNELS:
        return None
    if feat.lower() not in FEATURE_NAMES:
        return None
    return f"{ch}_{feat}"


def parse_selected_feature_keys(raw_features):
    if not raw_features:
        return ALL_FEATURE_KEYS
    out = []
    seen = set()
    for raw in raw_features:
        norm = normalize_feature_key(raw)
        if norm is None:
            raise ValueError(f"Invalid feature '{raw}'. Use CH[1-4]_[MAV|RMS|ZCR|WL|SSC].")
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def extract_features(csv_path: Path, use_ma: bool, use_smooth: bool, selected_feature_keys):
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

    feature_maps = {}
    for ch_name, ch_vals in zip(CHANNELS, channel_values):
        feats = channel_features(ch_vals, FEATURE_NAMES)
        if feats is None:
            print(f"Skipping {csv_path}: empty channel after preprocessing", file=sys.stderr)
            return None
        metric_map = {name.upper(): value for name, value in zip(FEATURE_NAMES, feats)}
        feature_maps[ch_name] = metric_map

    features = []
    for key in selected_feature_keys:
        ch, feat = key.split("_", 1)
        features.append(feature_maps[ch][feat])

    return features


def load_split(split_dir: Path, use_ma: bool, use_smooth: bool, selected_feature_keys):
    X, y = [], []
    for class_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        label = class_dir.name
        for csv_path in sorted(class_dir.glob("*.csv")):
            feat = extract_features(csv_path, use_ma, use_smooth, selected_feature_keys)
            if feat is None:
                continue
            X.append(feat)
            y.append(label)
    return X, y


def run_anova_and_posthoc(X, y, feature_names, alpha=0.05):
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    class_labels = sorted(set(y_arr.tolist()))
    comparisons = list(combinations(class_labels, 2))
    anova_pvals = np.full(len(feature_names), np.nan, dtype=float)
    posthoc_adj_p = np.full((len(comparisons), len(feature_names)), np.nan, dtype=float)

    print("\nANOVA results (training set):")
    for idx, feat_name in enumerate(feature_names):
        groups = [X_arr[y_arr == label, idx] for label in class_labels]
        if any(len(group) < 2 for group in groups):
            print(f"  {feat_name}: skipped (insufficient samples)")
            continue
        stat, p_value = f_oneway(*groups)
        anova_pvals[idx] = p_value
        print(f"  {feat_name}: F={stat:.4f}, p={p_value:.6g}")
        if p_value < alpha:
            print("    Post-hoc (Welch t-test, Bonferroni corrected):")
            for pair_idx, (left, right) in enumerate(comparisons):
                left_vals = X_arr[y_arr == left, idx]
                right_vals = X_arr[y_arr == right, idx]
                if len(left_vals) < 2 or len(right_vals) < 2:
                    continue
                t_stat, pair_p = ttest_ind(left_vals, right_vals, equal_var=False, nan_policy="omit")
                adj_p = min(pair_p * len(comparisons), 1.0)
                posthoc_adj_p[pair_idx, idx] = adj_p
                if adj_p < alpha:
                    print(f"      {left} vs {right}: t={t_stat:.4f}, p_adj={adj_p:.6g}")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(10, 0.55 * len(feature_names)), max(6, 1.2 * max(1, len(comparisons))))
    )
    eps = 1e-12

    anova_map = -np.log10(np.clip(anova_pvals, eps, 1.0))[np.newaxis, :]
    im1 = ax1.imshow(anova_map, aspect="auto", cmap="viridis")
    ax1.set_title("ANOVA significance (-log10 p)")
    ax1.set_yticks([0])
    ax1.set_yticklabels(["all classes"])
    ax1.set_xticks(range(len(feature_names)))
    ax1.set_xticklabels(feature_names, rotation=70, ha="right")
    fig.colorbar(im1, ax=ax1, fraction=0.025, pad=0.02)

    if len(comparisons) > 0:
        posthoc_map = -np.log10(np.clip(posthoc_adj_p, eps, 1.0))
        im2 = ax2.imshow(posthoc_map, aspect="auto", cmap="magma")
        ax2.set_title("Post-hoc significance (Welch + Bonferroni, -log10 p_adj)")
        ax2.set_yticks(range(len(comparisons)))
        ax2.set_yticklabels([f"{a} vs {b}" for a, b in comparisons])
        ax2.set_xticks(range(len(feature_names)))
        ax2.set_xticklabels(feature_names, rotation=70, ha="right")
        fig.colorbar(im2, ax=ax2, fraction=0.025, pad=0.02)
    else:
        ax2.axis("off")

    fig.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="Random Forest with raw normalization + windowed features")
    ap.add_argument("--data", default="data", help="Training root folder")
    ap.add_argument("--validation", default="validation", help="Validation root folder")
    ap.add_argument("--test", default="test", help="Test root folder")
    ap.add_argument("--n-estimators", default=300, type=int, help="Number of trees")
    ap.add_argument("--max-depth", default=None, type=int, help="Max tree depth")
    ap.add_argument("--min-samples-split", default=2, type=int, help="Min samples to split")
    ap.add_argument("--min-samples-leaf", default=1, type=int, help="Min samples in leaf")
    ap.add_argument("--ma", action="store_true", help="Apply moving average before feature extraction")
    ap.add_argument("--smooth", action="store_true", help="Apply exponential smoothing before feature extraction")
    ap.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Selected channel-features, e.g. CH1_MAV CH4_SSC. Default uses all channel-feature pairs.",
    )
    args = ap.parse_args()

    try:
        selected_feature_keys = parse_selected_feature_keys(args.features)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    data_dir = Path(args.data)
    val_dir = Path(args.validation)
    test_dir = Path(args.test)

    for d in [data_dir, val_dir, test_dir]:
        if not d.exists():
            print(f"Missing directory: {d}", file=sys.stderr)
            sys.exit(1)

    X_train, y_train = load_split(data_dir, args.ma, args.smooth, selected_feature_keys)
    X_val, y_val = load_split(val_dir, args.ma, args.smooth, selected_feature_keys)
    X_test, y_test = load_split(test_dir, args.ma, args.smooth, selected_feature_keys)

    if not X_train or not X_val or not X_test:
        print("One or more splits have no usable samples after trimming.", file=sys.stderr)
        sys.exit(1)

    run_anova_and_posthoc(X_train, y_train, selected_feature_keys)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=42,
        n_jobs=-1,
    )
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
        print(f"{name} classification report:")
        print(classification_report(y, pred, zero_division=0))

    evaluate("Validation", X_val, y_val)
    evaluate("Test", X_test, y_test)


if __name__ == "__main__":
    main()
