#!/usr/bin/env python3
"""Generate synthetic CSVs from the existing dataset without modifying originals.

The script copies the folder structure from a source root into an output root and
creates synthetic variants of each CSV by applying small per-channel perturbations:
- random amplitude scaling
- small additive bias
- gaussian noise proportional to the channel spread

This keeps timestamps and row count unchanged so the synthetic files remain
compatible with the existing training scripts.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

CHANNELS = ["CH1", "CH2", "CH3", "CH4"]
TIMESTAMP_COL = "timestamp"


def read_csv(csv_path: Path):
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None, None
        rows = list(reader)
        return reader.fieldnames, rows


def column_stats(rows):
    stats = {}
    for ch in CHANNELS:
        vals = []
        for row in rows:
            try:
                vals.append(float(row[ch]))
            except Exception:
                continue
        if not vals:
            stats[ch] = {"mean": 0.0, "std": 0.0}
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        stats[ch] = {"mean": mean, "std": var ** 0.5}
    return stats


def augment_rows(rows, stats, scale_range, noise_ratio, bias_ratio, rng):
    out_rows = []
    for row in rows:
        new_row = dict(row)
        for ch in CHANNELS:
            try:
                x = float(row[ch])
            except Exception:
                new_row[ch] = row.get(ch, "")
                continue

            scale = rng.uniform(scale_range[0], scale_range[1])
            sigma = stats[ch]["std"] * noise_ratio
            bias_sigma = stats[ch]["std"] * bias_ratio
            noise = rng.gauss(0.0, sigma) if sigma > 0 else 0.0
            bias = rng.gauss(0.0, bias_sigma) if bias_sigma > 0 else 0.0
            y = (x * scale) + bias + noise
            new_row[ch] = f"{y:.6f}"
        out_rows.append(new_row)
    return out_rows


def write_csv(csv_path: Path, fieldnames, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic CSVs from the existing dataset")
    ap.add_argument("--source", default="data", help="Source root folder")
    ap.add_argument("--output", default="synthetic_data", help="Output root folder")
    ap.add_argument("--copies", type=int, default=3, help="Synthetic copies per source CSV")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--scale-min", type=float, default=0.95, help="Minimum amplitude scale")
    ap.add_argument("--scale-max", type=float, default=1.05, help="Maximum amplitude scale")
    ap.add_argument("--noise-ratio", type=float, default=0.03, help="Noise std as a fraction of channel std")
    ap.add_argument("--bias-ratio", type=float, default=0.02, help="Bias std as a fraction of channel std")
    ap.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional class names to augment. Default is all class folders.",
    )
    args = ap.parse_args()

    if args.copies < 1:
        print("--copies must be at least 1", file=sys.stderr)
        sys.exit(1)
    if args.scale_min <= 0 or args.scale_max <= 0 or args.scale_min > args.scale_max:
        print("Invalid scale range", file=sys.stderr)
        sys.exit(1)

    source_root = Path(args.source)
    output_root = Path(args.output)
    if not source_root.exists():
        print(f"Missing source folder: {source_root}", file=sys.stderr)
        sys.exit(1)

    class_dirs = sorted([p for p in source_root.iterdir() if p.is_dir()])
    if args.classes:
        class_dirs = [p for p in class_dirs if p.name in set(args.classes)]
    if not class_dirs:
        print("No class folders found to augment.", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    created = 0

    for class_dir in class_dirs:
        csv_files = sorted(class_dir.glob("*.csv"))
        if not csv_files:
            continue
        for csv_file in csv_files:
            fieldnames, rows = read_csv(csv_file)
            if fieldnames is None or not rows:
                continue
            stats = column_stats(rows)
            rel = csv_file.relative_to(source_root)

            for copy_idx in range(args.copies):
                synthetic_rows = augment_rows(
                    rows,
                    stats,
                    (args.scale_min, args.scale_max),
                    args.noise_ratio,
                    args.bias_ratio,
                    rng,
                )
                out_name = f"{csv_file.stem}_synthetic_{copy_idx + 1}.csv"
                out_path = output_root / rel.parent / out_name
                write_csv(out_path, fieldnames, synthetic_rows)
                created += 1

    print(f"Created {created} synthetic CSV files in {output_root}")


if __name__ == "__main__":
    main()
