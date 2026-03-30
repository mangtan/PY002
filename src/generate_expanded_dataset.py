#!/usr/bin/env python3
"""
Generate an expanded UCI-Wine-format dataset with synthetic Italian red varieties.

Output format keeps compatibility with wine.data:
label,f1,f2,...,f13  (no header)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Expand UCI Wine dataset with synthetic classes")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("project/wine/wine.data"),
        help="Original UCI wine.data path",
    )
    parser.add_argument(
        "--output-data",
        type=Path,
        default=Path("project/wine/wine_expanded.data"),
        help="Expanded no-header data path",
    )
    parser.add_argument(
        "--output-meta",
        type=Path,
        default=Path("project/wine/wine_expanded_with_meta.csv"),
        help="Expanded data with metadata columns",
    )
    parser.add_argument(
        "--output-map",
        type=Path,
        default=Path("project/wine/label_map.json"),
        help="Label map json path",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=120,
        help="Synthetic samples per added variety",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--shift-scale",
        type=float,
        default=1.35,
        help="Scale factor for mean shifts (larger -> classes farther apart)",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.65,
        help="Scale factor for synthetic noise (smaller -> classes tighter)",
    )
    return parser.parse_args()


def build_variety_specs():
    """
    Each spec:
    - base: base UCI class id (1..3)
    - blend: optional second base class and weight
    - shift_z: feature shift in units of global std
    - scale: per-feature std scale
    """
    return [
        {
            "name": "Sangiovese",
            "base": 2,
            "blend": (3, 0.25),
            "shift_z": {0: 0.25, 1: 0.10, 6: -0.20, 9: 0.15, 10: 0.10},
            "scale": 0.85,
        },
        {
            "name": "Nebbiolo",
            "base": 1,
            "blend": (2, 0.20),
            "shift_z": {0: 0.35, 5: 0.45, 6: 0.35, 8: 0.40, 9: -0.10, 10: -0.05},
            "scale": 0.75,
        },
        {
            "name": "Barbera",
            "base": 3,
            "blend": (2, 0.20),
            "shift_z": {0: -0.15, 1: 0.25, 5: -0.35, 6: -0.35, 9: 0.40, 10: -0.15},
            "scale": 0.90,
        },
        {
            "name": "Montepulciano",
            "base": 3,
            "blend": (1, 0.15),
            "shift_z": {0: 0.20, 1: 0.05, 6: -0.10, 9: 0.35, 12: 0.20},
            "scale": 0.85,
        },
        {
            "name": "Aglianico",
            "base": 1,
            "blend": (3, 0.20),
            "shift_z": {0: 0.45, 1: 0.20, 5: 0.25, 8: 0.25, 9: 0.45, 12: 0.35},
            "scale": 0.80,
        },
        {
            "name": "Primitivo",
            "base": 1,
            "blend": (3, 0.15),
            "shift_z": {0: 0.55, 1: -0.10, 3: -0.20, 9: 0.50, 10: -0.25, 12: 0.40},
            "scale": 0.80,
        },
        {
            "name": "Nero dAvola",
            "base": 3,
            "blend": (1, 0.10),
            "shift_z": {0: 0.30, 1: 0.10, 6: 0.10, 9: 0.45, 12: 0.25},
            "scale": 0.85,
        },
        {
            "name": "Corvina",
            "base": 2,
            "blend": (3, 0.15),
            "shift_z": {0: -0.20, 1: 0.20, 5: -0.25, 6: -0.20, 9: -0.10, 12: -0.20},
            "scale": 0.90,
        },
        {
            "name": "Dolcetto",
            "base": 2,
            "blend": (1, 0.20),
            "shift_z": {0: 0.15, 1: -0.15, 5: 0.05, 6: -0.10, 9: 0.10, 12: 0.10},
            "scale": 0.90,
        },
    ]


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    raw = pd.read_csv(args.input, header=None)
    y = raw.iloc[:, 0].astype(int).to_numpy()
    x = raw.iloc[:, 1:].astype(float).to_numpy()

    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    feat_range = maxs - mins
    lower = mins - 0.10 * feat_range
    upper = maxs + 0.15 * feat_range
    std = x.std(axis=0, ddof=1)

    class_means = {c: x[y == c].mean(axis=0) for c in sorted(np.unique(y))}
    specs = build_variety_specs()

    records = []
    # original samples
    for i in range(len(raw)):
        row = raw.iloc[i].to_list()
        label = int(row[0])
        feat = [float(v) for v in row[1:]]
        records.append(
            {
                "label": label,
                "variety": f"UCI_Class_{label}",
                "source": "original",
                "features": feat,
            }
        )

    # synthetic classes start from label 4
    next_label = 4
    label_map = {
        1: "UCI_Class_1",
        2: "UCI_Class_2",
        3: "UCI_Class_3",
    }

    for spec in specs:
        base_mean = class_means[spec["base"]].copy()
        if spec.get("blend") is not None:
            blend_class, blend_weight = spec["blend"]
            base_mean = (1 - blend_weight) * base_mean + blend_weight * class_means[blend_class]

        mean_vec = base_mean.copy()
        for idx, z_shift in spec["shift_z"].items():
            mean_vec[idx] += (z_shift * args.shift_scale) * std[idx]

        sigma = np.maximum(std * spec["scale"] * args.noise_scale, 1e-6)
        samples = rng.normal(loc=mean_vec, scale=sigma, size=(args.per_class, x.shape[1]))
        samples = np.clip(samples, lower, upper)

        label_map[next_label] = spec["name"]
        for row in samples:
            records.append(
                {
                    "label": next_label,
                    "variety": spec["name"],
                    "source": "synthetic",
                    "features": row.tolist(),
                }
            )
        next_label += 1

    # Save no-header UCI-compatible data
    data_rows = []
    for rec in records:
        data_rows.append([rec["label"], *rec["features"]])
    out_df = pd.DataFrame(data_rows)
    args.output_data.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_data, index=False, header=False, float_format="%.5f")

    # Save meta CSV
    meta_cols = ["label", "variety", "source"] + FEATURE_NAMES
    meta_rows = []
    for rec in records:
        meta_rows.append([rec["label"], rec["variety"], rec["source"], *rec["features"]])
    meta_df = pd.DataFrame(meta_rows, columns=meta_cols)
    meta_df.to_csv(args.output_meta, index=False, float_format="%.5f")

    # Save map
    with args.output_map.open("w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    synthetic_count = len(records) - len(raw)
    print("Expanded dataset generated.")
    print(f"Original rows: {len(raw)}")
    print(f"Synthetic rows: {synthetic_count}")
    print(f"Total rows: {len(records)}")
    print(f"Classes: {len(label_map)}")
    print(f"Output data: {args.output_data}")
    print(f"Output meta: {args.output_meta}")
    print(f"Output label map: {args.output_map}")


if __name__ == "__main__":
    main()
