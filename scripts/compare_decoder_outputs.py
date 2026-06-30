"""Compare decoder 7-channel output between Trellis-Mac and trellis2mlx.

Loads decoder_output.npz from each pipeline and reports per-channel
statistics, spatial divergence patterns, and intersection flag agreement.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/compare_decoder_outputs.py \
        --trellis-mac /path/to/trellis-mac/decoder_output.npz \
        --trellis-mlx /path/to/trellis2mlx/checkpoints/decoder_output.npz \
        [--output-dir /tmp/decoder-comparison]
"""

import argparse
import json
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Compare decoder outputs")
    parser.add_argument("--trellis-mac", required=True,
                        help="Path to Trellis-Mac decoder_output.npz")
    parser.add_argument("--trellis-mlx", required=True,
                        help="Path to trellis2mlx decoder_output.npz")
    parser.add_argument("--output-dir", default="/tmp/decoder-comparison")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load both
    mac = np.load(args.trellis_mac)
    mlx = np.load(args.trellis_mlx)

    mac_feats = mac['feats']
    mac_coords = mac['coords']
    mlx_feats = mlx['feats']
    mlx_coords = mlx['coords']

    print(f"Trellis-Mac: feats {mac_feats.shape}, coords {mac_coords.shape}")
    print(f"trellis2mlx: feats {mlx_feats.shape}, coords {mlx_coords.shape}")

    # Check if they even have the same number of voxels
    if mac_feats.shape[0] != mlx_feats.shape[0]:
        print(f"\nDIFFERENT VOXEL COUNTS: {mac_feats.shape[0]:,} vs {mlx_feats.shape[0]:,}")
        print("Cannot do pointwise comparison — the decoder produced different "
              "numbers of active voxels. This means the subdivision masks diverge, "
              "which is upstream of the 7-channel features.")

        # Still useful: compare coordinate sets
        mac_spatial = mac_coords[:, 1:4] if mac_coords.shape[1] == 4 else mac_coords
        mlx_spatial = mlx_coords[:, 1:4] if mlx_coords.shape[1] == 4 else mlx_coords

        mac_set = set(map(tuple, mac_spatial))
        mlx_set = set(map(tuple, mlx_spatial))
        common = mac_set & mlx_set
        only_mac = mac_set - mlx_set
        only_mlx = mlx_set - mac_set

        print(f"\nCoordinate comparison:")
        print(f"  Common voxels: {len(common):,}")
        print(f"  Only in Trellis-Mac: {len(only_mac):,}")
        print(f"  Only in trellis2mlx: {len(only_mlx):,}")
        print(f"  Jaccard similarity: {len(common) / len(mac_set | mlx_set):.4f}")

        # For common voxels, compare features
        if common:
            # Build index maps
            mac_idx = {tuple(c): i for i, c in enumerate(mac_spatial)}
            mlx_idx = {tuple(c): i for i, c in enumerate(mlx_spatial)}

            common_list = sorted(common)
            n_common = min(len(common_list), 100000)
            sample = common_list[:n_common]

            mac_common_feats = np.array([mac_feats[mac_idx[c]] for c in sample])
            mlx_common_feats = np.array([mlx_feats[mlx_idx[c]] for c in sample])

            print(f"\nFeature comparison on {n_common:,} common voxels:")
            _compare_features(mac_common_feats, mlx_common_feats)
        return

    # Same voxel count — check if coordinates match
    coords_match = np.array_equal(mac_coords, mlx_coords)
    print(f"\nCoordinates match: {coords_match}")

    if not coords_match:
        # Try sorting by coordinates to align
        mac_order = np.lexsort(mac_coords[:, ::-1].T)
        mlx_order = np.lexsort(mlx_coords[:, ::-1].T)
        mac_sorted = mac_coords[mac_order]
        mlx_sorted = mlx_coords[mlx_order]
        if np.array_equal(mac_sorted, mlx_sorted):
            print("  Coordinates match after sorting — different ordering only")
            mac_feats = mac_feats[mac_order]
            mlx_feats = mlx_feats[mlx_order]
        else:
            print("  Coordinates differ even after sorting!")
            n_diff = (mac_sorted != mlx_sorted).any(axis=1).sum()
            print(f"  {n_diff:,} / {len(mac_sorted):,} coordinates differ")

    # Compare features
    print(f"\n=== Per-Channel Feature Comparison ===")
    _compare_features(mac_feats, mlx_feats)


def _compare_features(mac_feats, mlx_feats):
    """Compare 7-channel features between two arrays."""
    channel_names = [
        "vertex_offset_x", "vertex_offset_y", "vertex_offset_z",
        "intersect_flag_x", "intersect_flag_y", "intersect_flag_z",
        "quad_split_weight",
    ]

    for ch in range(min(mac_feats.shape[1], 7)):
        mac_ch = mac_feats[:, ch]
        mlx_ch = mlx_feats[:, ch]
        diff = np.abs(mac_ch - mlx_ch)
        name = channel_names[ch] if ch < len(channel_names) else f"ch{ch}"

        print(f"\n  {name}:")
        print(f"    Mac range: [{mac_ch.min():.4f}, {mac_ch.max():.4f}], "
              f"mean: {mac_ch.mean():.4f}")
        print(f"    MLX range: [{mlx_ch.min():.4f}, {mlx_ch.max():.4f}], "
              f"mean: {mlx_ch.mean():.4f}")
        print(f"    Abs diff: mean={diff.mean():.6f}, max={diff.max():.6f}, "
              f"p95={np.percentile(diff, 95):.6f}, p99={np.percentile(diff, 99):.6f}")

        # For intersection flags (channels 3-5): compare thresholded agreement
        if 3 <= ch <= 5:
            mac_flag = mac_ch > 0
            mlx_flag = mlx_ch > 0
            agree = (mac_flag == mlx_flag).mean()
            n_disagree = (mac_flag != mlx_flag).sum()
            print(f"    Flag agreement (>0): {agree:.4f} ({n_disagree:,} disagree)")
            # Which direction is the disagreement?
            mac_yes_mlx_no = (mac_flag & ~mlx_flag).sum()
            mac_no_mlx_yes = (~mac_flag & mlx_flag).sum()
            print(f"    Mac=yes MLX=no: {mac_yes_mlx_no:,}, Mac=no MLX=yes: {mac_no_mlx_yes:,}")

    # Overall correlation
    for ch in range(min(mac_feats.shape[1], 7)):
        mac_ch = mac_feats[:, ch]
        mlx_ch = mlx_feats[:, ch]
        if mac_ch.std() > 0 and mlx_ch.std() > 0:
            corr = np.corrcoef(mac_ch, mlx_ch)[0, 1]
            name = channel_names[ch] if ch < len(channel_names) else f"ch{ch}"
            print(f"  Correlation {name}: {corr:.6f}")


if __name__ == "__main__":
    main()
