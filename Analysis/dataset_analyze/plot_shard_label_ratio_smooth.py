#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def compute_ratios(index_csv: str):
    df = pd.read_csv(index_csv)
    df["shard_id"] = df["shard_id"].astype(int)
    df["label"] = df["label"].astype(int)

    shards = sorted(df["shard_id"].unique().tolist())
    labels = sorted(df["label"].unique().tolist())

    counts = (
        df.groupby(["shard_id", "label"])
          .size()
          .unstack(fill_value=0)
          .reindex(index=shards, columns=labels, fill_value=0)
    )
    ratios = counts.div(counts.sum(axis=1), axis=0).to_numpy(dtype=np.float64)  # (S,L)

    return np.array(shards, dtype=int), np.array(labels, dtype=int), ratios


def upsample_bilinear(Z, sx=6, sy=12):
    """
    Pure numpy bilinear upsample:
      Z: (S,L) -> (S*sx, L*sy)
    """
    S, L = Z.shape
    x = np.arange(S)
    y = np.arange(L)

    x2 = np.linspace(0, S - 1, S * sx)
    y2 = np.linspace(0, L - 1, L * sy)

    # interp along x for each y
    Zx = np.vstack([np.interp(x2, x, Z[:, j]) for j in range(L)]).T  # (S*sx, L)
    # interp along y for each x2
    Zxy = np.vstack([np.interp(y2, y, Zx[i, :]) for i in range(Zx.shape[0])])  # (S*sx, L*sy)
    return x2, y2, Zxy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="meta/index.csv path")
    ap.add_argument("--out_prefix", required=True, help="output prefix, e.g. /tmp/train_ratio")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--sx", type=int, default=8, help="upsample factor along shard axis")
    ap.add_argument("--sy", type=int, default=20, help="upsample factor along label axis")
    args = ap.parse_args()

    shards, labels, Z = compute_ratios(args.index)  # Z shape (S,L)
    S, L = Z.shape

    # upsample for smooth plots
    x2, y2, Z2 = upsample_bilinear(Z, sx=max(1, args.sx), sy=max(1, args.sy))

    # ---- 1) Smooth heatmap (best readability)
    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_subplot(111)

    im = ax.imshow(
        Z2.T,  # (L*, S*) so y=label, x=shard
        origin="lower",
        aspect="auto",
        interpolation="bicubic"
    )
    ax.set_title("Per-shard label ratio (smoothed heatmap)")
    ax.set_xlabel("shard")
    ax.set_ylabel("label")

    # tick labels (sparse)
    xt_step = max(1, int(len(x2) // 10))
    ax.set_xticks(np.arange(0, len(x2), xt_step))
    ax.set_xticklabels([str(int(round(x2[i]))) for i in range(0, len(x2), xt_step)])

    ax.set_yticks(np.linspace(0, len(y2) - 1, L))
    ax.set_yticklabels([str(int(l)) for l in labels])

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="ratio")
    plt.tight_layout()
    heat_path = f"{args.out_prefix}_heatmap.png"
    plt.savefig(heat_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # ---- 2) Smooth 3D surface
    Xg, Yg = np.meshgrid(x2, y2, indexing="xy")  # (L*, S*)
    Zg = Z2.T  # (L*, S*)

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Xg, Yg, Zg, linewidth=0, antialiased=True)

    ax.set_title("Per-shard label ratio (smoothed surface)")
    ax.set_xlabel("shard")
    ax.set_ylabel("label")
    ax.set_zlabel("ratio")

    # y ticks as labels 0..9
    ax.set_yticks(labels)
    ax.set_yticklabels([str(int(l)) for l in labels])

    surf_path = f"{args.out_prefix}_surface3d.png"
    plt.tight_layout()
    plt.savefig(surf_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print("[OK] saved:")
    print("  ", heat_path)
    print("  ", surf_path)
    print(f"[INFO] original grid: shards={S}, labels={L}, upsampled={Z2.shape[0]}x{Z2.shape[1]}")


if __name__ == "__main__":
    main()
