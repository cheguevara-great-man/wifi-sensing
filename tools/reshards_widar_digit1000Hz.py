#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd


def resolve_shard_path(shard_dir: Path, shard_id: int) -> Path:
    cands = [
        shard_dir / f"shard-{shard_id:05d}.npz",
        shard_dir / f"shard-{(shard_id + 1):05d}.npz",
        shard_dir / f"shard-{max(shard_id - 1, 0):05d}.npz",
    ]
    for p in cands:
        if p.exists():
            return p
    g = sorted(shard_dir.glob(f"*{shard_id}*.npz"))
    if g:
        return g[0]
    raise FileNotFoundError(f"Cannot find shard for shard_id={shard_id} under {shard_dir}")


def load_npz_shard(path: Path, x_key: str = "X") -> np.ndarray:
    """Return X as float32, shape (N,T,F). No downsample here."""
    npz = np.load(path, allow_pickle=False)
    if x_key not in npz:
        raise KeyError(f"{path}: key '{x_key}' not found. keys={list(npz.keys())}")
    X = npz[x_key]
    # normalize to (N,T,F)
    if X.ndim == 4 and X.shape[1] == 1:
        X = X[:, 0, :, :]
    elif X.ndim != 3:
        raise ValueError(f"{path}: unexpected X shape {X.shape}")
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    npz.close()
    return X


def preload_all_shards(src_root: Path, variant: str, needed_sids: np.ndarray, x_key: str = "X"):
    shard_dir = src_root / variant / "shards"
    out = {}
    for sid in sorted(map(int, needed_sids.tolist())):
        p = resolve_shard_path(shard_dir, sid)
        out[sid] = load_npz_shard(p, x_key=x_key)
    return out


def gather_X_for_rows(src_df: pd.DataFrame, row_ids: np.ndarray, shard_data: dict, x_key: str = "X") -> np.ndarray:
    """
    row_ids: indices in src_df (NOT orig_row_index). src_df must contain src_shard_id, src_offset.
    Return X_out: (len(row_ids), T, F)
    """
    sids = src_df.loc[row_ids, "src_shard_id"].to_numpy(np.int64)
    offs = src_df.loc[row_ids, "src_offset"].to_numpy(np.int64)

    # determine T,F from first sample
    X0 = shard_data[int(sids[0])]
    T, F = X0.shape[1], X0.shape[2]
    X_out = np.empty((len(row_ids), T, F), dtype=np.float32)

    # group by src shard id to reduce python loops
    uniq = np.unique(sids)
    for sid in uniq:
        m = (sids == sid)
        idx = np.nonzero(m)[0]
        X_out[idx] = shard_data[int(sid)][offs[idx]]
    return X_out


def write_one_shard(out_npz: Path, X: np.ndarray, y: np.ndarray):
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, X=X, y=y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", type=str, required=True, help="Original 1000Hz root (has amp/conj/meta)")
    ap.add_argument("--ref_root", type=str, required=True, help="Existing reshard output root (has train/test/meta/index.csv)")
    ap.add_argument("--out_root", type=str, required=True, help="Output root for 1000Hz reshards")
    ap.add_argument("--min_raw_len", type=int, default=1000, help="Must match the value used when creating ref index.csv")
    ap.add_argument("--workers", type=int, default=12, help="Parallel shard writers (recommend 8~16)")
    args = ap.parse_args()

    src_root = Path(args.src_root).resolve()
    ref_root = Path(args.ref_root).resolve()
    out_root = Path(args.out_root).resolve()

    # ---- rebuild the SAME "df after filtering+reset_index" so orig_row_index aligns
    src_index = src_root / "meta" / "index.csv"
    if not src_index.exists():
        raise FileNotFoundError(f"missing {src_index}")

    df = pd.read_csv(src_index)
    if "raw_len" in df.columns:
        df = df[df["raw_len"] >= args.min_raw_len].copy()
    df.reset_index(drop=True, inplace=True)  # this is what makes orig_row_index meaningful

    needed_src_sids = df["shard_id"].unique()

    # ---- preload original shards (big RAM but fastest)
    print(f"[PRELOAD] amp shards: {len(needed_src_sids)}")
    amp_data = preload_all_shards(src_root, "amp", needed_src_sids, x_key="X")
    any_sid = int(sorted(amp_data.keys())[0])
    print(f"[PRELOAD] amp example shard {any_sid}: {amp_data[any_sid].shape} dtype={amp_data[any_sid].dtype}")

    print(f"[PRELOAD] conj shards: {len(needed_src_sids)}")
    conj_data = preload_all_shards(src_root, "conj", needed_src_sids, x_key="X")
    any_sid = int(sorted(conj_data.keys())[0])
    print(f"[PRELOAD] conj example shard {any_sid}: {conj_data[any_sid].shape} dtype={conj_data[any_sid].dtype}")

    # ---- copy label_map.csv if exists
    (out_root / "train" / "meta").mkdir(parents=True, exist_ok=True)
    (out_root / "test" / "meta").mkdir(parents=True, exist_ok=True)
    for split in ["train", "test"]:
        for name in ["label_map.csv"]:
            src = src_root / "meta" / name
            if src.exists():
                dst = out_root / split / "meta" / name
                dst.write_bytes(src.read_bytes())

    # ---- process each split by ref index.csv
    for split in ["train", "test"]:
        ref_index = ref_root / split / "meta" / "index.csv"
        if not ref_index.exists():
            raise FileNotFoundError(f"missing {ref_index}")

        ref_df = pd.read_csv(ref_index)
        if "orig_row_index" not in ref_df.columns:
            raise RuntimeError(f"{ref_index} has no 'orig_row_index'. Need it to map back to src shard/offset.")

        # map back to src shard_id/offset via orig_row_index
        ori = ref_df["orig_row_index"].to_numpy(np.int64)
        if ori.max() >= len(df):
            raise RuntimeError(
                f"orig_row_index out of range: max={ori.max()} but len(filtered_src_df)={len(df)}. "
                f"min_raw_len mismatch? you used min_raw_len={args.min_raw_len}"
            )

        # build a working dataframe with both:
        #   new_shard_id/new_offset from ref, and src_shard_id/src_offset from original df
        work = ref_df.copy()
        work["src_shard_id"] = df.loc[ori, "shard_id"].to_numpy(np.int64)
        work["src_offset"] = df.loc[ori, "offset"].to_numpy(np.int64)

        # write meta/index.csv directly (same as ref; it already has correct new shard_id/offset)
        out_meta = out_root / split / "meta"
        out_meta.mkdir(parents=True, exist_ok=True)
        out_index = out_meta / "index.csv"
        work.to_csv(out_index, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"[{split}] wrote meta/index.csv: {out_index} rows={len(work)}")

        # group rows by NEW shard_id, order by NEW offset
        out_split = out_root / split
        (out_split / "amp" / "shards").mkdir(parents=True, exist_ok=True)
        (out_split / "conj" / "shards").mkdir(parents=True, exist_ok=True)

        groups = []
        for new_sid, g in work.groupby("shard_id", sort=True):
            g2 = g.sort_values("offset")
            # row ids are in work's row index space (0..len-1)
            groups.append((int(new_sid), g2.index.to_numpy(np.int64)))

        def job(new_sid: int, row_ids: np.ndarray):
            Xa = gather_X_for_rows(work, row_ids, amp_data, x_key="X")
            Xb = gather_X_for_rows(work, row_ids, conj_data, x_key="X")
            y = work.loc[row_ids, "label"].to_numpy(np.int64).reshape(-1, 1)

            write_one_shard(out_split / "amp" / "shards" / f"shard-{new_sid:05d}.npz", Xa, y)
            write_one_shard(out_split / "conj" / "shards" / f"shard-{new_sid:05d}.npz", Xb, y)

        print(f"[{split}] writing {len(groups)} shards with workers={args.workers} ...")
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
            futs = [ex.submit(job, sid, row_ids) for sid, row_ids in groups]
            for fu in as_completed(futs):
                fu.result()

        print(f"[{split}] done.")

    print("\nALL DONE.")
    print(f"OUT_ROOT = {out_root}")


if __name__ == "__main__":
    main()
