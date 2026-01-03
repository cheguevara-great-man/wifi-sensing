#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Re-shard Widar digit dataset (amp+conj) for fast IO and clean evaluation.

Input root structure:
  ROOT/
    meta/index.csv
    meta/label_map.csv  (optional)
    amp/shards/shard-xxxxx.npz
    conj/shards/shard-xxxxx.npz

Output structure:
  OUT/
    train/
      meta/index.csv
      meta/label_map.csv (copied if exists)
      amp/shards/shard-00000.npz ...
      conj/shards/shard-00000.npz ...
    test/
      meta/index.csv
      meta/label_map.csv
      amp/shards/...
      conj/shards/...

Each output shard contains <= shard_size samples (hard cap).
Trials (date,user,a,b,c,d) are kept intact and never split across train/test or across shards.
"""

import argparse
import csv
import os
import re
import shutil
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


SID_RX_RE = re.compile(r"-(r[1-9]\d*)\.dat$", re.IGNORECASE)


def parse_rx_from_row(row: pd.Series) -> str:
    rx = str(row.get("rx", "")).strip()
    if rx.lower().startswith("r"):
        return rx.lower()
    sid = str(row.get("sample_id", "")).strip()
    m = SID_RX_RE.search(sid)
    if m:
        return m.group(1).lower()
    rp = str(row.get("raw_relpath", "")).strip()
    base = os.path.splitext(os.path.basename(rp))[0]
    m2 = re.search(r"(r[1-9]\d*)$", base, re.IGNORECASE)
    return m2.group(1).lower() if m2 else ""


def build_trial_id(df: pd.DataFrame) -> pd.Series:
    needed = ["date", "user", "a", "b", "c", "d"]
    if all(c in df.columns for c in needed):
        return df[needed].astype(str).agg("|".join, axis=1)
    # fallback: strip "-rX.dat" from sample_id
    base = df["sample_id"].astype(str).str.replace(r"-(r[0-9]+)\.dat$", "", regex=True)
    return base


def resolve_shard_path(shard_dir: Path, shard_id: int) -> Path:
    # Prefer exact id; try +1/-1 shift as fallback (for 0/1-index mismatch).
    cands = [
        shard_dir / f"shard-{shard_id:05d}.npz",
        shard_dir / f"shard-{shard_id:05d}.npz".replace("0000", "0000"),  # no-op safety
        shard_dir / f"shard-{(shard_id+1):05d}.npz",
        shard_dir / f"shard-{max(shard_id-1, 0):05d}.npz",
    ]
    for p in cands:
        if p.exists():
            return p
    # last resort: brute glob
    g = sorted(shard_dir.glob(f"*{shard_id}*.npz"))
    if g:
        return g[0]
    raise FileNotFoundError(f"Cannot find shard file for shard_id={shard_id} under {shard_dir}")


def load_npz_shard(path: Path, x_key: str = "X", target_T: int = 500, downsample_factor: int = 4):
    npz = np.load(path, allow_pickle=False)
    X = npz[x_key]
    # normalize X shape to (N,T,F)
    if X.ndim == 4 and X.shape[1] == 1:
        X = X[:, 0, :, :]
    elif X.ndim != 3:
        raise ValueError(f"{path}: unexpected X shape {X.shape}")

    # enforce float32
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)

    # optional downsample time axis
    if target_T > 0:
        if X.shape[1] == 2000 and target_T == 500:
            X = X[:, ::downsample_factor, :]
        elif X.shape[1] != target_T:
            # leave as-is but warn by shape later
            pass

    npz.close()
    return X


def make_group_stratified_split(trials_df: pd.DataFrame, test_ratio: float, seed: int):
    """
    trials_df columns: trial_id, label, user, date, n_samples, n_rx
    Split by groups (trial_id) with stratification on (user,date,label).
    Ensure each stratum with n>=2 has at least 1 trial in both train and test.
    """
    rng = np.random.RandomState(seed)
    strata_cols = [c for c in ["user", "date", "label"] if c in trials_df.columns]
    if not strata_cols:
        strata_cols = ["label"]

    train_trials = []
    test_trials = []

    for _, g in trials_df.groupby(strata_cols, dropna=False):
        tids = g["trial_id"].tolist()
        rng.shuffle(tids)
        n = len(tids)
        if n == 1:
            # only one trial in this stratum: put into train to keep test cleaner
            train_trials.extend(tids)
            continue

        n_test = int(round(n * test_ratio))
        # enforce at least 1 and at most n-1
        n_test = max(1, min(n_test, n - 1))
        test_trials.extend(tids[:n_test])
        train_trials.extend(tids[n_test:])

    return set(train_trials), set(test_trials)


def build_shard_plan(df_rows: pd.DataFrame, trial_to_rowidx: dict, trial_meta: dict,
                     shard_size: int, seed: int):
    """
    Build list of shards. Each shard is an ordered list of global row indices (df index)
    shuffled within shard for label mixing. Trials are not split across shards.
    Selection across labels is weighted by remaining sample counts to approximate global distribution.
    """
    rng = np.random.RandomState(seed)

    # label -> deque of (trial_id, trial_size)
    label_q = {}
    remaining = {}

    labels = sorted(df_rows["label"].unique().tolist())
    for lb in labels:
        tids = [tid for tid, m in trial_meta.items() if m["label"] == lb]
        rng.shuffle(tids)
        dq = deque()
        total = 0
        for tid in tids:
            k = trial_meta[tid]["n_samples"]
            dq.append((tid, k))
            total += k
        label_q[lb] = dq
        remaining[lb] = total

    # helper: min trial size among any remaining label
    def min_next_trial_size():
        mins = []
        for lb, dq in label_q.items():
            if dq:
                mins.append(dq[0][1])
        return min(mins) if mins else None

    shard_plans = []
    total_used = 0

    while True:
        active_labels = [lb for lb, dq in label_q.items() if dq]
        if not active_labels:
            break

        cur_trials = []
        cur_n = 0

        while True:
            cap = shard_size - cur_n
            if cap <= 0:
                break

            # find labels whose next trial fits
            candidates = []
            weights = []
            for lb in active_labels:
                dq = label_q[lb]
                if not dq:
                    continue
                tid, k = dq[0]
                if k <= cap:
                    candidates.append(lb)
                    weights.append(remaining[lb])

            if not candidates:
                break

            weights = np.asarray(weights, dtype=np.float64)
            weights = weights / (weights.sum() + 1e-12)
            chosen = rng.choice(np.asarray(candidates), p=weights)

            tid, k = label_q[chosen].popleft()
            cur_trials.append(tid)
            cur_n += k
            remaining[chosen] -= k
            total_used += k

            # refresh active labels
            active_labels = [lb for lb, dq in label_q.items() if dq]

            # early stop if remaining capacity can't fit the smallest next trial
            mn = min_next_trial_size()
            if mn is None or (shard_size - cur_n) < mn:
                break

        # Collect row indices for this shard, then shuffle within shard
        row_idx = []
        for tid in cur_trials:
            row_idx.extend(trial_to_rowidx[tid])
        row_idx = np.asarray(row_idx, dtype=np.int64)
        rng.shuffle(row_idx)
        shard_plans.append(row_idx)

    # sanity
    expect = len(df_rows)
    got = sum(len(x) for x in shard_plans)
    if got != expect:
        raise RuntimeError(f"Shard plan size mismatch: expect {expect}, got {got}")

    return shard_plans


def gather_X_for_rows(df_rows: pd.DataFrame, row_idx: np.ndarray,
                      shard_data: dict, x_key: str = "X") -> np.ndarray:
    """
    Efficiently gather X for selected rows into (N,T,F).
    shard_data[csv_shard_id] = np.ndarray (Nshard,T,F)
    """
    sub = df_rows.loc[row_idx, ["shard_id", "offset"]].copy()
    sub["dst"] = np.arange(len(sub), dtype=np.int64)

    # infer shape from first sample
    first_sid = int(sub.iloc[0]["shard_id"])
    X0 = shard_data[first_sid]
    T, F = X0.shape[1], X0.shape[2]

    out = np.empty((len(sub), T, F), dtype=np.float32)

    for sid, g in sub.groupby("shard_id", sort=False):
        sid = int(sid)
        offs = g["offset"].to_numpy(dtype=np.int64)
        dst = g["dst"].to_numpy(dtype=np.int64)
        out[dst] = shard_data[sid][offs]

    return out


def write_one_shard(out_npz: Path, X: np.ndarray, y: np.ndarray):
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    # Uncompressed npz (fast): np.savez uses ZIP_STORED
    np.savez(out_npz, X=X.astype(np.float32, copy=False), y=y.astype(np.int64, copy=False))


def copy_meta(src_root: Path, dst_root: Path):
    src_meta = src_root / "meta"
    dst_meta = dst_root / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)
    for name in ["label_map.csv"]:
        p = src_meta / name
        if p.exists():
            shutil.copy2(p, dst_meta / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="input root containing amp/, conj/, meta/index.csv")
    ap.add_argument("--out", type=str, required=True, help="output root")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--shard_size", type=int, default=384, help="hard cap per new shard")
    ap.add_argument("--min_raw_len", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8, help="thread workers for writing shards")
    ap.add_argument("--drop_incomplete_trials", action="store_true", help="drop trials with n_rx<6")
    ap.add_argument("--target_T", type=int, default=500, help="if input X has T=2000 and target_T=500, downsample by factor")
    ap.add_argument("--downsample_factor", type=int, default=4)

    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    in_index = root / "meta" / "index.csv"
    if not in_index.exists():
        raise FileNotFoundError(f"index.csv not found: {in_index}")

    # -------- Load index.csv
    df = pd.read_csv(in_index)
    df.columns = [c.strip() for c in df.columns]

    need_cols = ["sample_id", "label", "shard_id", "offset", "raw_len"]
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise RuntimeError(f"Missing columns: {miss}. Available: {list(df.columns)}")

    # normalize dtypes
    df["label"] = df["label"].astype(int)
    df["shard_id"] = df["shard_id"].astype(int)
    df["offset"] = df["offset"].astype(int)
    df["raw_len"] = df["raw_len"].astype(int)

    # filter raw_len
    df = df[df["raw_len"] >= args.min_raw_len].copy()
    df.reset_index(drop=True, inplace=True)

    # add rx + trial_id
    if "rx" not in df.columns:
        df["rx"] = ""
    df["rx_parsed"] = df.apply(parse_rx_from_row, axis=1)
    df["trial_id"] = build_trial_id(df)

    # -------- Trial metadata
    g = df.groupby("trial_id", sort=False)
    trial_n = g.size().rename("n_samples")
    trial_rx = g["rx_parsed"].nunique().rename("n_rx")
    trial_label_uniq = g["label"].nunique().rename("n_label_uniq")

    trials_df = pd.concat([trial_n, trial_rx, trial_label_uniq], axis=1).reset_index()
    # attach user/date/label if available
    for c in ["user", "date"]:
        if c in df.columns:
            trials_df[c] = g[c].first().values
    trials_df["label"] = g["label"].first().values

    bad = trials_df[trials_df["n_label_uniq"] != 1]
    if len(bad) > 0:
        raise RuntimeError(f"Found trials with inconsistent labels, example:\n{bad.head()}")

    if args.drop_incomplete_trials:
        keep_trials = set(trials_df[trials_df["n_rx"] == 6]["trial_id"].tolist())
        df = df[df["trial_id"].isin(keep_trials)].copy()
        df.reset_index(drop=True, inplace=True)
        # rebuild trials_df
        g = df.groupby("trial_id", sort=False)
        trial_n = g.size().rename("n_samples")
        trial_rx = g["rx_parsed"].nunique().rename("n_rx")
        trials_df = pd.concat([trial_n, trial_rx], axis=1).reset_index()
        for c in ["user", "date"]:
            if c in df.columns:
                trials_df[c] = g[c].first().values
        trials_df["label"] = g["label"].first().values

    print(f"[INFO] rows={len(df)}, trials={trials_df['trial_id'].nunique()}, "
          f"users={trials_df['user'].nunique() if 'user' in trials_df.columns else 'NA'}, "
          f"dates={trials_df['date'].nunique() if 'date' in trials_df.columns else 'NA'}")
    print("[INFO] trial n_rx hist:", trials_df["n_rx"].value_counts().sort_index().to_dict())
    print("[INFO] label dist:", (df["label"].value_counts().sort_index() / len(df)).round(4).to_dict())

    # -------- Split by trial_id (group & stratified)
    train_trials, test_trials = make_group_stratified_split(
        trials_df, test_ratio=args.test_ratio, seed=args.seed
    )
    assert len(train_trials & test_trials) == 0

    split_map = {"train": train_trials, "test": test_trials}

    # -------- Preload original shards for each variant
    variants = ["amp", "conj"]
    shard_data = {v: {} for v in variants}

    for v in variants:
        shard_dir = root / v / "shards"
        if not shard_dir.exists():
            raise FileNotFoundError(f"Missing shard dir: {shard_dir}")

        needed_sids = sorted(df["shard_id"].unique().tolist())
        print(f"[{v}] preload {len(needed_sids)} shards from {shard_dir}")

        for sid in needed_sids:
            p = resolve_shard_path(shard_dir, int(sid))
            X = load_npz_shard(p, x_key="X", target_T=args.target_T, downsample_factor=args.downsample_factor)
            shard_data[v][int(sid)] = X

        # quick check
        any_sid = needed_sids[0]
        print(f"[{v}] example shard {any_sid}: X.shape={shard_data[v][any_sid].shape}, dtype={shard_data[v][any_sid].dtype}")

    # -------- Build per-trial row index list and meta dict for shard planning
    # Use df index (0..n-1) as global row id.
    trial_to_rowidx_all = df.groupby("trial_id", sort=False).apply(lambda s: s.index.to_list()).to_dict()

    trial_meta_all = {}
    for _, r in trials_df.iterrows():
        tid = r["trial_id"]
        trial_meta_all[tid] = {
            "label": int(r["label"]),
            "n_samples": int(r["n_samples"]),
            "n_rx": int(r["n_rx"]),
        }

    # -------- Process each split: plan shards -> write npz -> write index.csv
    for split_name, trial_set in split_map.items():
        df_split = df[df["trial_id"].isin(trial_set)].copy()
        df_split.reset_index(drop=False, inplace=True)  # keep original df index in column "index"
        df_split.rename(columns={"index": "orig_row_index"}, inplace=True)

        # rebuild mappings restricted to this split
        trial_to_rowidx = defaultdict(list)
        trial_meta = {}
        for tid, grp in df_split.groupby("trial_id", sort=False):
            # these are row indices in df_split (0..len-1), but we need to gather from df_split itself.
            trial_to_rowidx[tid] = grp.index.to_list()
            trial_meta[tid] = {
                "label": int(grp["label"].iloc[0]),
                "n_samples": int(len(grp)),
            }

        # plan shards on df_split (using df_split indices)
        print(f"[{split_name}] planning shards: rows={len(df_split)}, trials={len(trial_to_rowidx)}")
        shard_plans = build_shard_plan(
            df_rows=df_split,
            trial_to_rowidx=trial_to_rowidx,
            trial_meta=trial_meta,
            shard_size=args.shard_size,
            seed=args.seed + (0 if split_name == "train" else 9991),
        )
        print(f"[{split_name}] planned {len(shard_plans)} shards, shard_size cap={args.shard_size}")

        # Prepare output dirs
        out_root_split = out / split_name
        (out_root_split / "meta").mkdir(parents=True, exist_ok=True)
        for v in variants:
            (out_root_split / v / "shards").mkdir(parents=True, exist_ok=True)
        copy_meta(root, out_root_split)

        # Build index records in main thread (deterministic order)
        index_records = []

        # Write shards (threaded)
        def job_write_shard(new_sid: int, row_idx: np.ndarray):
            # gather X for each variant
            X_out = {}
            for v in variants:
                X_out[v] = gather_X_for_rows(df_split, row_idx, shard_data[v], x_key="X")

            # labels from df_split
            y = df_split.loc[row_idx, "label"].to_numpy(dtype=np.int64).reshape(-1, 1)

            # write npz files
            for v in variants:
                out_npz = out_root_split / v / "shards" / f"shard-{new_sid:05d}.npz"
                write_one_shard(out_npz, X_out[v], y)

        futures = []
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            for new_sid, row_idx in enumerate(shard_plans):
                futures.append(ex.submit(job_write_shard, new_sid, row_idx))

                # build index rows for this shard
                sub = df_split.loc[row_idx].copy()
                sub["shard_id"] = int(new_sid)
                sub["offset"] = np.arange(len(sub), dtype=np.int64)

                # keep original columns if present; ensure required columns exist
                # We write a superset: original columns + new shard/offset + rx_parsed (debug)
                cols_prefer = [
                    "sample_id", "date", "user", "rx", "a", "b", "c", "d",
                    "gesture_name", "label", "raw_relpath", "raw_len",
                    "shard_id", "offset"
                ]
                extra_cols = [c for c in cols_prefer if c in sub.columns]
                # always include required
                for c in ["sample_id", "label", "gesture_name", "shard_id", "offset"]:
                    if c not in extra_cols and c in sub.columns:
                        extra_cols.append(c)

                # add debug trace
                if "orig_row_index" in sub.columns:
                    sub["orig_row_index"] = sub["orig_row_index"].astype(int)
                    if "orig_row_index" not in extra_cols:
                        extra_cols.append("orig_row_index")
                if "rx_parsed" in sub.columns and "rx_parsed" not in extra_cols:
                    extra_cols.append("rx_parsed")
                if "trial_id" in sub.columns and "trial_id" not in extra_cols:
                    extra_cols.append("trial_id")

                index_records.append(sub[extra_cols])

            # wait all writes
            for fu in as_completed(futures):
                fu.result()

        # write index.csv (already in shard-id order, within shard offset order)
        index_df = pd.concat(index_records, axis=0, ignore_index=True)
        out_index = out_root_split / "meta" / "index.csv"
        index_df.to_csv(out_index, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"[{split_name}] wrote index.csv: {out_index} rows={len(index_df)}")

        # quick split stats
        print(f"[{split_name}] label dist:", (index_df["label"].value_counts().sort_index() / len(index_df)).round(4).to_dict())

    print("\nDone.")
    print(f"Train root: {out / 'train'}")
    print(f"Test  root: {out / 'test'}")
    print("\nTraining tip (fast IO): use split='all' and DataLoader(shuffle=False).")


if __name__ == "__main__":
    main()
