#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd


def load_npz(path: Path):
    d = np.load(path, allow_pickle=False)
    if "X" not in d.files or "y" not in d.files:
        raise KeyError(f"{path} missing keys. has={d.files}")
    X = d["X"]
    y = d["y"]
    return X, y


def get_npz_path(root: Path, split: str, variant: str, shard_id: int) -> Path:
    return root / split / variant / "shards" / f"shard-{shard_id:05d}.npz"


def verify_one_shard(new_root: Path, split: str, variant: str, shard_id: int,
                     df_new: pd.DataFrame, df_orig: pd.DataFrame,
                     max_checks: int = 50, atol: float = 1e-6):
    npz_path = get_npz_path(new_root, split, variant, shard_id)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    Xn, yn = load_npz(npz_path)

    # basic shape checks
    assert Xn.ndim == 3, f"{npz_path}: X.ndim={Xn.ndim}, shape={Xn.shape}"
    assert yn.ndim in (1, 2), f"{npz_path}: y.ndim={yn.ndim}, shape={yn.shape}"
    if yn.ndim == 2:
        assert yn.shape[1] == 1, f"{npz_path}: y second dim should be 1, got {yn.shape}"
        yn1 = yn[:, 0]
    else:
        yn1 = yn
    assert Xn.shape[0] == yn1.shape[0], f"{npz_path}: N mismatch X={Xn.shape}, y={yn.shape}"

    # index.csv rows for this shard
    sub = df_new[df_new["shard_id"] == shard_id].copy()
    assert len(sub) == Xn.shape[0], f"{npz_path}: index rows={len(sub)} != npz N={Xn.shape[0]}"
    assert sub["offset"].min() == 0, f"{npz_path}: offset min != 0"
    assert sub["offset"].max() == len(sub) - 1, f"{npz_path}: offset max != N-1"
    # label check with y
    off = sub["offset"].to_numpy(dtype=np.int64)
    lab_csv = sub["label"].to_numpy(dtype=np.int64)
    lab_npz = yn1[off].astype(np.int64)
    if not np.array_equal(lab_csv, lab_npz):
        bad = np.where(lab_csv != lab_npz)[0][:10]
        raise AssertionError(f"{npz_path}: label mismatch at rows {bad.tolist()}")

    # strong check: compare X with original by sample_id
    # (randomly check up to max_checks samples from this shard)
    sample_ids = sub["sample_id"].astype(str).tolist()
    k = min(max_checks, len(sample_ids))
    pick = random.sample(range(len(sample_ids)), k=k)

    # build quick lookup in original index.csv by sample_id
    # assumes sample_id unique
    orig_lookup = df_orig.set_index("sample_id")[["shard_id", "offset", "label"]]

    # load original shard(s) lazily with a tiny cache
    orig_cache = {}

    for i in pick:
        sid = sample_ids[i]
        new_off = int(sub.iloc[i]["offset"])
        new_label = int(sub.iloc[i]["label"])

        if sid not in orig_lookup.index:
            raise AssertionError(f"sample_id not found in orig index.csv: {sid}")

        o_shard = int(orig_lookup.loc[sid, "shard_id"])
        o_off = int(orig_lookup.loc[sid, "offset"])
        o_label = int(orig_lookup.loc[sid, "label"])

        if o_label != new_label:
            raise AssertionError(f"label changed for {sid}: orig={o_label} new={new_label}")

        # load original shard npz
        o_npz_path = new_root.parent / "__DUMMY__"  # placeholder, will override below
        # original root is sibling param, so pass separately:
        # we'll infer original root from df_orig metadata via caller
        # (set by caller through closure variable) -> simplest: store on function attribute
        raise RuntimeError("Internal: original root not set")

    return {
        "variant": variant,
        "shard_id": shard_id,
        "N": int(Xn.shape[0]),
        "labels_unique": int(len(np.unique(lab_csv))),
        "label_counts": dict(pd.Series(lab_csv).value_counts().sort_index().to_dict()),
    }


def verify_sample_content(new_root: Path, orig_root: Path, split: str, variant: str,
                          shard_ids: list[int], df_new: pd.DataFrame, df_orig: pd.DataFrame,
                          max_checks_per_shard: int, atol: float):
    """
    Same as verify_one_shard but with correct original root wired in.
    """
    results = []
    # original shard cache across checks
    orig_cache = {}

    def load_orig_shard(variant: str, shard_id: int):
        key = (variant, shard_id)
        if key in orig_cache:
            return orig_cache[key]
        p = orig_root / variant / "shards" / f"shard-{shard_id:05d}.npz"
        if not p.exists():
            # fallback try +1/-1 for 0/1-index mismatch
            p2 = orig_root / variant / "shards" / f"shard-{(shard_id+1):05d}.npz"
            p3 = orig_root / variant / "shards" / f"shard-{max(shard_id-1,0):05d}.npz"
            if p2.exists():
                p = p2
            elif p3.exists():
                p = p3
            else:
                raise FileNotFoundError(f"cannot find orig shard for {variant} shard_id={shard_id}")
        X, y = load_npz(p)
        # normalize y to (N,)
        if y.ndim == 2:
            y = y[:, 0]
        orig_cache[key] = (X, y)
        return X, y

    # lookup on original index
    orig_lookup = df_orig.set_index("sample_id")[["shard_id", "offset", "label"]]

    for shard_id in shard_ids:
        npz_path = new_root / split / variant / "shards" / f"shard-{shard_id:05d}.npz"
        Xn, yn = load_npz(npz_path)
        if yn.ndim == 2:
            yn1 = yn[:, 0]
        else:
            yn1 = yn

        sub = df_new[df_new["shard_id"] == shard_id].copy()
        assert len(sub) == Xn.shape[0], f"{npz_path}: index rows={len(sub)} != npz N={Xn.shape[0]}"
        assert sub["offset"].min() == 0 and sub["offset"].max() == len(sub) - 1

        off = sub["offset"].to_numpy(dtype=np.int64)
        lab_csv = sub["label"].to_numpy(dtype=np.int64)
        lab_npz = yn1[off].astype(np.int64)
        assert np.array_equal(lab_csv, lab_npz), f"{npz_path}: label mismatch between csv and npz y"

        # strong X check
        sample_ids = sub["sample_id"].astype(str).tolist()
        k = min(max_checks_per_shard, len(sample_ids))
        pick = random.sample(range(len(sample_ids)), k=k)

        for i in pick:
            sid = sample_ids[i]
            new_off = int(sub.iloc[i]["offset"])
            new_label = int(sub.iloc[i]["label"])

            if sid not in orig_lookup.index:
                raise AssertionError(f"sample_id not found in orig index.csv: {sid}")

            o_shard = int(orig_lookup.loc[sid, "shard_id"])
            o_off = int(orig_lookup.loc[sid, "offset"])
            o_label = int(orig_lookup.loc[sid, "label"])
            assert o_label == new_label, f"{sid}: label changed orig={o_label} new={new_label}"

            Xo, yo = load_orig_shard(variant, o_shard)

            # normalize original X to (N,T,F)
            if Xo.ndim == 4 and Xo.shape[1] == 1:
                Xo2 = Xo[:, 0, :, :]
            else:
                Xo2 = Xo
            xo = Xo2[o_off]
            xn = Xn[new_off]

            # handle possible T=2000 vs 500
            if xo.shape[0] == 2000 and xn.shape[0] == 500:
                xo = xo[::4, :]

            if not np.allclose(xo, xn, atol=atol, rtol=0):
                # report max diff
                md = float(np.max(np.abs(xo - xn)))
                raise AssertionError(f"{sid}: X mismatch (variant={variant}) max_abs_diff={md}")

        results.append({
            "variant": variant,
            "split": split,
            "shard_id": shard_id,
            "N": int(Xn.shape[0]),
            "unique_labels": int(len(np.unique(lab_csv))),
        })

    return results


def check_trial_leakage(train_index: Path, test_index: Path):
    df_tr = pd.read_csv(train_index)
    df_te = pd.read_csv(test_index)
    if "trial_id" not in df_tr.columns or "trial_id" not in df_te.columns:
        print("[WARN] trial_id not in index.csv, skip leakage check.")
        return
    tr = set(df_tr["trial_id"].astype(str).unique().tolist())
    te = set(df_te["trial_id"].astype(str).unique().tolist())
    inter = tr & te
    print(f"[trial leakage] train_trials={len(tr)} test_trials={len(te)} intersect={len(inter)}")
    if len(inter) > 0:
        # print a few examples
        ex = list(inter)[:10]
        raise AssertionError(f"Trial leakage detected! examples={ex}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_root", type=str, required=True, help="original root: .../Widar_digit")
    ap.add_argument("--new_root", type=str, required=True, help="new root: .../Widar_digit_resharded")
    ap.add_argument("--split", type=str, choices=["train", "test", "both"], default="both")
    ap.add_argument("--num_shards", type=int, default=3, help="how many shards to sample per split")
    ap.add_argument("--checks_per_shard", type=int, default=50, help="how many samples to strong-check per shard")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--atol", type=float, default=1e-6)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    orig_root = Path(args.orig_root).resolve()
    new_root = Path(args.new_root).resolve()

    # load original index (filtered raw_len>=1000 already in your stats; keep full for lookup)
    df_orig = pd.read_csv(orig_root / "meta" / "index.csv")
    df_orig.columns = [c.strip() for c in df_orig.columns]
    if "sample_id" not in df_orig.columns:
        raise RuntimeError("orig index.csv missing sample_id")

    # optional: leakage check
    if args.split in ("both",):
        check_trial_leakage(new_root / "train" / "meta" / "index.csv",
                            new_root / "test" / "meta" / "index.csv")

    splits = ["train", "test"] if args.split == "both" else [args.split]
    variants = ["amp", "conj"]

    all_results = []

    for sp in splits:
        df_new = pd.read_csv(new_root / sp / "meta" / "index.csv")
        df_new.columns = [c.strip() for c in df_new.columns]

        # pick shard ids to sample
        shard_ids = sorted(df_new["shard_id"].unique().tolist())
        k = min(args.num_shards, len(shard_ids))
        pick_shards = random.sample(shard_ids, k=k)

        print(f"\n=== Split={sp} pick_shards={pick_shards} ===")

        for v in variants:
            res = verify_sample_content(
                new_root=new_root,
                orig_root=orig_root,
                split=sp,
                variant=v,
                shard_ids=pick_shards,
                df_new=df_new,
                df_orig=df_orig,
                max_checks_per_shard=args.checks_per_shard,
                atol=args.atol
            )
            all_results.extend(res)
            for r in res:
                print(f"[OK] {r['split']}/{r['variant']} shard={r['shard_id']:05d} "
                      f"N={r['N']} unique_labels={r['unique_labels']}")

    print("\nAll checks passed ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
