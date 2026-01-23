#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline FISTA reconstruction -> save reconstructed shards for Widar_digit_{amp,conj}.

Why DCT by default?
- Your "amp" variant is real-valued. DCT-II (orthonormal) is a real, orthonormal transform
  that often gives strong energy compaction for smooth / piecewise-smooth time signals and
  avoids the "periodic boundary" assumption of FFT.
- With an orthonormal transform, the proximal operator is exact and simple:
    prox_{λ||Wx||_1}(v) = W^T soft(Wv, λ)
- You can still use FFT (Doppler) sparsity with --prior fft. That’s closer to “Doppler sparsity”
  but note: real-to-complex rFFT is not a square orthonormal matrix; we use a practical shrink
  in the rFFT domain (common baseline), which is typically good enough for comparisons.

Input layout (same as your WidarDigitShardDataset expects):
  IN_ROOT/
    train/
      amp/shards/shard-00000.npz ...
      conj/shards/...
      meta/index.csv
    test/
      ...

Output layout:
  OUT_ROOT/
    train/
      amp/shards/shard-00000.npz ...   (same shard ids & offsets as input)
      meta/index.csv                   (copied)
    test/
      ...

Example:
  python gen_widar_digit_fista_shards_v2.py \
    --in_root /home/cxy/data/code/datasets/sense-fi/Widar_digit \
    --out_root /home/cxy/data/code/datasets/sense-fi/Widar_digit_fista_sr0.25_poisson_lam0.02_it40_dct \
    --variant amp --sample_rate 0.25 --sample_method poisson \
    --prior dct --lam 0.02 --niter 40 --batch_size 128 --seed 0 --device cuda
"""
from __future__ import annotations

import argparse
import os
import shutil
import math
from typing import Tuple, List

import numpy as np
import torch


# ----------------------------
# Fast orthonormal DCT-II / IDCT (DCT-III) via FFT
# Orthonormal scaling matches SciPy dct(x, type=2, norm="ortho")
# ----------------------------
def _dct_u(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized DCT-II (last dim)."""
    N = x.shape[-1]
    x_flat = x.reshape(-1, N)
    v = torch.cat([x_flat, x_flat.flip([1])], dim=1)  # (B, 2N)
    Vc = torch.fft.fft(v, dim=1)
    k = torch.arange(N, device=x.device, dtype=torch.float32)
    ang = -math.pi * k / (2.0 * N)
    W = torch.cos(ang) + 1j * torch.sin(ang)
    V = Vc[:, :N] * W
    y = V.real * 0.5
    return y.reshape(*x.shape)

def _idct_u(X: torch.Tensor) -> torch.Tensor:
    """Inverse of _dct_u (unnormalized IDCT-III), last dim."""
    N = X.shape[-1]
    X_flat = X.reshape(-1, N)
    k = torch.arange(N, device=X.device, dtype=torch.float32)
    ang = math.pi * k / (2.0 * N)
    W = torch.cos(ang) + 1j * torch.sin(ang)
    V = torch.complex(X_flat, torch.zeros_like(X_flat)) * (2.0 * W)
    zero = torch.zeros((V.shape[0], 1), device=V.device, dtype=V.dtype)
    V_full = torch.cat([V, zero, V[:, 1:].flip([1]).conj()], dim=1)  # (B, 2N)
    v = torch.fft.ifft(V_full, dim=1).real
    return v[:, :N].reshape(*X.shape)

def dct_ortho(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal DCT-II (last dim)."""
    N = x.shape[-1]
    X = _dct_u(x)
    X0 = X[..., :1] / math.sqrt(N)
    Xk = X[..., 1:] * math.sqrt(2.0 / N)
    return torch.cat([X0, Xk], dim=-1)

def idct_ortho(Xo: torch.Tensor) -> torch.Tensor:
    """Inverse of dct_ortho (orthonormal IDCT)."""
    N = Xo.shape[-1]
    X0 = Xo[..., :1] * math.sqrt(N)
    Xk = Xo[..., 1:] * math.sqrt(N / 2.0)
    X = torch.cat([X0, Xk], dim=-1)
    return _idct_u(X)


def soft_threshold_real(x: torch.Tensor, thr: float) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(torch.abs(x) - thr, min=0.0)

def soft_threshold_complex(z: torch.Tensor, thr: float, eps: float = 1e-8) -> torch.Tensor:
    """Complex soft threshold: shrink magnitude."""
    mag = torch.abs(z)
    scale = torch.clamp(1.0 - thr / (mag + eps), min=0.0)
    return z * scale


def prox_l1_dct(v: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Prox for lam * ||DCT(v)||_1 under orthonormal DCT:
      prox(v) = IDCT( soft( DCT(v), lam ) )
    v: (..., T) real
    """
    V = dct_ortho(v)
    V = soft_threshold_real(V, lam)
    return idct_ortho(V)

def prox_l1_rfft(v: torch.Tensor, lam: float, eps: float = 1e-8) -> torch.Tensor:
    """
    Practical prox-like step using rFFT-domain complex soft-thresholding.
    v: (..., T) real
    """
    V = torch.fft.rfft(v, dim=-1, norm="ortho")
    V = soft_threshold_complex(V, lam, eps=eps)
    return torch.fft.irfft(V, n=v.shape[-1], dim=-1, norm="ortho")


@torch.no_grad()
def fista_recon_time(
    y: torch.Tensor,
    mask: torch.Tensor,
    lam: float = 0.02,
    niter: int = 40,
    prior: str = "dct",
) -> torch.Tensor:
    """
    Solve:  min_x 0.5 || M(x - y) ||_2^2 + lam * || W(x) ||_1
    where M is a {0,1} mask (same shape as y), W is DCT or rFFT along time.
    Using FISTA with step=1 (Lipschitz constant of M is 1).

    y, mask: (B*, T) float32
    return: x (B*, T)
    """
    if prior not in ("dct", "fft"):
        raise ValueError(f"prior must be dct|fft, got {prior}")

    prox = prox_l1_dct if prior == "dct" else prox_l1_rfft

    # init with zero-filled observation
    x = y.clone()
    z = x.clone()
    t = 1.0

    for _ in range(int(niter)):
        # grad of 0.5||M(z-y)||^2 is M(z-y)
        grad = (z - y) * mask
        x_next = prox(z - grad, lam)

        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        z = x_next + ((t - 1.0) / t_next) * (x_next - x)
        x = x_next
        t = t_next

    # hard data consistency at known points
    x = x * (1.0 - mask) + y * mask
    return x


# ----------------------------
# Mask generation (time sampling)
# ----------------------------
def _pick_indices(T: int, K: int, method: str, rng: np.random.Generator) -> np.ndarray:
    K = int(max(2, min(T, K)))
    if method in ("uniform_nearest", "equidistant"):
        idx = np.round(np.linspace(0, T - 1, K)).astype(np.int64)
    elif method == "gaussian":
        mu = (T - 1) / 2.0
        sigma = max(1.0, T / 6.0)
        idx = np.round(rng.normal(loc=mu, scale=sigma, size=K)).astype(np.int64)
        idx = np.clip(idx, 0, T - 1)
    elif method == "poisson":
        # mimic your snippet: exponential intervals normalized to total duration
        intervals = rng.exponential(scale=1.0, size=K - 1)
        intervals = intervals / max(intervals.sum(), 1e-12) * (T - 1)
        pick = np.concatenate(([0.0], np.cumsum(intervals)))
        idx = np.round(pick).astype(np.int64)
        idx = np.clip(idx, 0, T - 1)
    else:
        raise ValueError(f"Unknown sample_method={method}")

    # enforce endpoints + uniqueness + fill up if duplicates reduced count
    idx = np.unique(idx)
    if idx.size == 0:
        idx = np.array([0, T - 1], dtype=np.int64)
    idx[0] = 0
    idx[-1] = T - 1

    # pad with random indices if needed
    while idx.size < K:
        extra = rng.integers(0, T, size=(K - idx.size,), dtype=np.int64)
        idx = np.unique(np.concatenate([idx, extra]))
    idx = np.sort(idx)[:K]
    idx[0] = 0
    idx[-1] = T - 1
    return idx


def make_time_mask(T: int, F: int, sample_rate: float, method: str,
                   seed: int, shard_id: int, offset: int) -> np.ndarray:
    """
    Return mask of shape (T, F), float32 {0,1}, same mask along F.
    Deterministic per (seed, shard_id, offset).
    """
    sr = float(sample_rate)
    if sr >= 1.0:
        return np.ones((T, F), dtype=np.float32)
    K = int(round(T * sr))
    # deterministic RNG per-sample
    s = (int(seed) * 1000003) ^ (int(shard_id) * 10007) ^ (int(offset) * 97)
    rng = np.random.default_rng(s & 0xFFFFFFFF)
    idx = _pick_indices(T, K, method, rng)
    m_t = np.zeros((T,), dtype=np.float32)
    m_t[idx] = 1.0
    return np.repeat(m_t[:, None], F, axis=1)


# ----------------------------
# Shard IO helpers
# ----------------------------
def load_shard(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    npz = np.load(npz_path, allow_pickle=True)
    X = npz["X"]  # (N,T,F) or (N,1,T,F)
    y = npz["y"]
    npz.close()
    if X.ndim == 4 and X.shape[1] == 1:
        X = X[:, 0]
    if X.ndim != 3:
        raise ValueError(f"Unexpected X shape {X.shape} in {npz_path}")
    return X.astype(np.float32, copy=False), y.astype(np.int64, copy=False)


def save_shard(npz_path: str, X: np.ndarray, y: np.ndarray):
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez_compressed(npz_path, X=X.astype(np.float32, copy=False), y=y.astype(np.int64, copy=False))


def copy_meta(in_split_root: str, out_split_root: str):
    in_meta = os.path.join(in_split_root, "meta")
    out_meta = os.path.join(out_split_root, "meta")
    if os.path.isdir(out_meta):
        return
    if not os.path.isdir(in_meta):
        raise FileNotFoundError(f"meta dir not found: {in_meta}")
    shutil.copytree(in_meta, out_meta)


def list_shards(shard_dir: str) -> List[str]:
    if not os.path.isdir(shard_dir):
        raise FileNotFoundError(f"shard dir not found: {shard_dir}")
    names = [n for n in os.listdir(shard_dir) if n.endswith(".npz") and n.startswith("shard-")]
    names.sort()
    return [os.path.join(shard_dir, n) for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", required=True, help="Original Widar_digit root")
    ap.add_argument("--out_root", required=True, help="Output root for reconstructed dataset")
    ap.add_argument("--variant", default="amp", choices=["amp", "conj"])
    ap.add_argument("--splits", default="train,test", help="Comma-separated splits to process")
    ap.add_argument("--sample_rate", type=float, default=0.25)
    ap.add_argument("--sample_method", type=str, default="poisson",
                    choices=["uniform_nearest", "equidistant", "gaussian", "poisson"])
    ap.add_argument("--prior", type=str, default="dct", choices=["dct", "fft"],
                    help="Sparsity prior transform along time: dct (real, orthonormal) or fft (rfft magnitude shrink)")
    ap.add_argument("--lam", type=float, default=0.02)
    ap.add_argument("--niter", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda", help="cpu | cuda | cuda:N (e.g., cuda:0, cuda:1)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output shards")
    args = ap.parse_args()

    in_root = os.path.abspath(args.in_root)
    out_root = os.path.abspath(args.out_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    variant = args.variant

    # parse device: cpu | cuda | cuda:N
    dev_str = args.device.strip().lower()

    if dev_str == "cpu":
        device = torch.device("cpu")
    elif dev_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available, but --device is set to cuda.")
        # allow "cuda" or "cuda:0" / "cuda:1"
        device = torch.device(dev_str if ":" in dev_str else "cuda")
    else:
        raise ValueError(f"Invalid --device {args.device}. Use cpu|cuda|cuda:N")

    print(f"[device] {device}  prior={args.prior}  lam={args.lam}  niter={args.niter}")
    if device.type == "cuda":
        torch.cuda.set_device(device)  # ensure current device
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        print(f"[gpu] cuda:{idx}  {name}")

    for split in splits:
        in_split_root = os.path.join(in_root, split)
        out_split_root = os.path.join(out_root, split)
        in_shard_dir = os.path.join(in_split_root, variant, "shards")
        out_shard_dir = os.path.join(out_split_root, variant, "shards")
        os.makedirs(out_shard_dir, exist_ok=True)
        copy_meta(in_split_root, out_split_root)

        shard_paths = list_shards(in_shard_dir)
        print(f"[{split}/{variant}] shards={len(shard_paths)}  sample_rate={args.sample_rate}  method={args.sample_method}")

        for sp in shard_paths:
            shard_name = os.path.basename(sp)
            shard_id = int(shard_name.split("-")[1].split(".")[0])
            out_sp = os.path.join(out_shard_dir, shard_name)
            if (not args.overwrite) and os.path.exists(out_sp):
                print(f"  - skip exists: {out_sp}")
                continue

            X, y = load_shard(sp)  # X: (N,T,F)
            N, T, F = X.shape
            X_out = np.empty_like(X, dtype=np.float32)

            # process in mini-batches
            bs = int(args.batch_size)
            for s0 in range(0, N, bs):
                s1 = min(N, s0 + bs)
                xb = X[s0:s1]  # (B,T,F)
                B = xb.shape[0]

                # per-sample mask (B,T,F)
                masks = np.stack(
                    [make_time_mask(T, F, args.sample_rate, args.sample_method, args.seed, shard_id, off)
                     for off in range(s0, s1)],
                    axis=0
                )  # (B,T,F)
                yb = xb * masks  # zero-filled observation

                # torch -> (B*F, T)
                y_t = torch.from_numpy(yb).to(device=device, dtype=torch.float32)  # (B,T,F)
                m_t = torch.from_numpy(masks).to(device=device, dtype=torch.float32)
                y_tf = y_t.permute(0, 2, 1).contiguous().view(B * F, T)
                m_tf = m_t.permute(0, 2, 1).contiguous().view(B * F, T)

                x_tf = fista_recon_time(y_tf, m_tf, lam=float(args.lam), niter=int(args.niter), prior=args.prior)

                # back to numpy (B,T,F)
                x_t = x_tf.view(B, F, T).permute(0, 2, 1).contiguous()
                X_out[s0:s1] = x_t.detach().cpu().numpy()

            save_shard(out_sp, X_out, y)
            print(f"  - wrote: {out_sp}  shape={X_out.shape}")

    print("\nDone.")
    print("Training tip: point your dataloader root_dir to OUT_ROOT, set sample_rate=1.0, use_mask_0=0, is_rec=0.")


if __name__ == "__main__":
    main()
