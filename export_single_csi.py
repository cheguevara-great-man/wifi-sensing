import os
import numpy as np
import torch

try:
    import scipy.io as sio
except Exception as exc:
    raise ImportError("scipy is required for .mat I/O. Please install scipy.") from exc

from Widar_digit_model import Widar_digit_amp_model
from dataset import _interp_from_mask

# ==== hardcoded inputs (edit these) ====
MAT_CSI_PATH = "path/to/csi.mat"
MAT_CSI_KEY = "csi"
MAT_MASK_PATH = "/home/cxy/data/code/datasets/sense-fi/Widar_digit/mask_10_90Hz_random/picked_bgi5_rate25_masks.mat"
# either provide a single mask under MAT_MASK_KEY, or (N,T) masks under MAT_MASKS_KEY
MAT_MASK_KEY = "mask"
MAT_MASKS_KEY = "masks"
MAT_BGI_KEY = "bgi_bin"
OUT_DIR = "out_single_csi"

# classifier used in training (adjust if needed)
CLASSIFIER_NAME = "ResNet18"
NUM_CLASSES = 10

# model checkpoints per mode (edit these)
MODES = {
    "interp_mabf": {
        "rec_model": "mabf_c",
        "csdc_blocks": 1,
        "ckpt": "path/to/interp_mabf.pth",
        "input": "interp",
    },
    "zf_mabf": {
        "rec_model": "mabf_c",
        "csdc_blocks": 1,
        "ckpt": "path/to/zf_mabf.pth",
        "input": "zf",
    },
    "zf_fista": {
        "rec_model": "fista",
        "csdc_blocks": 30,
        "ckpt": "path/to/zf_fista.pth",
        "input": "zf",
    },
    "zf_istanet": {
        "rec_model": "istanet",
        "csdc_blocks": 9,
        "ckpt": "path/to/zf_istanet.pth",
        "input": "zf",
    },
    "interp_cls": {
        "rec_model": None,
        "ckpt": None,
        "input": "interp",
    },
    "zf_cls": {
        "rec_model": None,
        "ckpt": None,
        "input": "zf",
    },
}

# optional: run only one mode (set to key in MODES)
ONLY_MODE = None


def _load_mat_var(path, key):
    data = sio.loadmat(path)
    if key not in data:
        raise KeyError(f"{key} not found in {path}. keys={list(data.keys())}")
    return data[key]


def _normalize_csi_shape(csi):
    csi = np.asarray(csi)
    csi = np.squeeze(csi)
    if csi.ndim == 2:
        if csi.shape[1] != 90:
            raise ValueError(f"Expected csi shape (T,90), got {csi.shape}")
        return csi.astype(np.float32, copy=False)
    if csi.ndim == 3:
        # allow (3,30,T) or (T,3,30) or (T,30,3)
        if csi.shape[0] == 3 and csi.shape[1] == 30:
            csi = np.transpose(csi, (2, 0, 1))  # (T,3,30)
            csi = csi.reshape(csi.shape[0], 3 * 30)
            return csi.astype(np.float32, copy=False)
        if csi.shape[1] == 3 and csi.shape[2] == 30:
            csi = csi.reshape(csi.shape[0], 3 * 30)
            return csi.astype(np.float32, copy=False)
        if csi.shape[1] == 30 and csi.shape[2] == 3:
            csi = np.transpose(csi, (0, 2, 1))  # (T,3,30)
            csi = csi.reshape(csi.shape[0], 3 * 30)
            return csi.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported csi shape: {csi.shape}")


def _infer_output_layout(csi_raw):
    csi_raw = np.asarray(csi_raw)
    csi_raw = np.squeeze(csi_raw)
    if csi_raw.ndim == 3 and csi_raw.shape[0] == 3 and csi_raw.shape[1] == 30:
        return "ast"  # (A,S,T)
    return "t90"


def _format_output(csi_t90, layout):
    if layout == "ast":
        T = csi_t90.shape[0]
        csi_ast = csi_t90.reshape(T, 3, 30).transpose(1, 2, 0)  # (3,30,T)
        return csi_ast
    return csi_t90


def _normalize_mask(mask, T):
    mask = np.asarray(mask).astype(np.float32)
    mask = np.squeeze(mask)
    if mask.ndim != 1:
        raise ValueError(f"Expected mask shape (T,), got {mask.shape}")
    if mask.shape[0] != T:
        raise ValueError(f"Mask length {mask.shape[0]} != T {T}")
    return mask


def _bgi_bins_from_mat(bgi_bin):
    bgi_bin = np.asarray(bgi_bin).reshape(-1)
    out = []
    for v in bgi_bin:
        if isinstance(v, bytes):
            out.append(v.decode("utf-8"))
        elif isinstance(v, str):
            out.append(v)
        elif isinstance(v, np.ndarray):
            if v.dtype.kind in ("U", "S"):
                out.append(str(v.item()))
            else:
                out.append(str(v))
        else:
            out.append(str(v))
    return out


def _sanitize_tag(s: str) -> str:
    s = s.strip()
    s = s.replace(".", "p")
    s = s.replace("-", "_")
    s = s.replace(" ", "")
    return s


def _prepare_inputs(csi_gt, mask_1d, mode):
    mask_2d = mask_1d[:, None].repeat(csi_gt.shape[1], axis=1)
    x_masked = csi_gt * mask_2d
    if mode == "zf":
        x_in = x_masked
    elif mode == "interp":
        x_in = _interp_from_mask(x_masked, mask_1d, "linear")
    else:
        raise ValueError(f"Unknown input mode: {mode}")
    return x_masked, x_in, mask_2d


def _save_mat(path, csi, layout):
    csi_out = _format_output(csi, layout)
    sio.savemat(path, {"csi": csi_out.astype(np.float32, copy=False)})


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    csi = _load_mat_var(MAT_CSI_PATH, MAT_CSI_KEY)
    out_layout = _infer_output_layout(csi)
    mask_payload = sio.loadmat(MAT_MASK_PATH)
    if MAT_MASKS_KEY in mask_payload:
        masks = mask_payload[MAT_MASKS_KEY]
        bgi_bin = mask_payload.get(MAT_BGI_KEY, None)
    else:
        masks = _load_mat_var(MAT_MASK_PATH, MAT_MASK_KEY)
        bgi_bin = None
    csi_gt = _normalize_csi_shape(csi)
    T = csi_gt.shape[0]
    masks = np.asarray(masks)
    if masks.ndim == 1:
        masks = masks[None, :]
    if masks.shape[0] == T and masks.shape[1] != T:
        masks = masks.T
    if masks.shape[1] != T:
        raise ValueError(f"Expected masks shape (N,T) with T={T}, got {masks.shape}")
    if bgi_bin is None:
        bin_labels = [f"mask{i}" for i in range(masks.shape[0])]
    else:
        bin_labels = _bgi_bins_from_mat(bgi_bin)
        if len(bin_labels) != masks.shape[0]:
            raise ValueError(f"bgi_bin length {len(bin_labels)} != masks {masks.shape[0]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for bin_label, mask_row in zip(bin_labels, masks):
        mask_1d = _normalize_mask(mask_row, T)
        bin_tag = _sanitize_tag(str(bin_label))
        for name, cfg in MODES.items():
            if ONLY_MODE is not None and name != ONLY_MODE:
                continue
            x_masked, x_in, mask_2d = _prepare_inputs(csi_gt, mask_1d, cfg["input"])
            x_rec = x_in
            if cfg["rec_model"] is not None:
                model = Widar_digit_amp_model(
                    model_name=CLASSIFIER_NAME,
                    num_classes=NUM_CLASSES,
                    T=T,
                    is_rec=1,
                    csdc_blocks=cfg.get("csdc_blocks", 1),
                    rec_model=cfg["rec_model"],
                ).to(device)
                ckpt = cfg.get("ckpt", None)
                if not ckpt:
                    raise ValueError(f"Missing ckpt path for mode {name}")
                state = torch.load(ckpt, map_location=device)
                model.load_state_dict(state, strict=False)
                model.eval()
                with torch.no_grad():
                    x_t = torch.from_numpy(x_in).unsqueeze(0).unsqueeze(0).to(device)
                    m_t = torch.from_numpy(mask_2d).unsqueeze(0).unsqueeze(0).to(device)
                    _, x_recon = model(x_t, m_t)
                    x_rec = x_recon.squeeze(0).squeeze(0).cpu().numpy()

            rec_path = os.path.join(OUT_DIR, f"{name}_{bin_tag}_rec.mat")
            _save_mat(rec_path, x_rec, out_layout)
            print(f"[{name} | {bin_label}] saved: {os.path.basename(rec_path)}")


if __name__ == "__main__":
    main()
