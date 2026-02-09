import os  # 引入 os 模块
# ==================== 解决 num_workers 和 numpy 的冲突 ====================
#为了设置进程为单线程，减少cpu占用
# 明确控制 OpenBLAS 的线程数，这是你当前NumPy的主要后端
os.environ['OPENBLAS_NUM_THREADS'] = '1'
# OpenBLAS 内部使用 OpenMP，所以这个也很重要
os.environ['OMP_NUM_THREADS'] = '1'
# 由于你的numpy没有链接MKL，这两个变量可以不设，但设了也无害，可以保留以防万一未来更改环境
os.environ['MKL_NUM_THREADS'] = '1'
# 这个通常是macOS特有的，Linux服务器基本用不到，但保留无害
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
# 如果你确定不使用BLIS，这个可以不设，保留也无害
os.environ['BLIS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1' # 有时也需要这个
# =========================================================================
import numpy as np
import torch
#为了设置进程为单线程，减少cpu占用
torch.set_num_threads(1)
import torch.nn as nn
import argparse
from util import load_data_n_model
import time
import csv # 1. 引入 csv 模块
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast  # 记得加上这个 import
from skimage.metrics import structural_similarity as ssim_func

# BGI masks path (edit as needed)
BGI_MASK_PT = "/home/cxy/data/code/datasets/sense-fi/Widar_digit/mask_10_90Hz_random/synthetic_masks_bgi_0p2_rate25.pt"


def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def is_main():
    return get_rank() == 0



# train_one_epoch 和 test_one_epoch 函数与上一个回答中的版本相同
# 这里为了完整性再次包含它们

'''def train_one_epoch(model, tensor_loader, criterion, device, optimizer):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    num_samples = 0

    for data in tensor_loader:
        inputs, labels = data
        inputs = inputs.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)
        #labels = labels.type(torch.LongTensor)

        optimizer.zero_grad()
        outputs = model(inputs)
        #outputs = outputs.to(device)
        #outputs = outputs.type(torch.FloatTensor)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)
        predict_y = torch.argmax(outputs, dim=1)
        epoch_accuracy += (predict_y == labels).sum().item()
        num_samples += labels.size(0)

    epoch_loss = epoch_loss / num_samples
    epoch_accuracy = epoch_accuracy / num_samples
    return epoch_loss, epoch_accuracy'''
use_amp = True  # 控制是否启用 AMP（默认启用）

def train_one_epoch(
    model, tensor_loader, criterion, device, optimizer,
    is_rec: int = 0, criterion_rec=None, alpha: float = 0.5,
    lam_miss=2.0,beta=0.1,log_parts=False,
    grad_check=False           # 是否检查梯度/参数是否在更新（debug 用）
):
    model.train()
    epoch_loss = 0.0
    epoch_correct = 0
    num_samples = 0
    # --- [新增] loss 分量统计 ---
    sum_ce = 0.0
    sum_miss_term = 0.0
    sum_known_term = 0.0
    sum_miss_ratio = 0.0
    sum_scale = 0.0
    sum_mse_all_equiv = 0.0
    part_cnt = 0


    first_param_before = None
    if grad_check:
        # 用来验证 optimizer 真的在更新参数
        first_param_before = next(model.parameters()).detach().clone()

    for batch in tensor_loader:
        # ---- 1) 搬到 device（保持 dtype 正确）
        optimizer.zero_grad(set_to_none=True)
        if int(is_rec) == 0:
            inputs, labels = batch
            inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            '''outputs = model(inputs)
            loss = criterion(outputs, labels)'''
            # 使用 autocast 来启用 bf16
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(inputs)
            loss = criterion(outputs, labels)
        else:
            inputs, mask, labels, inputs_gt = batch
            inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
            mask = mask.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            inputs_gt = inputs_gt.to(device, dtype=torch.float32, non_blocking=True)
            # 使用 autocast 来启用 bf16
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                outputs, x_recon = model(inputs, mask)
                # loss = criterion(outputs, labels) + float(alpha) * criterion_rec(x_recon, inputs_gt)
                # 将 loss 计算放在 fp32
            x_recon_fp32 = x_recon.float()
            inputs_gt_fp32 = inputs_gt.float()
            m = mask.to(dtype=x_recon_fp32.dtype).clamp(0.0, 1.0)
            miss = 1.0 - m
            diff = (x_recon_fp32 - inputs_gt_fp32)
            mse_miss = (diff.mul(miss)).pow(2).sum() / (miss.sum() + 1e-8)
            mse_known = (diff.mul(m)).pow(2).sum() / (m.sum() + 1e-8)

            # miss_ratio = Nmiss / Nall（这是你说的“占比随采样率变”的关键）
            Nall = float(m.numel())
            miss_ratio = miss.sum() / (Nall + 1e-8)
            known_ratio = 1.0 - miss_ratio
            ce = criterion(outputs.float(), labels)
            loss = ce + lam_miss *  (miss_ratio * mse_miss)  + beta * (known_ratio * mse_known)
            #loss = ce + lam_miss *  (mse_miss)  + beta * (mse_known)

            if log_parts:
                # miss_ratio = Nmiss / Nall
                miss_ratio = (miss.sum() / (m.numel() + 1e-8)).detach()
                scale = ((m.numel()) / (miss.sum() + 1e-8)).detach()  # Nall/Nmiss
                mse_all_equiv = (mse_miss * miss_ratio).detach()      # ≈ old MSE_all (known误差≈0时)

                sum_ce += ce.detach().float().item()
                sum_miss_term += (lam_miss * mse_miss).detach().float().item()
                sum_known_term += (beta * mse_known).detach().float().item()
                sum_miss_ratio += miss_ratio.float().item()
                sum_scale += scale.float().item()
                sum_mse_all_equiv += mse_all_equiv.float().item()
                part_cnt += 1

        loss.backward()
        optimizer.step()

        # ---- 4) 统计 epoch 指标
        bs = inputs.size(0)
        epoch_loss += loss.item() * bs
        pred = outputs.argmax(dim=1)
        epoch_correct += (pred == labels).sum().item()
        num_samples += bs


    if num_samples == 0:
        return 0.0, 0.0
    if is_dist():
        t = torch.tensor([epoch_loss, epoch_correct, num_samples],
                         device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        epoch_loss, epoch_correct, num_samples = t.tolist()
    epoch_loss = epoch_loss / num_samples
    epoch_accuracy = epoch_correct / num_samples

    if grad_check and first_param_before is not None:
        with torch.no_grad():
            first_param_after = next(model.parameters()).detach()
            delta = (first_param_after - first_param_before).abs().mean().item()
        if is_main():print(f"[grad_check] first_param abs-mean delta = {delta:.6e}")
    # --- [新增] DDP 聚合 + 打印 ---
    if log_parts and part_cnt > 0:
        if is_dist():
            t = torch.tensor(
                [sum_ce, sum_miss_term, sum_known_term, sum_miss_ratio, sum_scale, sum_mse_all_equiv, float(part_cnt)],
                device=device, dtype=torch.float64
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            sum_ce, sum_miss_term, sum_known_term, sum_miss_ratio, sum_scale, sum_mse_all_equiv, part_cnt = t.tolist()

        if is_main():
            denom = max(1.0, part_cnt)
            ce_m = sum_ce / denom
            miss_m = sum_miss_term / denom
            known_m = sum_known_term / denom
            mr = sum_miss_ratio / denom
            sc = sum_scale / denom
            mse_all_eq = sum_mse_all_equiv / denom
            # old 0.5*MSE_all 等效项
            old_like = 0.5 * mse_all_eq
            print(
                f"[loss_parts] miss_ratio={mr:.4f}  Nall/Nmiss={sc:.2f}  "
                f"CE={ce_m:.4f}  lam*mse_miss={miss_m:.4f}  beta*mse_known={known_m:.4f}  "
                f"mse_all_equiv={mse_all_eq:.6f}  old_like(0.5*MSE_all)={old_like:.6f}"
            )

    return epoch_loss, epoch_accuracy


def test_one_epoch(model, tensor_loader, criterion, device,
                   is_rec: int = 0, criterion_rec=None, alpha: float = 0.5,lam_miss=2.0,beta=0.1,):
    model.eval()
    total_loss, total_correct, num_samples = 0.0, 0, 0

    with torch.no_grad():
        for batch in tensor_loader:
            if int(is_rec) == 0:
                inputs, labels = batch
                inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
                labels = labels.to(device, dtype=torch.long, non_blocking=True)
                # 使用 autocast 启用 bf16
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    outputs = model(inputs)
                loss = criterion(outputs, labels)

            else:
                inputs, mask, labels, inputs_gt = batch
                inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
                mask = mask.to(device, dtype=torch.float32, non_blocking=True)
                labels = labels.to(device, dtype=torch.long, non_blocking=True)
                inputs_gt = inputs_gt.to(device, dtype=torch.float32, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    outputs, x_recon = model(inputs, mask)
                    # loss = criterion(outputs, labels) + float(alpha) * criterion_rec(x_recon, inputs_gt)
                # 将 loss 计算放在 fp32
                x_recon_fp32 = x_recon.float()
                inputs_gt_fp32 = inputs_gt.float()
                m = mask.to(dtype=x_recon_fp32.dtype).clamp(0.0, 1.0)
                miss = 1.0 - m
                diff = (x_recon_fp32 - inputs_gt_fp32)
                mse_miss = (diff.mul(miss)).pow(2).sum() / (miss.sum() + 1e-8)
                mse_known = (diff.mul(m)).pow(2).sum() / (m.sum() + 1e-8)
                ce = criterion(outputs, labels)
                Nall = float(m.numel())
                miss_ratio = miss.sum() / (Nall + 1e-8)
                known_ratio = 1.0 - miss_ratio
                loss = ce + lam_miss * (miss_ratio * mse_miss) + beta * (known_ratio * mse_known)
                #loss = ce + lam_miss * mse_miss + beta * mse_known
            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            num_samples += bs
        if num_samples == 0:
            return 0.0, 0.0
        if is_dist():
            t = torch.tensor([total_loss, float(total_correct), float(num_samples)],
                             device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total_loss, total_correct, num_samples = t.tolist()
        return total_loss / num_samples, total_correct / num_samples


def _spectrogram_mean(x_tf, n_fft=256, hop=128, win=256):
    # x_tf: (B, T, F) -> average over F, then STFT over T
    x_mean = x_tf.mean(dim=-1)  # (B, T)
    window = torch.hann_window(win, device=x_mean.device, dtype=x_mean.dtype)
    spec = torch.stft(x_mean, n_fft=n_fft, hop_length=hop, win_length=win,
                      window=window, return_complex=True)
    spec_mag = spec.abs()  # (B, Fbins, Frames)
    return spec_mag


def _calculate_ssim_standard(img1, img2):
    """
    Standard SSIM with min-max normalization to [0,1].
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    min_val = min(img1.min(), img2.min())
    max_val = max(img1.max(), img2.max())
    range_val = max_val - min_val + 1e-8
    img1 = (img1 - min_val) / range_val
    img2 = (img2 - min_val) / range_val
    return float(ssim_func(img1, img2, data_range=1.0))


def _pcc_global(x, y, eps=1e-8):
    # x, y: 1D numpy arrays
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    vx = x - x.mean()
    vy = y - y.mean()
    denom = (np.sqrt((vx * vx).sum()) * np.sqrt((vy * vy).sum())) + eps
    return float((vx * vy).sum() / denom)


def test_one_epoch_with_metrics(model, tensor_loader, criterion, device,
                                is_rec: int = 0, criterion_rec=None, alpha: float = 0.5,
                                lam_miss=2.0, beta=0.1):
    model.eval()
    total_loss, total_correct, num_samples = 0.0, 0, 0
    sum_ssim = 0.0
    sum_pcc = 0.0
    sum_log_nmse = 0.0
    metric_cnt = 0

    with torch.no_grad():
        for batch in tensor_loader:
            if int(is_rec) == 0:
                inputs, labels = batch
                inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
                labels = labels.to(device, dtype=torch.long, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    outputs = model(inputs)
                loss = criterion(outputs, labels)
            else:
                inputs, mask, labels, inputs_gt = batch
                inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
                mask = mask.to(device, dtype=torch.float32, non_blocking=True)
                labels = labels.to(device, dtype=torch.long, non_blocking=True)
                inputs_gt = inputs_gt.to(device, dtype=torch.float32, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    outputs, x_recon = model(inputs, mask)
                x_recon_fp32 = x_recon.float()
                inputs_gt_fp32 = inputs_gt.float()
                m = mask.to(dtype=x_recon_fp32.dtype).clamp(0.0, 1.0)
                miss = 1.0 - m
                diff = (x_recon_fp32 - inputs_gt_fp32)
                mse_miss = (diff.mul(miss)).pow(2).sum() / (miss.sum() + 1e-8)
                mse_known = (diff.mul(m)).pow(2).sum() / (m.sum() + 1e-8)
                ce = criterion(outputs, labels)
                Nall = float(m.numel())
                miss_ratio = miss.sum() / (Nall + 1e-8)
                known_ratio = 1.0 - miss_ratio
                loss = ce + lam_miss * (miss_ratio * mse_miss) + beta * (known_ratio * mse_known)

                # metrics: SSIM over spectrograms, PCC over waveform, log NMSE on missing points
                # x_recon_fp32 / inputs_gt_fp32: (B,1,T,F)
                x_rec_tf = x_recon_fp32.squeeze(1)  # (B,T,F)
                x_gt_tf = inputs_gt_fp32.squeeze(1)  # (B,T,F)
                spec_rec = _spectrogram_mean(x_rec_tf)
                spec_gt = _spectrogram_mean(x_gt_tf)
                spec_rec_np = spec_rec.detach().cpu().numpy()
                spec_gt_np = spec_gt.detach().cpu().numpy()
                x_rec_np = x_rec_tf.detach().cpu().numpy().reshape(x_rec_tf.size(0), -1)
                x_gt_np = x_gt_tf.detach().cpu().numpy().reshape(x_gt_tf.size(0), -1)

                # per-sample metrics
                for i in range(x_rec_tf.size(0)):
                    sum_ssim += _calculate_ssim_standard(spec_gt_np[i], spec_rec_np[i])
                    sum_pcc += _pcc_global(x_gt_np[i], x_rec_np[i])

                # log NMSE on missing points
                num = (diff.mul(miss)).pow(2).flatten(1).sum(1)
                den = (inputs_gt_fp32.mul(miss)).pow(2).flatten(1).sum(1) + 1e-8
                log_nmse = 10.0 * torch.log10(num / den + 1e-8)
                sum_log_nmse += log_nmse.sum().item()
                metric_cnt += x_rec_tf.size(0)

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            num_samples += bs

        if num_samples == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        if is_dist():
            t = torch.tensor(
                [total_loss, float(total_correct), float(num_samples),
                 sum_ssim, sum_pcc, sum_log_nmse, float(metric_cnt)],
                device=device, dtype=torch.float64
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total_loss = t[0].item()
            total_correct = t[1].item()
            num_samples = t[2].item()
            sum_ssim = t[3].item()
            sum_pcc = t[4].item()
            sum_log_nmse = t[5].item()
            metric_cnt = t[6].item()

        acc = total_correct / num_samples
        loss_mean = total_loss / num_samples
        if metric_cnt == 0:
            return loss_mean, acc, 0.0, 0.0, 0.0
        return loss_mean, acc, sum_ssim / metric_cnt, sum_pcc / metric_cnt, sum_log_nmse / metric_cnt


def save_metrics_to_csv(filepath, history):
    """
    将性能历史记录（一个字典列表）保存到CSV文件。
    Args:
        filepath (str): CSV文件的完整路径。
        history (list of dict): 包含 'epoch', 'loss', 'accuracy' 的字典列表。
    """
    if not history:
        return

    # 使用 'w' 模式打开文件，newline='' 是csv模块的推荐做法
    with open(filepath, 'w', newline='') as f:
        # 定义CSV文件的列名，与history字典中的键对应
        fieldnames = ['epoch', 'loss', 'accuracy']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # 写入表头
        writer.writeheader()
        # 写入所有行数据
        writer.writerows(history)


def run_per_rate_eval(model, test_loader, criterion, device, args, ddp, criterion_rec=None):
    # per-rate evaluation for trafficlike masks
    if not hasattr(test_loader.dataset, "get_available_rates"):
        return
    rates = test_loader.dataset.get_available_rates()
    if not rates:
        return
    rate_history = []
    if is_main():
        print("Running per-rate evaluation...")
    for r in rates:
        if hasattr(test_loader.dataset, "set_rate_filter"):
            test_loader.dataset.set_rate_filter(r)
        if hasattr(test_loader.dataset, "set_eval_subset"):
            test_loader.dataset.set_eval_subset(len(test_loader.dataset), seed=1000 + int(r))
        r_loss, r_acc, r_ssim, r_pcc, r_log_nmse = test_one_epoch_with_metrics(
            model, test_loader, criterion, device,
            is_rec=args.is_rec, criterion_rec=criterion_rec, alpha=args.rec_alpha,
            lam_miss=args.lam_miss, beta=args.beta
        )
        if is_main():
            print(f"[rate {r}] Loss: {r_loss:.5f}, Accuracy: {r_acc:.4f}, "
                  f"SSIM: {r_ssim:.4f}, PCC: {r_pcc:.4f}, logNMSE_miss: {r_log_nmse:.4f}")
        rate_history.append({
            'rate_hz': int(r),
            'loss': r_loss,
            'accuracy': r_acc,
            'ssim': r_ssim,
            'pcc': r_pcc,
            'log_nmse_miss': r_log_nmse
        })
    if hasattr(test_loader.dataset, "set_rate_filter"):
        test_loader.dataset.set_rate_filter(None)
    if hasattr(test_loader.dataset, "set_eval_subset"):
        test_loader.dataset.set_eval_subset(len(test_loader.dataset), seed=0)
    if is_main():
        rate_path = os.path.join(args.metrics_save_dir, 'test_metrics_by_rate.csv')
        with open(rate_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['rate_hz', 'loss', 'accuracy', 'ssim', 'pcc', 'log_nmse_miss']
            )
            writer.writeheader()
            writer.writerows(rate_history)
        print(f"Saved per-rate metrics to: {rate_path}")


def run_per_bgi_eval(model, test_loader, criterion, device, args, ddp, criterion_rec=None, bgi_mask_pt=None):
    if not bgi_mask_pt:
        return
    if not hasattr(test_loader.dataset, "set_eval_masks"):
        return
    payload = torch.load(bgi_mask_pt, map_location="cpu")
    masks = payload.get("masks", None)
    bgi_bin = payload.get("bgi_bin", None)
    if masks is None or bgi_bin is None:
        raise ValueError("bgi_mask_pt must contain 'masks' and 'bgi_bin'")
    test_loader.dataset.set_eval_masks(masks, bgi_bin=bgi_bin)

    bins = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    bin_history = []
    if is_main():
        print("Running per-BGI-bin evaluation...")
    for i, b in enumerate(bins):
        if hasattr(test_loader.dataset, "set_bgi_bin_filter"):
            test_loader.dataset.set_bgi_bin_filter(b)
        if hasattr(test_loader.dataset, "set_eval_subset"):
            test_loader.dataset.set_eval_subset(len(test_loader.dataset), seed=2000 + i)
        r_loss, r_acc, r_ssim, r_pcc, r_log_nmse = test_one_epoch_with_metrics(
            model, test_loader, criterion, device,
            is_rec=args.is_rec, criterion_rec=criterion_rec, alpha=args.rec_alpha,
            lam_miss=args.lam_miss, beta=args.beta
        )
        if is_main():
            print(f"[bin {b}] Loss: {r_loss:.5f}, Accuracy: {r_acc:.4f}, "
                  f"SSIM: {r_ssim:.4f}, PCC: {r_pcc:.4f}, logNMSE_miss: {r_log_nmse:.4f}")
        bin_history.append({
            'bgi_bin': b,
            'loss': r_loss,
            'accuracy': r_acc,
            'ssim': r_ssim,
            'pcc': r_pcc,
            'log_nmse_miss': r_log_nmse
        })

    if hasattr(test_loader.dataset, "set_bgi_bin_filter"):
        test_loader.dataset.set_bgi_bin_filter(None)
    if hasattr(test_loader.dataset, "set_eval_subset"):
        test_loader.dataset.set_eval_subset(len(test_loader.dataset), seed=0)
    if hasattr(test_loader.dataset, "clear_eval_masks"):
        test_loader.dataset.clear_eval_masks()

    if is_main():
        bin_path = os.path.join(args.metrics_save_dir, 'test_metrics_by_bgi.csv')
        with open(bin_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['bgi_bin', 'loss', 'accuracy', 'ssim', 'pcc', 'log_nmse_miss']
            )
            writer.writeheader()
            writer.writerows(bin_history)
        print(f"Saved per-BGI-bin metrics to: {bin_path}")

def main():
    root = '../datasets/sense-fi/'
    if not os.path.isdir(root):
        if is_main():
            print(f"错误: 数据集根目录 '{root}' 未找到。")
            print("请确认您的脚本（run.py）是否在 'code/sense-fi/' 文件夹下，")
            print("并且 'datasets' 文件夹在 'code/' 的上一级目录。")
        return
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
    parser.add_argument('--dataset', choices = ['UT_HAR_data','NTU-Fi-HumanID','NTU-Fi_HAR','Widar','Widar_digit_amp','Widar_digit_conj'])
    parser.add_argument('--model', choices = ['MLP','LeNet','ResNet18','ResNet50','ResNet101','RNN','GRU','LSTM','BiLSTM', 'CNN+GRU','ViT'])
    # 新增的参数，用于自定义实验名称，并设为必填项
    #parser.add_argument('--exp_name', required=True, type=str, help='自定义实验名称，将用于创建模型保存目录。')
    parser.add_argument('--sample_rate', type=float, default=1.0, help='二次降采样的比例 (0.05到1.0)，对应25Hz到500Hz。默认为1.0，即不进行二次采样。')
    parser.add_argument('--sample_method', type=str,default='uniform_nearest',choices=['uniform_nearest', 'equidistant', 'gaussian', 'poisson', 'trafficlike'],help='降采样方法。默认为 "uniform_nearest"。')
    parser.add_argument('--interpolation', type=str,default='linear',choices=['linear', 'cubic', 'nearest', 'idw', 'rbf','spline','akima'],help='升采样时使用的插值方法。默认为 "linear"。')
    parser.add_argument('--use_energy_input', type=int, default=1, choices=[0, 1],help='是否使用能量信息 (1:是, 0:否)。默认为 1 (是)。')
    parser.add_argument('--use_mask_0', type=int, default=0, choices=[0, 1 , 2],help='是否使用 mask_0 (1:是, 0:否,2:不mask直接return降采样后的)。默认为 0 (否)。')
    parser.add_argument('--traffic_train_pt', type=str, default='/home/cxy/data/code/datasets/sense-fi/Widar_digit/mask_10_90Hz_random/train.pt', help='trafficlike train masks .pt')
    parser.add_argument('--traffic_test_pt', type=str, default='/home/cxy/data/code/datasets/sense-fi/Widar_digit/mask_10_90Hz_random/test.pt', help='trafficlike test masks .pt')
    # 新增两个参数，用于接收完整的保存目录
    parser.add_argument('--model_save_dir', required=True, type=str, help='模型检查点的完整保存目录。')
    parser.add_argument('--metrics_save_dir', required=True, type=str, help='性能指标文件的完整保存目录。')
    parser.add_argument('--is_rec', type=int, default=0, choices=[0, 1], help='1: 重建+分类；0: 仅分类')
    parser.add_argument('--rec_alpha', type=float, default=0.5, help='重建损失权重')
    parser.add_argument('--csdc_blocks', type=int, default=1, help='重建blocks数量')
    parser.add_argument('--rec_model', type=str, default='csdc', choices=['csdc', 'istanet','mabf','mabf_c','mabf_1d_mix','mabf2','fista', 'fista_fft', 'fista_dct','fista_blockfft'], help='重建模型类型')
    parser.add_argument('--global_batch_size', type=int, default=128, help='全局batch(所有GPU加起来)')
    parser.add_argument('--num_workers_train', type=int, default=6)
    parser.add_argument('--num_workers_test', type=int, default=2)
    parser.add_argument('--lam_miss', type=float, default=1.0, help='重建损失中缺失部分的权重')
    parser.add_argument('--beta', type=float, default=0.0, help='重建损失中已知部分的权重')
    parser.add_argument('--test_only', action='store_true', help='仅测试，不训练')
    parser.add_argument('--ckpt_path', type=str, default=None, help='测试用模型权重路径(.pth)')
    parser.add_argument('--eval_rate', action='store_true', help='测试：按采样率分桶')
    parser.add_argument('--eval_bgi', action='store_true', help='测试：按BGI分桶')
    args = parser.parse_args()
    # ---- DDP init (torchrun 会设置这些环境变量) ----
    ddp = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)
    if ddp:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        rank = 0

    # ---- 全局128 => 每卡 batch = 128/world_size ----
    if args.global_batch_size % world_size != 0:
        raise ValueError(f"global_batch_size={args.global_batch_size} 不能被 world_size={world_size} 整除")
    per_gpu_bs = args.global_batch_size // world_size
    train_loader, test_loader, model, train_epoch = load_data_n_model(
        args.dataset, args.model, root,
        args.sample_rate, args.sample_method, args.interpolation,
        args.use_energy_input, args.use_mask_0,
        args.is_rec, args.csdc_blocks, args.rec_model,
        batch_size=per_gpu_bs,
        num_workers_train=args.num_workers_train,
        num_workers_test=args.num_workers_test,
        distributed=ddp, rank=rank, world_size=world_size,
        traffic_train_pt=args.traffic_train_pt,
        traffic_test_pt=args.traffic_test_pt
    )

    #train_loader, test_loader, model, train_epoch = load_data_n_model(args.dataset, args.model, root,args.sample_rate, args.sample_method ,args.interpolation,args.use_energy_input ,args.use_mask_0 ,args.is_rec,args.csdc_blocks)
    criterion = nn.CrossEntropyLoss()
    criterion_rec = nn.MSELoss(reduction='mean') if args.is_rec else None

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # load checkpoint if provided
    if args.ckpt_path:
        state = torch.load(args.ckpt_path, map_location=device)
        if ddp:
            model.module.load_state_dict(state, strict=True)
        else:
            model.load_state_dict(state, strict=True)
        if is_main():
            print(f"Loaded checkpoint: {args.ckpt_path}")

    if args.test_only:
        if not args.ckpt_path:
            raise ValueError("test_only requires --ckpt_path")
        # 固定一份验证掩码子集（用于稳定对比）
        if hasattr(test_loader.dataset, "set_rate_filter"):
            test_loader.dataset.set_rate_filter(None)
        if hasattr(test_loader.dataset, "set_eval_subset"):
            test_loader.dataset.set_eval_subset(len(test_loader.dataset), seed=0)
        test_loss, test_acc = test_one_epoch(model, test_loader, criterion, device,
                                             is_rec=args.is_rec, criterion_rec=criterion_rec,
                                             alpha=args.rec_alpha, lam_miss=args.lam_miss, beta=args.beta)
        if is_main():
            print(f"Test/Validation -> Loss: {test_loss:.5f}, Accuracy: {test_acc:.4f}")
        if args.eval_rate:
            run_per_rate_eval(model, test_loader, criterion, device, args, ddp, criterion_rec=criterion_rec)
        if args.eval_bgi:
            run_per_bgi_eval(model, test_loader, criterion, device, args, ddp, criterion_rec=criterion_rec, bgi_mask_pt=BGI_MASK_PT)
        if ddp:
            dist.destroy_process_group()
        return


    # --- 目录创建 ---
    # 现在 run.py 只负责确保目录存在，不再构建它
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.metrics_save_dir, exist_ok=True)
    if is_main():
        print(f"✅ 模型将保存至: {os.path.abspath(args.model_save_dir)}")
        print(f"📊 性能指标将保存至: {os.path.abspath(args.metrics_save_dir)}")
    # ================================================================
    # 2. 计算保存间隔和保存点
    num_saves = 4
    if train_epoch < num_saves:
        # 如果总epoch数小于10，则每个epoch都保存
        save_interval = 1
    else:
        save_interval = train_epoch // num_saves

    # 创建一个包含所有需要保存的epoch编号的集合，方便快速查找
    save_epochs = set(range(save_interval, train_epoch + 1, save_interval))
    # 确保最后一个epoch总是被保存
    save_epochs.add(train_epoch)

    if is_main():print(f"模型将会在以下Epoch结束时保存: {sorted(list(save_epochs))}")
    # ==========================================================

    # ==================== 4. 新增：初始化历史记录列表 ====================
    train_history = []
    test_history = []

    # 固定一份验证掩码子集（用于早停更稳定）
    if hasattr(test_loader.dataset, "set_rate_filter"):
        test_loader.dataset.set_rate_filter(None)
    if hasattr(test_loader.dataset, "set_eval_subset"):
        test_loader.dataset.set_eval_subset(len(test_loader.dataset), seed=0)

    # [新增] 早停相关的变量
    best_test_acc = 0.0  # 记录历史最佳准确率
    patience = 20  # 容忍度：如果 20 个 epoch 没提升就停止
    patience_counter = 0  # 计数器
    # ===================================================
    # --- 训练主循环 ---
    total_train_start = time.time()
    for epoch in range(1, train_epoch + 1):  # 循环从1开始，方便与epoch编号对应
        if ddp and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        if is_main():print(f"--- Epoch {epoch}/{train_epoch} ---")
        epoch_start = time.time()
        log_parts = (epoch <= 3)# 前3个epoch打印loss分量
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, device, optimizer,
                                                is_rec=args.is_rec, criterion_rec=criterion_rec, alpha=args.rec_alpha,lam_miss=args.lam_miss,beta=args.beta,log_parts=log_parts)
        if is_main():print(f"Train -> Loss: {train_loss:.5f}, Accuracy: {train_acc:.4f}")

        test_loss, test_acc = test_one_epoch(model, test_loader, criterion, device,
                                             is_rec=args.is_rec, criterion_rec=criterion_rec, alpha=args.rec_alpha, lam_miss=args.lam_miss,beta=args.beta)
        if is_main():print(f"Test/Validation -> Loss: {test_loss:.5f}, Accuracy: {test_acc:.4f}")
        if is_main():
            epoch_time = time.time() - epoch_start
            print(f"Epoch time: {epoch_time:.2f} s")

        # ==================== 5. 新增：收集当前epoch的数据 ====================
        train_history.append({'epoch': epoch, 'loss': train_loss, 'accuracy': train_acc})
        test_history.append({'epoch': epoch, 'loss': test_loss, 'accuracy': test_acc})

        # ==================== [核心修改] 早停与最佳模型保存 ====================
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0  # 重置计数器

            # 保存最佳模型 (覆盖式保存，始终只有一个 best_model.pth)
            # 只有当 Epoch > 10 以后，才真正开始执行保存硬盘的操作
            if epoch > 20:
                best_model_path = os.path.join(args.model_save_dir, 'best_model.pth')
                if is_main():
                    state = model.module.state_dict() if ddp else model.state_dict()
                    torch.save(state, best_model_path)
                    print(f"🌟 新纪录！最佳模型已保存 (Acc: {best_test_acc:.4f})")
            else:
                if is_main():print(f"🌟 新纪录 (Acc: {best_test_acc:.4f}) - 训练初期暂不保存")

        else:
            # 同样，前 10 个 Epoch 也不消耗 patience（宽容期）
            if epoch > 20:
                patience_counter += 1
                if is_main():print(f"⚠️ 性能未提升 ({patience_counter}/{patience})")

        # 检查是否需要停止
        if patience_counter >= patience:
            if is_main():
                print(f"\n🛑 触发早停机制！测试集准确率已连续 {patience} 个 Epoch 未提升。")
                print(f"   当前最佳准确率: {best_test_acc:.4f}")
                print(f"   在 Epoch {epoch} 停止训练。")
            break  # 跳出 for 循环
        # ===================================================================

        # --- 检查是否到达保存点 ---#前面有保存最佳模型了，所以这里不再保存。
        '''if epoch in save_epochs:
            model_save_path = os.path.join(args.model_save_dir, f'model_epoch_{epoch}.pth')
            print(f"💾 到达保存点，正在保存模型到: {model_save_path}")
            torch.save(model.state_dict(), model_save_path)'''

    total_train_end = time.time()
    if is_main():
        print("\n--- 训练完成 ---")
        print(f"⏱️ 总训练耗时：{total_train_end - total_train_start:.2f} 秒")

    # 使用新的目录参数来构建路径
    train_metrics_path = os.path.join(args.metrics_save_dir, 'train_metrics.csv')
    test_metrics_path = os.path.join(args.metrics_save_dir, 'test_metrics.csv')

    if is_main():
        print(f"📊 正在保存训练历史到: {train_metrics_path}")
        save_metrics_to_csv(train_metrics_path, train_history)

    if is_main():
        print(f"📊 正在保存测试历史到: {test_metrics_path}")
        save_metrics_to_csv(test_metrics_path, test_history)

    if args.eval_rate:
        run_per_rate_eval(model, test_loader, criterion, device, args, ddp, criterion_rec=criterion_rec)
    if args.eval_bgi:
        run_per_bgi_eval(model, test_loader, criterion, device, args, ddp, criterion_rec=criterion_rec, bgi_mask_pt=BGI_MASK_PT)

    #print(f"💾 所有检查点已保存在目录: {args.model_save_dir}")
    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
