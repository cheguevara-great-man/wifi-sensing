#本文件为为了测试NMSE而修改的run.py版本，其他部分与run.py相同
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


def nmse_miss_num_den(x_hat, x_gt, mask, eps=1e-12):
    """
    NMSE_miss = ||(1-M)⊙(x_hat-x_gt)||_F^2 / (||(1-M)⊙x_gt||_F^2 + eps)
    这里 mask == M: 观测点为1，缺失点为0
    返回：num, den (torch.float64 标量张量)
    """
    miss = (1.0 - mask)
    diff = (x_hat - x_gt) * miss
    num = (diff * diff).sum(dtype=torch.float64)
    den = ((x_gt * miss) * (x_gt * miss)).sum(dtype=torch.float64)
    return num, den + eps

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

def train_one_epoch(
    model, tensor_loader, criterion, device, optimizer,
    is_rec: int = 0, criterion_rec=None, alpha: float = 0.5,
    grad_check=False           # 是否检查梯度/参数是否在更新（debug 用）
):
    model.train()
    epoch_loss = 0.0
    epoch_correct = 0
    num_samples = 0

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
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        else:
            inputs, mask, labels, inputs_gt = batch
            inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
            mask = mask.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            inputs_gt = inputs_gt.to(device, dtype=torch.float32, non_blocking=True)

            outputs, x_recon = model(inputs, mask)
            loss = criterion(outputs, labels) + float(alpha) * criterion_rec(x_recon, inputs_gt)

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

    return epoch_loss, epoch_accuracy


def test_one_epoch(model, tensor_loader, criterion, device,
                   is_rec: int = 0, criterion_rec=None, alpha: float = 0.5,compute_nmse_miss: bool = True):
    model.eval()
    total_loss, total_correct, num_samples = 0.0, 0, 0
    nmse_num = torch.zeros(1, device=device, dtype=torch.float64)
    nmse_den = torch.zeros(1, device=device, dtype=torch.float64)

    with torch.no_grad():
        for batch in tensor_loader:
            if int(is_rec) == 0:
                inputs, labels = batch
                inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
                labels = labels.to(device, dtype=torch.long, non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

            else:
                inputs, mask, labels, inputs_gt = batch
                inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
                mask = mask.to(device, dtype=torch.float32, non_blocking=True)
                labels = labels.to(device, dtype=torch.long, non_blocking=True)
                inputs_gt = inputs_gt.to(device, dtype=torch.float32, non_blocking=True)

                outputs, x_recon = model(inputs, mask)
                if compute_nmse_miss:
                    num_i, den_i = nmse_miss_num_den(x_recon, inputs_gt, mask)
                    nmse_num += num_i
                    nmse_den += den_i

                loss = criterion(outputs, labels) + float(alpha) * criterion_rec(x_recon,  inputs_gt)

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

            if compute_nmse_miss:
                dist.all_reduce(nmse_num, op=dist.ReduceOp.SUM)
                dist.all_reduce(nmse_den, op=dist.ReduceOp.SUM)

        test_loss = total_loss / num_samples
        test_acc  = total_correct / num_samples

        if compute_nmse_miss:
            nmse_miss = (nmse_num / nmse_den).item()
            return test_loss, test_acc, nmse_miss
        else:
            return test_loss, test_acc



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
    parser.add_argument('--sample_method', type=str,default='uniform_nearest',choices=['uniform_nearest', 'equidistant', 'gaussian', 'poisson'],help='降采样方法。默认为 "uniform_nearest"。')
    parser.add_argument('--interpolation', type=str,default='linear',choices=['linear', 'cubic', 'nearest', 'idw', 'rbf','spline','akima'],help='升采样时使用的插值方法。默认为 "linear"。')
    parser.add_argument('--use_energy_input', type=int, default=1, choices=[0, 1],help='是否使用能量信息 (1:是, 0:否)。默认为 1 (是)。')
    parser.add_argument('--use_mask_0', type=int, default=0, choices=[0, 1 , 2],help='是否使用 mask_0 (1:是, 0:否,2:不mask直接return降采样后的)。默认为 0 (否)。')
    # 新增两个参数，用于接收完整的保存目录
    parser.add_argument('--model_save_dir', required=True, type=str, help='模型检查点的完整保存目录。')
    parser.add_argument('--metrics_save_dir', required=True, type=str, help='性能指标文件的完整保存目录。')
    parser.add_argument('--is_rec', type=int, default=0, choices=[0, 1], help='1: 重建+分类；0: 仅分类')
    parser.add_argument('--rec_alpha', type=float, default=0.5, help='重建损失权重')
    parser.add_argument('--csdc_blocks', type=int, default=1, help='重建blocks数量')
    parser.add_argument('--rec_model', type=str, default='csdc', choices=['csdc', 'istanet'], help='重建模型类型')
    parser.add_argument('--global_batch_size', type=int, default=128, help='全局batch(所有GPU加起来)')
    parser.add_argument('--num_workers_train', type=int, default=6)
    parser.add_argument('--num_workers_test', type=int, default=2)
    parser.add_argument('--eval_only', action='store_true', help='只跑测试，不训练')
    parser.add_argument('--ckpt_path', type=str, default='', help='要加载的模型权重路径（默认用 model_save_dir/best_model.pth）')

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
        distributed=ddp, rank=rank, world_size=world_size
    )

    #train_loader, test_loader, model, train_epoch = load_data_n_model(args.dataset, args.model, root,args.sample_rate, args.sample_method ,args.interpolation,args.use_energy_input ,args.use_mask_0 ,args.is_rec,args.csdc_blocks)
    criterion = nn.CrossEntropyLoss()
    criterion_rec = nn.MSELoss(reduction='mean') if args.is_rec else None

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ===== eval only: load checkpoint and run test =====
    if args.eval_only:
        ckpt_path = args.ckpt_path.strip()
        if ckpt_path == '':
            ckpt_path = os.path.join(args.model_save_dir, 'best_model.pth')
        if is_main():
            print(f"🧪 Eval-only mode. Loading ckpt from: {ckpt_path}")

        state = torch.load(ckpt_path, map_location='cpu')
        if ddp:
            model.module.load_state_dict(state, strict=True)
        else:
            model.load_state_dict(state, strict=True)

        # 只在 is_rec=1 且 dataloader 提供 (inputs, mask, labels, inputs_gt) 时才有意义
        out = test_one_epoch(model, test_loader, criterion, device,
                             is_rec=args.is_rec, criterion_rec=criterion_rec, alpha=args.rec_alpha,
                             compute_nmse_miss=True)

        if len(out) == 3:
            test_loss, test_acc, nmse_miss = out
            if is_main():
                print(f"[Eval] Loss={test_loss:.5f}, Acc={test_acc:.4f}, NMSE_miss={nmse_miss:.6e}")
        else:
            test_loss, test_acc = out
            if is_main():
                print(f"[Eval] Loss={test_loss:.5f}, Acc={test_acc:.4f}")
                print("⚠️ NMSE_miss 未计算：需要 --is_rec=1 且 test batch 含 mask 和 gt。")

        if ddp:
            dist.destroy_process_group()
        return

    # ===== training mode =====
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



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
        if is_main():print(f"--- Epoch {epoch}/{train_epoch} ---")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, device, optimizer,
                                                is_rec=args.is_rec, criterion_rec=criterion_rec, alpha=args.rec_alpha)
        if is_main():print(f"Train -> Loss: {train_loss:.5f}, Accuracy: {train_acc:.4f}")

        test_loss, test_acc = test_one_epoch(model, test_loader, criterion, device,
                                             is_rec=args.is_rec, criterion_rec=criterion_rec, alpha=args.rec_alpha)
        if is_main():print(f"Test/Validation -> Loss: {test_loss:.5f}, Accuracy: {test_acc:.4f}")

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

    #print(f"💾 所有检查点已保存在目录: {args.model_save_dir}")
    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
