import numpy as np
import torch
import torch.nn as nn
import argparse
from util import load_data_n_model
import time
import os  # 引入 os 模块
import csv # 1. 引入 csv 模块

# ==================== 解决 num_workers 和 numpy 的冲突 ====================
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
# =========================================================================


# train_one_epoch 和 test_one_epoch 函数与上一个回答中的版本相同
# 这里为了完整性再次包含它们

def train_one_epoch(model, tensor_loader, criterion, device, optimizer):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    num_samples = 0

    for data in tensor_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        labels = labels.type(torch.LongTensor)

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.to(device)
        outputs = outputs.type(torch.FloatTensor)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)
        predict_y = torch.argmax(outputs, dim=1)
        epoch_accuracy += (predict_y == labels).sum().item()
        num_samples += labels.size(0)

    epoch_loss = epoch_loss / num_samples
    epoch_accuracy = epoch_accuracy / num_samples
    return epoch_loss, epoch_accuracy


def test_one_epoch(model, tensor_loader, criterion, device):
    model.eval()
    test_acc = 0.0
    test_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for data in tensor_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)

            outputs = model(inputs)
            outputs = outputs.type(torch.FloatTensor)
            outputs.to(device)
            loss = criterion(outputs, labels)

            predict_y = torch.argmax(outputs, dim=1)
            test_acc += (predict_y == labels).sum().item()
            test_loss += loss.item() * inputs.size(0)
            num_samples += labels.size(0)
        test_acc = test_acc / num_samples
        test_loss = test_loss / num_samples
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
        print(f"错误: 数据集根目录 '{root}' 未找到。")
        print("请确认您的脚本（run.py）是否在 'code/sense-fi/' 文件夹下，")
        print("并且 'datasets' 文件夹在 'code/' 的上一级目录。")
        return
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
    parser.add_argument('--dataset', choices = ['UT_HAR_data','NTU-Fi-HumanID','NTU-Fi_HAR','Widar'])
    parser.add_argument('--model', choices = ['MLP','LeNet','ResNet18','ResNet50','ResNet101','RNN','GRU','LSTM','BiLSTM', 'CNN+GRU','ViT'])
    # 新增的参数，用于自定义实验名称，并设为必填项
    #parser.add_argument('--exp_name', required=True, type=str, help='自定义实验名称，将用于创建模型保存目录。')
    parser.add_argument('--sample_rate', type=float, default=1.0, help='二次降采样的比例 (0.05到1.0)，对应25Hz到500Hz。默认为1.0，即不进行二次采样。')
    parser.add_argument('--interpolation', type=str,default='linear',choices=['linear', 'cubic', 'nearest', 'idw', 'rbf'],help='升采样时使用的插值方法。默认为 "linear"。')
    # 新增两个参数，用于接收完整的保存目录
    parser.add_argument('--model_save_dir', required=True, type=str, help='模型检查点的完整保存目录。')
    parser.add_argument('--metrics_save_dir', required=True, type=str, help='性能指标文件的完整保存目录。')
    args = parser.parse_args()

    train_loader, test_loader, model, train_epoch = load_data_n_model(args.dataset, args.model, root,args.sample_rate,args.interpolation)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    # --- 目录创建 ---
    # 现在 run.py 只负责确保目录存在，不再构建它
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.metrics_save_dir, exist_ok=True)
    print(f"✅ 模型将保存至: {os.path.abspath(args.model_save_dir)}")
    print(f"📊 性能指标将保存至: {os.path.abspath(args.metrics_save_dir)}")
    # ================================================================
    # 2. 计算保存间隔和保存点
    num_saves = 10
    if train_epoch < num_saves:
        # 如果总epoch数小于10，则每个epoch都保存
        save_interval = 1
    else:
        save_interval = train_epoch // num_saves

    # 创建一个包含所有需要保存的epoch编号的集合，方便快速查找
    save_epochs = set(range(save_interval, train_epoch + 1, save_interval))
    # 确保最后一个epoch总是被保存
    save_epochs.add(train_epoch)
    print(f"模型将会在以下Epoch结束时保存: {sorted(list(save_epochs))}")
    # ==========================================================

    # ==================== 4. 新增：初始化历史记录列表 ====================
    train_history = []
    test_history = []

    # --- 训练主循环 ---
    total_train_start = time.time()
    for epoch in range(1, train_epoch + 1):  # 循环从1开始，方便与epoch编号对应
        print(f"--- Epoch {epoch}/{train_epoch} ---")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, device, optimizer)
        print(f"Train -> Loss: {train_loss:.5f}, Accuracy: {train_acc:.4f}")

        test_loss, test_acc = test_one_epoch(model, test_loader, criterion, device)
        print(f"Test/Validation -> Loss: {test_loss:.5f}, Accuracy: {test_acc:.4f}")

        # ==================== 5. 新增：收集当前epoch的数据 ====================
        train_history.append({'epoch': epoch, 'loss': train_loss, 'accuracy': train_acc})
        test_history.append({'epoch': epoch, 'loss': test_loss, 'accuracy': test_acc})

        # --- 检查是否到达保存点 ---
        '''if epoch in save_epochs:
            model_save_path = os.path.join(args.model_save_dir, f'model_epoch_{epoch}.pth')
            print(f"💾 到达保存点，正在保存模型到: {model_save_path}")
            torch.save(model.state_dict(), model_save_path)'''

    total_train_end = time.time()
    print("\n--- 训练完成 ---")
    print(f"⏱️ 总训练耗时：{total_train_end - total_train_start:.2f} 秒")

    # 使用新的目录参数来构建路径
    train_metrics_path = os.path.join(args.metrics_save_dir, 'train_metrics.csv')
    test_metrics_path = os.path.join(args.metrics_save_dir, 'test_metrics.csv')

    print(f"📊 正在保存训练历史到: {train_metrics_path}")
    save_metrics_to_csv(train_metrics_path, train_history)

    print(f"📊 正在保存测试历史到: {test_metrics_path}")
    save_metrics_to_csv(test_metrics_path, test_history)

    #print(f"💾 所有检查点已保存在目录: {args.model_save_dir}")

if __name__ == "__main__":
    main()