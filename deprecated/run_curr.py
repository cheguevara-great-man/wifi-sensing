#可能是中间状态的run.py，
import numpy as np
import torch
import torch.nn as nn
import argparse
from util import load_data_n_model
import time
import os  # 引入 os 模块


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


def main():
    root = '../datasets/'
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
    parser.add_argument('--dataset', choices=['UT_HAR_data', 'NTU-Fi-HumanID', 'NTU-Fi_HAR', 'Widar'])
    parser.add_argument('--model',
                        choices=['MLP', 'LeNet', 'ResNet18', 'ResNet50', 'ResNet101', 'RNN', 'GRU', 'LSTM', 'BiLSTM',
                                 'CNN+GRU', 'ViT'])
    args = parser.parse_args()

    train_loader, test_loader, model, train_epoch = load_data_n_model(args.dataset, args.model, root)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ==================== 新增：等间隔保存逻辑 ====================
    # 1. 创建保存目录
    save_dir_base = 'saved_models_checkpoints'
    # 为每次实验创建一个独立的子文件夹
    experiment_name = f'{args.dataset}_{args.model}'
    save_dir = os.path.join(save_dir_base, experiment_name)
    os.makedirs(save_dir, exist_ok=True)

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

    # --- 训练主循环 ---
    total_train_start = time.time()
    for epoch in range(1, train_epoch + 1):  # 循环从1开始，方便与epoch编号对应
        print(f"--- Epoch {epoch}/{train_epoch} ---")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, device, optimizer)
        print(f"Train -> Loss: {train_loss:.5f}, Accuracy: {train_acc:.4f}")

        test_loss, test_acc = test_one_epoch(model, test_loader, criterion, device)
        print(f"Test/Validation -> Loss: {test_loss:.5f}, Accuracy: {test_acc:.4f}")

        # --- 检查是否到达保存点 ---
        if epoch in save_epochs:
            model_save_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')
            print(f"💾 到达保存点，正在保存模型到: {model_save_path}")
            torch.save(model.state_dict(), model_save_path)

    total_train_end = time.time()
    print("\n--- 训练完成 ---")
    print(f"⏱️ 总训练耗时：{total_train_end - total_train_start:.2f} 秒")
    print(f"💾 所有检查点已保存在目录: {save_dir}")


if __name__ == "__main__":
    main()