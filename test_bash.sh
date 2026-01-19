#!/bin/bash
set -u

# ================== 1. 模拟配置 ==================
GPU_LIST=(1 2 3)    # 你真实的 GPU 列表
GPUS_PER_TASK=3       # 你的分组设置
CHECK_INTERVAL=2      # [测试用] 设为 2秒，方便你快速看到计数器累加
THRESHOLD=3           # 阈值 (3次 * 2秒 = 6秒后启动)

# 初始化变量
declare -A GPU_IDLE_COUNT=()
declare -A GROUP_PID=() # 这里为空，模拟没有任务在跑

# 生成分组 (和你代码一样)

declare -a GPU_GROUPS=()
len=${#GPU_LIST[@]}
for ((i=0; i+GPUS_PER_TASK<=len; i+=GPUS_PER_TASK)); do
  group=("${GPU_LIST[@]:i:GPUS_PER_TASK}")
  GPU_GROUPS+=("${group[*]}")
done

# 模拟一个待办任务
PENDING_TASKS=("Test_Task_A")

echo "=== 开始真实环境测试 ==="
echo "目标 GPU 组: ${GPU_GROUPS[*]}"
echo "检测间隔: ${CHECK_INTERVAL}秒 | 阈值: ${THRESHOLD}次"
echo "----------------------------------------"

# ================== 2. 核心逻辑 (完全复刻你的修改) ==================
while ((${#PENDING_TASKS[@]} > 0)); do

  # 遍历 GPU 组
  for group_str in "${GPU_GROUPS[@]}"; do

    # --- [你的核心修改] ---
    # 1. 查物理利用率 (真实调用 nvidia-smi)
    gpus="${group_str// /,}"
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpus" | sort -nr | head -n1)

    # 2. 计数逻辑
    if [[ "$util" -lt 5 ]]; then
        GPU_IDLE_COUNT[$group_str]=$(( ${GPU_IDLE_COUNT[$group_str]:-0} + 1 ))
        status="✅ 空闲"
    else
        GPU_IDLE_COUNT[$group_str]=0
        status="🚧 忙碌 ($util%)"
    fi

    # 打印实时状态给看
    curr_count=${GPU_IDLE_COUNT[$group_str]}
    echo "[$(date +%H:%M:%S)] Group [$group_str] | $status | 计数器: $curr_count / $THRESHOLD"

    # 3. 阈值判断
    if [[ "$curr_count" -lt "$THRESHOLD" ]]; then
        continue  # 跳过，不派活
    fi
    # --- [修改结束] ---

    # --- 模拟启动任务 ---
    echo "----------------------------------------"
    echo "🎉 条件满足！逻辑通过！"
    echo "   可以在真实脚本中: 🚀 启动任务 on GPUs [$group_str]"
    echo "----------------------------------------"

    # 移除任务，结束测试
    unset 'PENDING_TASKS[0]'
    break 2
  done

  # 等待 (测试时用 2秒，真实脚本里你是 30秒)
  sleep $CHECK_INTERVAL
done