#!/usr/bin/env bash
#没有降采样也没有插值，只是基于原始csi对所有模型训练。
#####################################################################
#运行
#先确认EXP_NAME是否正确，GPU是否空闲，代码是否正确
# nohup ./run_parallel_ultimate.sh > ultimate_run.log 2>&1 &
# run_parallel_ultimate.sh
# 在多张GPU上并行运行多个异构模型训练任务。
# 采用“资源需求感知”的动态调度策略，无固定并发上限，
# 会尽可能地将GPU填满，直到没有GPU能容纳任何待处理任务为止。
#####################################################################

# --- 用户配置区 ---

# 1. 要运行的Python脚本
PYTHON_SCRIPT="run.py"
# 2. 固定的数据集名称
DATASET_NAME="NTU-Fi_HAR"
# 3. 可用的GPU设备ID列表
GPU_LIST=(0 1 2 3)
# 4. 自定义实验名称
EXP_NAME="energy_500hz_baseline_$(date +%Y%m%d_%H%M)"

# 5. 【重要】模型显存需求表 (单位: MiB) - 已根据您提供的数据更新
#    安全起见，在您的测试值基础上增加了约10%的缓冲。
declare -A MODEL_MEM_REQUIREMENTS
MODEL_MEM_REQUIREMENTS=(
    ['MLP']=5800
    ['LeNet']=2250
    ['ResNet18']=2550
    ['ResNet50']=2850
    ['ResNet101']=3200
    ['RNN']=2050
    ['GRU']=2100
    ['LSTM']=2150
    ['BiLSTM']=2450
    ['CNN+GRU']=3150
    ['ViT']=5800
)

# --- 脚本核心逻辑 ---

# ... (此处省略 检查bash版本/nvidia-smi/脚本文件 的代码) ...

# 待处理的模型列表
declare -a PENDING_MODELS=("${!MODEL_MEM_REQUIREMENTS[@]}")
# 存放活跃子进程PID的数组
declare -a RUNNING_PIDS

# 函数：找到能满足显存需求且最空闲的GPU
# 参数1: model_name
find_suitable_gpu() {
    local model_name="$1"
    local required_mem=${MODEL_MEM_REQUIREMENTS[$model_name]}
    local best_gpu=-1
    local max_free_mem=-1
    for gpu_id in "${GPU_LIST[@]}"; do
        local mem_free
        mem_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id")
        if ! [[ "$mem_free" =~ ^[0-9]+$ ]]; then continue; fi

        if (( mem_free >= required_mem )); then
            if (( mem_free > max_free_mem )); then
                max_free_mem=$mem_free
                best_gpu=$gpu_id
            fi
        fi
    done
    echo "$best_gpu"
}

# --- 主程序 ---
echo "=============================================================="
echo "极限并行训练脚本启动 (带独立日志)：$(date)"
echo "  数据集: $DATASET_NAME"
echo "  待处理模型数量: ${#PENDING_MODELS[@]}"
echo "  可用GPU: ${GPU_LIST[*]}"
echo "=============================================================="




# ==================== 新增：为所有日志创建一个总目录 ====================
DATASET_ROOT_DIR="../datasets/sense-fi"
LOG_BASE_DIR="${DATASET_ROOT_DIR}/${DATASET_NAME}/Logs/${EXP_NAME}"
mkdir -p "$LOG_BASE_DIR"
echo "所有子任务的日志将保存在: $(realpath ${LOG_BASE_DIR})" # 使用 realpath 显示绝对路径
# ======================================================================




# 当还有模型待处理或有任务在运行时，循环继续
while ((${#PENDING_MODELS[@]} > 0)); do

    launched_in_this_pass=false

    # 遍历所有待处理的模型
    # 使用索引遍历，方便从数组中删除已启动的任务
    for i in "${!PENDING_MODELS[@]}"; do
        model_name=${PENDING_MODELS[i]}

        # 寻找一个合适的GPU
        chosen_gpu=$(find_suitable_gpu "$model_name")

        # 如果找到了合适的GPU
        if [[ "$chosen_gpu" -ne -1 ]]; then
            echo "[`date '+%H:%M:%S'`] 分配任务: Model=$model_name -> GPU $chosen_gpu"

            # ==================== 核心修改：为子进程创建并重定向日志 ====================
            # 1. 定义这个任务专属的日志文件路径
            log_file="${LOG_BASE_DIR}/${model_name}.log"
            # 2. 修改启动命令，加入输出重定向
            CUDA_VISIBLE_DEVICES=$chosen_gpu \
            python -u "$PYTHON_SCRIPT" \
                --dataset "$DATASET_NAME" \
                --model "$model_name" \
                --exp_name "$EXP_NAME" \
                > "$log_file" 2>&1 &
            # =========================================================================

            # 记录PID并从待处理列表中移除该模型
            pid=$!
            RUNNING_PIDS+=("$pid")
            unset 'PENDING_MODELS[i]'

            echo "[`date '+%H:%M:%S'`] ✅ 任务已启动，PID: $pid, 日志: $log_file"

            launched_in_this_pass=true
            sleep 2 # 给GPU一点反应时间
        fi
    done

    # 重新索引数组，去除空位
    PENDING_MODELS=("${PENDING_MODELS[@]}")

    # 如果在这一轮扫描中没有启动任何任务，说明所有GPU都满了
    # 此时需要等待任一正在运行的任务结束，以释放资源
    if ! $launched_in_this_pass && ((${#RUNNING_PIDS[@]} > 0)); then
        echo "[`date '+%H:%M:%S'`] 所有GPU均无足够空间容纳剩余任务，等待任一任务结束..."
        # 等待任一子进程结束
        wait -n
        # 清理已结束的进程ID
        running_pids=()
        for pid in "${RUNNING_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then running_pids+=("$pid"); fi
        done
        RUNNING_PIDS=("${running_pids[@]}")
    fi
done

# --- 等待所有任务结束 ---
echo "=============================================================="
echo "所有任务已提交，等待所有后台任务完成..."
wait
echo "=============================================================="
echo "🎉 全部训练任务完成！完成时间: $(date)"
echo "=============================================================="
exit 0