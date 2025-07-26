#!/usr/bin/env bash

#####################################################################
# profile_memory_parallel.sh
# 在多张GPU上并行剖析所有模型的峰值显存占用，以“波次”方式执行，
# 兼顾了速度和测量准确性。
#####################################################################

# --- 用户配置区 ---

# 1. Python脚本和数据集
PYTHON_SCRIPT="run.py"
DATASET_NAME="NTU-Fi_HAR"
# 2. 【固定】要测试的采样率和插值方法
SAMPLE_RATE=0.5
INTERPOLATION_METHOD="cubic"
# 3. 【固定】用于测试的GPU ID列表
GPU_LIST=(0 1 2 3)
# 4. 自定义实验名称
EXP_NAME="memory_profiling_parallel_$(date +%Y%m%d_%H%M)"
# 5. 要剖析的模型列表
'''MODELS=(
    'MLP' 'LeNet' 'ResNet18' 'ResNet50' 'ResNet101' 'RNN' 'GRU'
    'LSTM' 'BiLSTM' 'CNN+GRU' 'ViT'
)'''
MODELS=(
 'BiLSTM' 'CNN+GRU' 'ViT'
)
# --- 脚本核心逻辑 ---

declare -A PEAK_MEMORY_USAGE # 存储最终结果
declare -A CURRENT_WAVE_PIDS # 存储当前波次的 {PID: model_name}
declare -A CURRENT_WAVE_PEAKS # 存储当前波次的 {PID: peak_mem}

# 函数：清理已结束的进程 (用于主监控循环)
cleanup_pids() {
    local running_pids=()
    for pid in "${!CURRENT_WAVE_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            running_pids+=("$pid")
        fi
    done
    # 更新正在运行的PID列表
    local updated_pids=()
    for pid in "${running_pids[@]}"; do updated_pids+=("$pid"); done
    # Bash 4.3+ supports `unset CURRENT_WAVE_PIDS; declare -A CURRENT_WAVE_PIDS=(...)`
    # This is a more portable way:
    local temp_pids_map_str=""
    for pid in "${running_pids[@]}"; do
        model=${CURRENT_WAVE_PIDS[$pid]}
        temp_pids_map_str+="[$pid]=\"$model\" "
    done
    eval "CURRENT_WAVE_PIDS=($temp_pids_map_str)"
}

# --- 主程序 ---
DATASET_ROOT_DIR="../datasets/sense-fi"
TOTAL_MODELS=${#MODELS[@]}
NUM_GPUS=${#GPU_LIST[@]}

echo "=============================================================="
echo "并行模型显存剖析脚本启动：$(date)"
echo "  配置: sample_rate=${SAMPLE_RATE}, interpolation=${INTERPOLATION_METHOD}"
echo "  将在 ${NUM_GPUS} 张 GPU 上分波次运行 ${TOTAL_MODELS} 个模型。"
echo "=============================================================="

# 按GPU数量分波次处理模型
for (( i=0; i<TOTAL_MODELS; i+=NUM_GPUS )); do
    # 获取当前波次要运行的模型
    wave_models=("${MODELS[@]:i:NUM_GPUS}")
    wave_num=$((i / NUM_GPUS + 1))
    echo -e "\n--- 开始执行第 ${wave_num} 波任务 ---"
    echo "  - 本波次模型: ${wave_models[*]}"

    # 清空上一波次的数据
    unset CURRENT_WAVE_PIDS; declare -A CURRENT_WAVE_PIDS
    unset CURRENT_WAVE_PEAKS; declare -A CURRENT_WAVE_PEAKS

    # --- 启动当前波次的所有任务 ---
    for j in "${!wave_models[@]}"; do
        model_name=${wave_models[j]}
        gpu_id=${GPU_LIST[j]}

        temp_dir_base="${DATASET_ROOT_DIR}/${DATASET_NAME}/temp_profiling/${EXP_NAME}"
        model_dir="${temp_dir_base}/${model_name}/Model"
        metrics_dir="${temp_dir_base}/${model_name}/Metrics"
        mkdir -p "$model_dir" "$metrics_dir"

        echo "  - 启动模型: ${model_name} on GPU ${gpu_id}"
        CUDA_VISIBLE_DEVICES=$gpu_id \
        python -u "$PYTHON_SCRIPT" \
            --dataset "$DATASET_NAME" --model "$model_name" \
            --model_save_dir "$model_dir" --metrics_save_dir "$metrics_dir" \
            --sample_rate "$SAMPLE_RATE" --interpolation "$INTERPOLATION_METHOD" \
            &

        pid=$!
        CURRENT_WAVE_PIDS[$pid]=$model_name
        CURRENT_WAVE_PEAKS[$pid]=0
    done

    echo "  - 本波次所有任务已启动，进入监控模式..."
    sleep 5 # 等待所有进程初始化

    # --- 并行监控当前波次的所有任务 ---
    while ((${#CURRENT_WAVE_PIDS[@]} > 0)); do
        for pid in "${!CURRENT_WAVE_PIDS[@]}"; do
            model_name=${CURRENT_WAVE_PIDS[$pid]}
            current_mem=$(nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits | grep "^${pid}," | awk '{print $2}')

            if [[ -n "$current_mem" ]] && (( current_mem > CURRENT_WAVE_PEAKS[$pid] )); then
                CURRENT_WAVE_PEAKS[$pid]=$current_mem
            fi
        done
        # 实时打印当前所有任务的峰值
        printf "\r  - [监控中] "
        for pid in "${!CURRENT_WAVE_PIDS[@]}"; do
            model_name=${CURRENT_WAVE_PIDS[$pid]}
            peak=${CURRENT_WAVE_PEAKS[$pid]}
            printf "%s: %'d MiB | " "$model_name" "$peak"
        done

        cleanup_pids # 清理已结束的进程
        sleep 1
    done
    printf "\r\033[K" # 清除最后的监控行
    echo "  - 第 ${wave_num} 波任务全部结束。"

    # --- 记录本波次的最终结果 ---
    for pid in "${!CURRENT_WAVE_PEAKS[@]}"; do
        model_name=$(echo "${CURRENT_WAVE_PIDS[$pid]}" | tr -d '[:space:]') # This is a fallback
        # A safer way to get model name after pid is gone
        for p in "${!CURRENT_WAVE_PIDS[@]}"; do
             if [[ "$p" -eq "$pid" ]]; then model_name=${CURRENT_WAVE_PIDS[$p]}; fi
        done
        peak_mem=${CURRENT_WAVE_PEAKS[$pid]}
        PEAK_MEMORY_USAGE[$model_name]=$peak_mem
        echo "  - [结果] 模型 ${model_name} 的峰值显存占用: ${peak_mem} MiB"
    done
done


# --- 打印最终总结报告 ---
echo -e "\n\n=============================================================="
echo "剖析完成！所有模型的峰值显存占用如下："
echo "=============================================================="
echo "您可以将此配置直接复制到您的并行脚本中。"
echo "declare -A MODEL_MEM_REQUIREMENTS"
echo "MODEL_MEM_REQUIREMENTS=("

for model_name in "${MODELS[@]}"; do
    peak_mem=${PEAK_MEMORY_USAGE[$model_name]:-0} # Default to 0 if not found
    buffered_mem=$((peak_mem * 115 / 100))
    echo "    ['${model_name}']=${buffered_mem}  # 测量峰值: ${peak_mem} MiB"
done
echo ")"
echo "=============================================================="

# 清理临时目录
echo "正在清理临时文件..."
rm -rf "${DATASET_ROOT_DIR}/${DATASET_NAME}/temp_profiling"
echo "清理完毕。"
exit 0