#!/usr/bin/env bash
## nohup ./run_parallel_rate_interp.sh > rate_interp_run.log 2>&1 &
#####################################################################
# run_grid_search.sh
# 终极版并行脚本，用于大规模网格搜索实验。
# 循环遍历采样率、插值方法和模型，并使用动态GPU调度策略。
#####################################################################

# --- 用户配置区 ---

# 1. Python脚本和数据集
PYTHON_SCRIPT="run.py"
DATASET_NAME="NTU-Fi-HumanID"
# 2. 可用GPU
GPU_LIST=(0 1 2 3 4 5 6)
# 3. 基础实验名称 (将作为所有子实验目录的前缀)
BASE_EXP_NAME="energy_rate_interp_$(date +%Y%m%d_%H%M)"
use_energy_input=1      # 1: 使用能量信息 (True)
use_mask_0=0            # 0: 不使用 mask_0 (False)
# 4. 【第一层循环】采样率 (对应 --sample_rate)
#SAMPLE_RATES=(0.01 0.02 0.03 0.04 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
#SAMPLE_RATES=(0.004 0.006 0.008 0.01 0.02 0.03 0.04 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
SAMPLE_RATES=(1)

# 5. 【第二层循环】插值方法 (对应 --interpolation)
#INTERPOLATION_METHODS=(linear cubic nearest idw rbf)
INTERPOLATION_METHODS=(linear cubic nearest )
# 6. 【第三层循环】模型及其显存需求 (单位: MiB)
declare -A MODEL_MEM_REQUIREMENTS
MODEL_MEM_REQUIREMENTS=(
    ['MLP']=6872
    ['LeNet']=2574
    ['ResNet18']=2941
    ['ResNet50']=3271
    ['ResNet101']=3723
    ['RNN']=2314
    ['GRU']=2478
    ['LSTM']=2462
    ['BiLSTM']=2844
    ['CNN+GRU']=3663
    ['ViT']=6765
)
MODELS=("${!MODEL_MEM_REQUIREMENTS[@]}")


# --- 脚本核心逻辑 ---

# 函数：找到能满足显存需求且最空闲的GPU
find_suitable_gpu() {
    local model_name="$1"
    local required_mem=${MODEL_MEM_REQUIREMENTS[$model_name]}
    local best_gpu=-1
    local max_free_mem=-1
    for gpu_id in "${GPU_LIST[@]}"; do
        local mem_free
        mem_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null)
        if ! [[ "$mem_free" =~ ^[0-9]+$ ]]; then continue; fi
        if (( mem_free >= required_mem )) && (( mem_free > max_free_mem )); then
            max_free_mem=$mem_free
            best_gpu=$gpu_id
        fi
    done
    echo "$best_gpu"
}

# --- 任务生成 ---
declare -a PENDING_TASKS
for rate in "${SAMPLE_RATES[@]}"; do
    for method in "${INTERPOLATION_METHODS[@]}"; do
        for model in "${MODELS[@]}"; do
            # 将任务参数组合成一个字符串，用冒号分隔
            PENDING_TASKS+=("${rate}:${method}:${model}")
        done
    done
done
TOTAL_TASKS=${#PENDING_TASKS[@]}
echo "✅ 任务列表生成完毕，总共需要运行 ${TOTAL_TASKS} 个实验。"

# --- 主程序 ---
declare -a RUNNING_PIDS
DATASET_ROOT_DIR="../datasets/sense-fi"

echo "=============================================================="
echo "大规模网格搜索并行脚本启动：$(date)"
echo "  总任务数: ${TOTAL_TASKS}"
echo "=============================================================="

while ((${#PENDING_TASKS[@]} > 0)); do
    launched_in_this_pass=false
    for i in "${!PENDING_TASKS[@]}"; do
        task_string=${PENDING_TASKS[i]}

        # 解析任务字符串
        IFS=':' read -r sample_rate interpolation_method model_name <<< "$task_string"

        chosen_gpu=$(find_suitable_gpu "$model_name")

        if [[ "$chosen_gpu" -ne -1 ]]; then
            # --- 动态构建实验名、日志和模型保存路径 ---
            exp_sub_dir="rate_${sample_rate}/interp_${interpolation_method}"
             # 2. 构建模型、指标和日志的最终保存目录
            model_dir="${DATASET_ROOT_DIR}/${DATASET_NAME}/EXP/${BASE_EXP_NAME}/Model Parameters/${exp_sub_dir}/${model_name}"
            metrics_dir="${DATASET_ROOT_DIR}/${DATASET_NAME}/EXP/${BASE_EXP_NAME}/Metrics/${exp_sub_dir}/${model_name}"
            log_dir="${DATASET_ROOT_DIR}/${DATASET_NAME}/EXP/${BASE_EXP_NAME}/Logs/${exp_sub_dir}/${model_name}"
            # 3. 确保所有目录都存在
            mkdir -p "$model_dir" "$metrics_dir" "$log_dir"
            log_file="${log_dir}/training.log"
            echo "[`date '+%H:%M:%S'`] [${#RUNNING_PIDS[@]} running, ${#PENDING_TASKS[@]} left] 分配任务: ${task_string} -> GPU ${chosen_gpu}"

            # --- 启动Python子进程 ---
            CUDA_VISIBLE_DEVICES=$chosen_gpu \
            python -u "$PYTHON_SCRIPT" \
                --dataset "$DATASET_NAME" \
                --model "$model_name" \
                --sample_rate "$sample_rate" \
                --interpolation "$interpolation_method" \
                --use_energy_input "$use_energy_input" \
                --use_mask_0 "$use_mask_0" \
                --model_save_dir "$model_dir" \
                --metrics_save_dir "$metrics_dir" \
                > "$log_file" 2>&1 &

            pid=$!
            RUNNING_PIDS+=("$pid")
            unset 'PENDING_TASKS[i]'
            echo "[`date '+%H:%M:%S'`] ✅ 任务已启动，PID: $pid, 日志: $log_file"

            launched_in_this_pass=true
            sleep 10
        fi
    done

    PENDING_TASKS=("${PENDING_TASKS[@]}")

    if ! $launched_in_this_pass && ((${#RUNNING_PIDS[@]} > 0)); then
        echo "[`date '+%H:%M:%S'`] [${#RUNNING_PIDS[@]} running, ${#PENDING_TASKS[@]} left] 所有GPU均无足够空间容纳剩余任务，等待任一任务结束..."
        wait -n
        running_pids=()
        for pid in "${RUNNING_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then running_pids+=("$pid"); fi
        done
        RUNNING_PIDS=("${running_pids[@]}")
    fi
done

echo "=============================================================="
echo "所有任务已提交，等待所有后台任务完成..."
wait
echo "=============================================================="
echo "🎉 全部网格搜索任务完成！完成时间: $(date)"
echo "=============================================================="
exit 0