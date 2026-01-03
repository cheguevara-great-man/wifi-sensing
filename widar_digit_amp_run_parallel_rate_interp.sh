#!/usr/bin/env bash
## nohup ./widar_digit_amp_run_parallel_rate_interp.sh > rate_interp_run.log 2>&1 &
#####################################################################
# run_grid_search.sh
# 【修复版】严格限制：一张 GPU 必须跑完一个任务，才能接下一个
#####################################################################

# --- 用户配置区 ---

PYTHON_SCRIPT="run.py"
DATASET_NAME="Widar_digit_amp"
# 确保这里列出了你所有的 GPU ID
GPU_LIST=(2 3 4 5 6 7)

#BASE_EXP_NAME="amp_rate_interp_$(date +%Y%m%d_%H%M)"
BASE_EXP_NAME="amp_rate_interp_20260102_2336"

use_energy_input=0
use_mask_0=0

# 1. 采样方法 (3种)
SAMPLE_METHODS=(equidistant)

# 2. 采样率 (6种)
SAMPLE_RATES=(0.05 0.1 0.2 0.5 1)

# 3. 插值方法 (4种)
INTERPOLATION_METHODS=(cubic nearest)

# 4. 模型 (5种) - 显存预估仅作参考，本脚本强制分配
declare -A MODEL_MEM_REQUIREMENTS
MODEL_MEM_REQUIREMENTS=(
    #['MLP']=2860
    ['LeNet']=3669
    ['ResNet18']=15704
    ['RNN']=2277
    #['BiLSTM']=3131
)
MODELS=("${!MODEL_MEM_REQUIREMENTS[@]}")

# --- 核心逻辑：生成任务列表 ---
declare -a PENDING_TASKS

for s_method in "${SAMPLE_METHODS[@]}"; do
    for rate in "${SAMPLE_RATES[@]}"; do

        # 【关键修复】当采样率为 1 时，强制只用 linear，跳过其他插值
        # 这会把任务数从 360 减少到 315 左右，避免无效计算
        if [[ "$rate" == "1" ]] || [[ "$rate" == "1.0" ]]; then
            CURRENT_INTERP_LIST=("linear")
        else
            CURRENT_INTERP_LIST=("${INTERPOLATION_METHODS[@]}")
        fi

        for i_method in "${CURRENT_INTERP_LIST[@]}"; do
            for model in "${MODELS[@]}"; do
                PENDING_TASKS+=("${s_method}:${rate}:${i_method}:${model}")
            done
        done
    done
done

TOTAL_TASKS=${#PENDING_TASKS[@]}
echo "✅ 任务列表生成完毕，总共需要运行 ${TOTAL_TASKS} 个实验。"

# --- 核心逻辑：GPU 调度 ---
declare -A GPU_BUSY_PID # 记录每张卡跑的 PID

echo "=============================================================="
echo "启动时间：$(date)"
echo "策略：严格独占模式 (One GPU = One Task)"
echo "=============================================================="

DATASET_ROOT_DIR="../datasets/sense-fi"
declare -a RUNNING_PIDS

while ((${#PENDING_TASKS[@]} > 0)); do
    launched_in_this_pass=false

    # 遍历每一个待处理任务
    for i in "${!PENDING_TASKS[@]}"; do
        task_string=${PENDING_TASKS[i]}

        # 寻找一个空闲的 GPU
        chosen_gpu=-1
        for gpu_id in "${GPU_LIST[@]}"; do
            pid="${GPU_BUSY_PID[$gpu_id]}"
            # 如果 PID 为空，或者 PID 对应的进程已经不在了，说明这张卡空闲
            if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
                chosen_gpu=$gpu_id
                break # 找到一张空闲卡，立刻停止寻找
            fi
        done

        # 如果找到了空闲 GPU，就启动任务
        if [[ "$chosen_gpu" -ne -1 ]]; then
            IFS=':' read -r sample_method sample_rate interpolation_method model_name <<< "$task_string"

            # 构建目录
            exp_sub_dir="method_${sample_method}/rate_${sample_rate}/interp_${interpolation_method}"
            model_dir="${DATASET_ROOT_DIR}/${DATASET_NAME}/EXP/${BASE_EXP_NAME}/Model Parameters/${exp_sub_dir}/${model_name}"
            metrics_dir="${DATASET_ROOT_DIR}/${DATASET_NAME}/EXP/${BASE_EXP_NAME}/Metrics/${exp_sub_dir}/${model_name}"
            log_dir="${DATASET_ROOT_DIR}/${DATASET_NAME}/EXP/${BASE_EXP_NAME}/Logs/${exp_sub_dir}/${model_name}"
            mkdir -p "$model_dir" "$metrics_dir" "$log_dir"
            log_file="${log_dir}/training.log"

            echo "[$(date '+%H:%M:%S')] 🚀 启动任务: ${task_string} on GPU ${chosen_gpu}"

            # 限制线程数并启动
            OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
            CUDA_VISIBLE_DEVICES=$chosen_gpu \
            python -u "$PYTHON_SCRIPT" \
                --dataset "$DATASET_NAME" \
                --model "$model_name" \
                --sample_rate "$sample_rate" \
                --sample_method "$sample_method" \
                --interpolation "$interpolation_method" \
                --use_energy_input "$use_energy_input" \
                --use_mask_0 "$use_mask_0" \
                --model_save_dir "$model_dir" \
                --metrics_save_dir "$metrics_dir" \
                > "$log_file" 2>&1 &

            pid=$!
            # 【锁定】标记这张卡正在忙
            GPU_BUSY_PID[$chosen_gpu]=$pid
            RUNNING_PIDS+=("$pid")

            # 从待处理列表中移除该任务
            unset 'PENDING_TASKS[i]'
            launched_in_this_pass=true

            # 休息 1 秒，避免瞬间并发冲击 CPU
            sleep 1
        fi
    done

    # 重新整理数组索引
    PENDING_TASKS=("${PENDING_TASKS[@]}")

    # 如果這一轮没启动任何任务，说明所有卡都忙，休息一会儿
    if ! $launched_in_this_pass; then
        echo "[$(date '+%H:%M:%S')] 💤 所有 8 张 GPU 都在忙，等待任务完成... (剩余任务: ${#PENDING_TASKS[@]})"
        sleep 30
    fi
done

echo "🎉 所有任务已提交，等待最后几个任务完成..."
wait
echo "✅ 全部完成！"