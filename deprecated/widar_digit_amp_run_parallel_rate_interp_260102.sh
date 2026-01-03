#!/usr/bin/env bash
## nohup ./widar_digit_amp_run_parallel_rate_interp.sh > rate_interp_run.log 2>&1 &
#####################################################################
# run_grid_search.sh
# 终极版并行脚本：已加入 GPU 独占锁机制，解决 CPU 过载问题。
#####################################################################

# --- 用户配置区 ---

# 1. Python脚本和数据集
PYTHON_SCRIPT="run.py"
DATASET_NAME="Widar_digit_amp"
# 2. 可用GPU (建议全写上，脚本会自动调度)
GPU_LIST=(0 1 2 3 4 5 6 7 )
# 3. 基础实验名称
BASE_EXP_NAME="amp_rate_interp_$(date +%Y%m%d_%H%M)"
use_energy_input=0      # 1: 使用能量信息 (True)
use_mask_0=0            # 0: 不使用 mask_0 (False)

# 4. 采样方法
SAMPLE_METHODS=(equidistant gaussian poisson)

# 4. 采样率
SAMPLE_RATES=(0.05 0.1 0.2 0.25 0.5 1)

# 5. 插值方法 (当 rate < 1 时遍历这些)
INTERPOLATION_METHODS=(linear cubic nearest akima)

# 6. 模型及其显存需求 (已乘 1.2 冗余)
declare -A MODEL_MEM_REQUIREMENTS
MODEL_MEM_REQUIREMENTS=(
    ['MLP']=2860
    ['LeNet']=3669
    ['ResNet18']=15704
    #['ResNet50']=20142
    #['ResNet101']=21047
    ['RNN']=2277
    #['GRU']=2558
    #['LSTM']=2675
    ['BiLSTM']=3131
)
MODELS=("${!MODEL_MEM_REQUIREMENTS[@]}")


# --- 脚本核心逻辑 (修改部分) ---

# 【新增】关联数组，用于记录每张 GPU 上正在运行的 PID
declare -A GPU_BUSY_PID

# 【新增】检查 GPU 是否忙碌（即上面的 PID 是否还在活著）
is_gpu_busy() {
    local gpu_id="$1"
    local pid="${GPU_BUSY_PID[$gpu_id]}"

    # 如果没有记录 PID，说明空闲
    if [[ -z "$pid" ]]; then
        return 1 # false, not busy
    fi

    # 检查 PID 是否存在
    if kill -0 "$pid" 2>/dev/null; then
        return 0 # true, busy (进程还活着)
    else
        # 进程已死，清理记录，返回空闲
        unset GPU_BUSY_PID["$gpu_id"]
        return 1 # false, not busy
    fi
}

# 函数：找到能满足显存需求 且 当前未运行任务 的GPU
find_suitable_gpu() {
    local model_name="$1"
    local required_mem=${MODEL_MEM_REQUIREMENTS[$model_name]}
    local best_gpu=-1
    local max_free_mem=-1

    for gpu_id in "${GPU_LIST[@]}"; do
        # 1. 先检查这张卡是不是已经被脚本分配了任务且任务还没跑完
        if is_gpu_busy "$gpu_id"; then
            continue # 跳过忙碌的 GPU
        fi

        # 2. 只有空闲的卡才去查显存
        local mem_free
        mem_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null)

        if ! [[ "$mem_free" =~ ^[0-9]+$ ]]; then continue; fi

        # 3. 显存足够
        if (( mem_free >= required_mem )); then
            # 这里简单起见，只要显存够且空闲，直接返回第一个找到的即可
            # 不需要非得找 max_free，因为我们已经限制了一卡一任务
            echo "$gpu_id"
            return
        fi
    done
    echo "-1"
}


# --- 任务生成 ---
declare -a PENDING_TASKS

for s_method in "${SAMPLE_METHODS[@]}"; do
    for rate in "${SAMPLE_RATES[@]}"; do

        # 判断：如果 rate 是 1，只跑一次 linear
        if [[ "$rate" == "1" ]] || [[ "$rate" == "1.0" ]]; then
            CURRENT_INTERP_LIST=("linear")
        else
            CURRENT_INTERP_LIST=("${INTERPOLATION_METHODS[@]}")
        fi

        # 【Bug修复】这里必须遍历 CURRENT_INTERP_LIST，而不是 INTERPOLATION_METHODS
        for i_method in "${CURRENT_INTERP_LIST[@]}"; do
            for model in "${MODELS[@]}"; do
                PENDING_TASKS+=("${s_method}:${rate}:${i_method}:${model}")
            done
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
echo "  并行策略: 严格限制每张 GPU 只运行 1 个任务"
echo "=============================================================="

while ((${#PENDING_TASKS[@]} > 0)); do
    launched_in_this_pass=false
    for i in "${!PENDING_TASKS[@]}"; do
        task_string=${PENDING_TASKS[i]}

        # 解析任务字符串
        IFS=':' read -r sample_method sample_rate interpolation_method model_name <<< "$task_string"

        chosen_gpu=$(find_suitable_gpu "$model_name")

        if [[ "$chosen_gpu" -ne -1 ]]; then
            # --- 动态构建实验名 ---
            exp_sub_dir="method_${sample_method}/rate_${sample_rate}/interp_${interpolation_method}"
            model_dir="${DATASET_ROOT_DIR}/${DATASET_NAME}/EXP/${BASE_EXP_NAME}/Model Parameters/${exp_sub_dir}/${model_name}"
            metrics_dir="${DATASET_ROOT_DIR}/${DATASET_NAME}/EXP/${BASE_EXP_NAME}/Metrics/${exp_sub_dir}/${model_name}"
            log_dir="${DATASET_ROOT_DIR}/${DATASET_NAME}/EXP/${BASE_EXP_NAME}/Logs/${exp_sub_dir}/${model_name}"

            mkdir -p "$model_dir" "$metrics_dir" "$log_dir"
            log_file="${log_dir}/training.log"

            echo "[`date '+%H:%M:%S'`] [Running: ${#RUNNING_PIDS[@]} | Left: ${#PENDING_TASKS[@]}] Start: ${task_string} -> GPU ${chosen_gpu}"

            # --- 启动Python子进程 (关键修改：限制线程数) ---
            # 强制设置 OMP/MKL 线程数为 1，防止 CPU 过载
            OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
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

            # 【关键】将 PID 绑定到 GPU，防止该 GPU 被重复分配
            GPU_BUSY_PID["$chosen_gpu"]="$pid"
            RUNNING_PIDS+=("$pid")

            unset 'PENDING_TASKS[i]'
            launched_in_this_pass=true

            # 稍微快一点的间隔，因为有 PID 锁，不怕冲突
            sleep 2
        fi
    done

    # 重新整理数组索引
    PENDING_TASKS=("${PENDING_TASKS[@]}")

    if ! $launched_in_this_pass && ((${#RUNNING_PIDS[@]} > 0)); then
        # 如果一轮下来没启动任何任务，说明 GPU 满了，等待任意一个任务结束
        # sleep 10 秒避免死循环空转太快
        sleep 10

        # 清理已完成的 PID 列表（仅用于显示计数，实际调度靠 GPU_BUSY_PID）
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
echo "🎉 全部网格搜索任务完成！完成时间: $(date)"
exit 0