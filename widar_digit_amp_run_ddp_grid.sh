#!/usr/bin/env bash
# nohup ./widar_digit_amp_run_ddp_grid.sh > rate_run_ddp_grid.log 2>&1 &
set -u
FREEZE_COPY=1      # 1=复制sensefi目录到_runs后在副本里跑；0=直接当前目录跑
AUTO_CLEAN=0       # 1=跑完自动删除副本目录；0=不自动删
# ====== [新增] 冻结运行：复制整个 sensefi 目录到同级 _runs 下（排除指定子目录） ======
# 防无限循环：第一次复制后会 export FROZEN_RUN=1，副本里再执行不会再次复制
if [[ "${FREEZE_COPY:-1}" == "1" && -z "${FROZEN_RUN:-}" ]]; then
  command -v rsync >/dev/null 2>&1 || { echo "❌ 缺少 rsync，请先安装"; exit 1; }
  # 脚本所在目录就是你要复制的 sensefi 目录
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJ_DIR="$SCRIPT_DIR"
  PROJ_NAME="$(basename "$PROJ_DIR")"
  # 副本放到 sensefi 同级目录的 _runs/ 下，这样 ../datasets/... 相对路径不变
  RUN_PARENT="$(dirname "$PROJ_DIR")"
  TS="$(date +%Y%m%d_%H%M%S)"
  RUN_DIR="${RUN_PARENT}/${PROJ_NAME}_run_${TS}"

  #mkdir -p "$RUN_PARENT"
  echo "✅ FREEZE_COPY=1: 复制目录 $PROJ_DIR -> $RUN_DIR"
  rsync -a \
    --exclude '.git/' \
    --exclude 'Analysis/' \
    --exclude 'cur/' \
    --exclude 'deprecated/' \
    --exclude 'img/' \
    --exclude 'parrllel_gpu/' \
    --exclude 'single_gpu/' \
    --exclude 'tools/' \
    --exclude '__pycache__/' --exclude '*.pyc' \
    "$PROJ_DIR/" "$RUN_DIR/"
  export FROZEN_RUN=1
  export FROZEN_DIR="$RUN_DIR"
  cd "$RUN_DIR"
  # 方式1（推荐）：重跑“同名脚本”（不需要写死脚本名）
  exec bash "./$(basename "$0")" "$@"
  # 方式2（可选）：如果你非要写死脚本名，就用这一行替换上面那行：
  # exec bash "./widar_digit_amp_run_ddp_grid.sh" "$@"
fi
# ======================================================================
PYTHON_SCRIPT="run.py"


#DATASET_NAME="Widar_digit_conj"
DATASET_NAME="Widar_digit_amp"

DATASET_ROOT_DIR="../datasets/sense-fi"


#BASE_EXP_NAME="amp_rate_mask_rec_fastista_fft_0_02_iter80_$(date +%Y%m%d_%H%M%S)"
#BASE_EXP_NAME="amp_rate_mask_rec_mabf_blk3_fftblock1_20260118_003818"
BASE_EXP_NAME="amp_rate_mask_rec_mabf_blk3_hard_dc_lam1_changeloss_beta0_$(date +%Y%m%d_%H%M%S)"
use_energy_input=0
#use_mask_0=1
#is_rec=1
#istanet csdc
rec_model_method=mabf
csdc_blocks=3
rec_alpha=0.5

lam_miss=1.0
beta=0.0
use_mask_0=1
is_rec=1
#SAMPLE_METHODS=(equidistant gaussian poisson)
SAMPLE_METHODS=(equidistant)
#SAMPLE_RATES=(1)
SAMPLE_RATES=(0.05 0.1 0.2 0.5)
#SAMPLE_RATES=(0.0125 0.025 0.05 0.1 0.125 0.25 0.5 1)
# 你希望使用的物理 GPU（顺序决定分组方式）
GPU_LIST=(2 3 4)           # 例：两卡
#GPU_LIST=(0 1 2 3)      # 例：四卡
# 每个任务使用几张 GPU：1=单卡；2=两卡DDP；4=四卡DDP
GPUS_PER_TASK=3
# 全局 batch（所有GPU加起来）
GLOBAL_BATCH_SIZE=126

#INTERPOLATION_METHODS=(linear cubic nearest)
INTERPOLATION_METHODS=(linear)

MODELS=(ResNet18)
CLEAN_LOCKS=0      # 1=启动时强制清空所有锁(慎用!); 0=不清锁(安全默认)

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

LOCK_DIR="/tmp/widar_gpu_locks_mkdir"
mkdir -p "$LOCK_DIR"

timestamp() { date '+%H:%M:%S'; }

# ---- 生成固定分组：GPU_LIST 按顺序切块，每块 GPUS_PER_TASK 张 ----
declare -a GPU_GROUPS=()
len=${#GPU_LIST[@]}
need=${GPUS_PER_TASK}
if (( need < 1 )); then echo "GPUS_PER_TASK must >=1"; exit 1; fi
if (( len < need )); then echo "GPU_LIST 太短"; exit 1; fi

for ((i=0; i+need<=len; i+=need)); do
  group=("${GPU_LIST[@]:i:need}")
  GPU_GROUPS+=("${group[*]}")
done

echo "=============================================================="
echo "启动时间：$(date)"
echo "GPU_LIST=(${GPU_LIST[*]})"
echo "GPU_GROUPS:"
for g in "${GPU_GROUPS[@]}"; do echo "  - [$g]"; done
echo "GPUS_PER_TASK=$GPUS_PER_TASK"
echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
echo "EXP=$BASE_EXP_NAME"
echo "=============================================================="
#





# 清理可能残留的锁（上次异常退出会留下）
# 你也可以注释掉这行，手动清理
#rm -rf "${LOCK_DIR}/gpu_"*.lockdir 2>/dev/null || true
# =========================================================

# [智能锁清理逻辑]
# 只有当你明确设置 CLEAN_LOCKS=1 时才执行清理
# 场景：你是第一个启动的脚本，或者你确定之前跑崩了留下了死锁
# =========================================================
if [[ "${CLEAN_LOCKS:-0}" == "1" ]]; then
    echo "⚠️  [警告] CLEAN_LOCKS=1: 正在强制清理残留的 GPU 锁(仅限本脚本GPU_LIST)..."
    for gpu in "${GPU_LIST[@]}"; do
      rm -rf "${LOCK_DIR}/gpu_${gpu}.lockdir" 2>/dev/null || true
    done
else
    echo "ℹ️  [安全模式] 跳过清理锁 (CLEAN_LOCKS=0)。"
    echo "    提示: 如果发现任务一直卡在等待状态，请尝试运行: CLEAN_LOCKS=1 $0"
fi
# =========================================================
# ---- 任务列表 ----
declare -a PENDING_TASKS=()
for s_method in "${SAMPLE_METHODS[@]}"; do
  for rate in "${SAMPLE_RATES[@]}"; do
    if [[ "$rate" == "1" || "$rate" == "1.0" ]]; then
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
echo "✅ 任务列表生成完毕，总共需要运行 ${#PENDING_TASKS[@]} 个实验。"

# ---- 锁：尝试为一个 group 上锁，成功返回0 ----
acquire_group_lock() {
  local group_str="$1"
  # shellcheck disable=SC2206
  local gpus=($group_str)
  local created=()

  for gpu in "${gpus[@]}"; do
    local d="${LOCK_DIR}/gpu_${gpu}.lockdir"
    if mkdir "$d" 2>/dev/null; then
      created+=("$d")
    else
      # rollback
      for x in "${created[@]}"; do rmdir "$x" 2>/dev/null || true; done
      return 1
    fi
  done
  return 0
}

release_group_lock() {
  local group_str="$1"
  # shellcheck disable=SC2206
  local gpus=($group_str)
  for gpu in "${gpus[@]}"; do
    local d="${LOCK_DIR}/gpu_${gpu}.lockdir"
    rmdir "$d" 2>/dev/null || rm -rf "$d" 2>/dev/null || true
  done
}

# ---- 调度 ----
declare -A GROUP_PID=()

while ((${#PENDING_TASKS[@]} > 0)); do
  launched=false

  # 先刷新 group 是否空闲（pid 不存在就认为空闲，并释放残留锁）
  for group_str in "${GPU_GROUPS[@]}"; do
    pid="${GROUP_PID[$group_str]-}"
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      unset 'GROUP_PID[$group_str]'
      release_group_lock "$group_str"
    fi
  done

  for i in "${!PENDING_TASKS[@]}"; do
    task="${PENDING_TASKS[i]}"
    IFS=':' read -r sample_method sample_rate interpolation_method model_name <<< "$task"

    # 找一个空闲 group：能成功 mkdir 上锁的就是空闲
    chosen_group=""
    for group_str in "${GPU_GROUPS[@]}"; do
      if [[ -z "${GROUP_PID[$group_str]-}" ]]; then
        if acquire_group_lock "$group_str"; then
          chosen_group="$group_str"
          break
        fi
      fi
    done

    if [[ -z "$chosen_group" ]]; then
      continue
    fi

    # 目录
    exp_sub_dir="method_${sample_method}/rate_${sample_rate}/interp_${interpolation_method}"
    model_dir="${DATASET_ROOT_DIR}/${DATASET_NAME}/EXP/${BASE_EXP_NAME}/Model Parameters/${exp_sub_dir}/${model_name}"
    metrics_dir="${DATASET_ROOT_DIR}/${DATASET_NAME}/EXP/${BASE_EXP_NAME}/Metrics/${exp_sub_dir}/${model_name}"
    log_dir="${DATASET_ROOT_DIR}/${DATASET_NAME}/EXP/${BASE_EXP_NAME}/Logs/${exp_sub_dir}/${model_name}"
    mkdir -p "$model_dir" "$metrics_dir" "$log_dir"
    log_file="${log_dir}/training.log"

    echo "[$(timestamp)] 🚀 启动任务: ${task} on GPUs [${chosen_group}]"

    (
      set +e
      trap 'release_group_lock "'"$chosen_group"'"' EXIT

      #cuda_visible=$(IFS=, ; echo ${chosen_group})
      cuda_visible="${chosen_group// /,}"   # "0 1" -> "0,1"


      if [[ "$GPUS_PER_TASK" -eq 1 ]]; then
        CUDA_VISIBLE_DEVICES="${cuda_visible}" \
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
          --is_rec "$is_rec" \
          --rec_alpha "$rec_alpha" \
          --csdc_blocks "$csdc_blocks" \
          --rec_model "$rec_model_method" \
          --global_batch_size "$GLOBAL_BATCH_SIZE" \
          --lam_miss "$lam_miss" \
          --beta "$beta" \
          > "$log_file" 2>&1
      else
        CUDA_VISIBLE_DEVICES="${cuda_visible}" \
        torchrun --standalone --nproc_per_node="$GPUS_PER_TASK" "$PYTHON_SCRIPT" \
          --dataset "$DATASET_NAME" \
          --model "$model_name" \
          --sample_rate "$sample_rate" \
          --sample_method "$sample_method" \
          --interpolation "$interpolation_method" \
          --use_energy_input "$use_energy_input" \
          --use_mask_0 "$use_mask_0" \
          --model_save_dir "$model_dir" \
          --metrics_save_dir "$metrics_dir" \
          --is_rec "$is_rec" \
          --rec_alpha "$rec_alpha" \
          --csdc_blocks "$csdc_blocks" \
          --rec_model "$rec_model_method" \
          --global_batch_size "$GLOBAL_BATCH_SIZE" \
          --lam_miss "$lam_miss" \
          --beta "$beta" \
          > "$log_file" 2>&1
      fi
      exit $?
    ) &

    pid=$!
    GROUP_PID[$chosen_group]=$pid

    unset 'PENDING_TASKS[i]'
    launched=true
    sleep 1
  done

  PENDING_TASKS=("${PENDING_TASKS[@]}")

  if ! $launched; then
    echo "[$(timestamp)] 💤 没有空闲 GPU 组，等待... (剩余任务: ${#PENDING_TASKS[@]})"
    sleep 30
  fi
done

echo "🎉 所有任务已提交，等待最后任务完成..."
wait
echo "✅ 全部完成！"

if [[ "${AUTO_CLEAN:-0}" == "1" && "${FROZEN_RUN:-}" == "1" && -n "${FROZEN_DIR:-}" ]]; then
  echo "🧹 AUTO_CLEAN=1: 删除冻结目录: $FROZEN_DIR"
  rm -rf "$FROZEN_DIR"
fi
