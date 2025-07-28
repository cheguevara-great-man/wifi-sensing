
# 定义变量以便复用和修改
DATASET="NTU-Fi_HAR"
MODEL="LeNet"
SAMPLE_RATE="0.1"
INTERPOLATION="cubic"
EXP_NAME="debug_test_$(date +%Y%m%d_%H%M%S)"

# 创建保存目录
mkdir -p "${EXP_NAME}/models"
mkdir -p "${EXP_NAME}/metrics"

# 执行 Python 脚本
python -u run.py \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --sample_rate "$SAMPLE_RATE" \
    --interpolation "$INTERPOLATION" \
    --model_save_dir "${EXP_NAME}/models" \
    --metrics_save_dir "${EXP_NAME}/metrics" \
    > "${EXP_NAME}/training.log" 2>&1