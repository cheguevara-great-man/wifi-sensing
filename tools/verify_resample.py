##为了验证均匀取点会不会导致重复，结论是不会重复
import numpy as np


def demonstrate_resampling_repetition(original_len, resample_len):
    """
    演示当采样点数接近原始点数时，均匀挑选是否会产生重复。
    """
    print(f"--- 正在从 {original_len} 个点中挑选 {resample_len} 个点 ---")

    # 1. 生成均匀分布的浮点数索引
    pick_indices_float = np.linspace(0, original_len - 1, resample_len)
    print(f"生成的浮点数索引:\n{np.round(pick_indices_float, 2)}")

    # 2. 将它们四舍五入为整数索引
    pick_indices_int = np.round(pick_indices_float).astype(int)
    print(f"四舍五入后的整数索引:\n{pick_indices_int}")

    # 3. 检查是否有重复
    unique_indices = np.unique(pick_indices_int)
    num_unique = len(unique_indices)

    print(f"\n--- 结果分析 ---")
    print(f"期望的独立索引数: {resample_len}")
    print(f"实际得到的独立索引数: {num_unique}")

    if num_unique < resample_len:
        print(f"🔴 结论: 出现了重复！有 {resample_len - num_unique} 个索引是重复的。")
        # 找出重复的元素
        counts = np.bincount(pick_indices_int)
        repeated_indices = np.where(counts > 1)[0]
        print(f"   重复的索引是: {repeated_indices}")
    else:
        print(f"🟢 结论: 没有出现重复。")

    print("-" * 40)


# --- 案例1: 500中取450 (高密度采样) ---
demonstrate_resampling_repetition(original_len=500, resample_len=450)

# --- 案例2: 10中取9 (更直观的小例子) ---
demonstrate_resampling_repetition(original_len=10, resample_len=9)

# --- 案例3: 500中取200 (稀疏采样) ---
demonstrate_resampling_repetition(original_len=500, resample_len=200)