
import numpy as np
import os

# 这会打印出numpy核心库的路径，通常在这里找到它的链接库
print(np.__file__)
# 通常你需要找到同目录下的 _multiarray_umath.cpython*.so 或类似名称的文件
# 示例路径：/path/to/your/venv/lib/python3.x/site-packages/numpy/core/_multiarray_umath.cpython-3x-x86_64-linux-gnu.so

np.show_config()