"""调试 cos_sim 差异问题

问题：
- 之前显示 cos_sim ≈ 0.9
- 现在用 eta=0 确定性采样显示 cos_sim ≈ -0.5

原因分析：
1. DDIMScheduler 默认的 eta 是多少？
2. 每次采样是否有随机性？
"""

import sys
import numpy as np
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray

import torch
from pathlib import Path
import importlib.util

def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

phase4_root = Path('src/phase4')
scheduler_mod = _import_module('scheduler', phase4_root / 'diffusion' / 'scheduler.py')

# 检查 DDIMScheduler 的默认 eta
import inspect
sig = inspect.signature(scheduler_mod.DDIMScheduler.sample)
print("DDIMScheduler.sample 函数签名:")
print(f"  {sig}")
print()

# 查看 sample 方法源码
print("DDIMScheduler.sample 方法源码:")
print(inspect.getsource(scheduler_mod.DDIMScheduler.sample))
print()

# 检查 scheduler 初始化
sig_init = inspect.signature(scheduler_mod.DDIMScheduler.__init__)
print("DDIMScheduler.__init__ 函数签名:")
print(f"  {sig_init}")
