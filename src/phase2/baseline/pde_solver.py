"""
有限差分扩散求解（Track A Baseline）。
- 在 walkable_mask 上做平滑扩散，非可走区域扩散系数降低（近似绝缘）。
- 返回平滑后的场以及梯度/score 场，供流线或导航使用。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve, distance_transform_edt

LAPLACE_KERNEL = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=float)


def diffuse_density(
    target_density: np.ndarray,
    walkable_mask: np.ndarray,
    num_iters: int = 400,
    alpha: float = 0.15,
    base_diffusivity: float = 1e-3,
    clamp_min: float | None = 0.0,
    clamp_max: float | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    通过离散拉普拉斯迭代平滑目标密度。
    - walkable_mask 为 0/1；非可走区域扩散系数接近 0。
    - alpha 越小越稳定；建议 alpha <= 0.25。
    """
    field = np.asarray(target_density, dtype=float).copy()
    mask = (np.asarray(walkable_mask) > 0).astype(float)
    diffusivity = base_diffusivity + (1.0 - base_diffusivity) * mask

    for _ in range(num_iters):
        lap = convolve(field, LAPLACE_KERNEL, mode="nearest")
        field = field + alpha * diffusivity * lap
        if clamp_min is not None:
            field[field < clamp_min] = clamp_min
        if clamp_max is not None:
            field[field > clamp_max] = clamp_max

    if normalize:
        field = field / (field.max() + 1e-9)
    return field


def compute_distance_field(target_density: np.ndarray, walkable_mask: np.ndarray, threshold: float = 0.01, max_iters: int = 3000):
    """
    计算到 sink 区域的测地距离场（考虑障碍物）。
    使用向量化的迭代扩散方法。
    sink 区域定义为 target_density > threshold 的像素。
    返回距离场（在 sink 处为 0，远离 sink 处为正值）。
    """
    # sink 区域
    sink_mask = target_density > threshold
    mask = (walkable_mask > 0) | sink_mask  # sink 也算可行
    
    H, W = target_density.shape
    INF = float(H + W) * 2
    
    # 初始化距离场
    dist = np.full((H, W), INF, dtype=np.float32)
    dist[sink_mask] = 0.0
    
    # 向量化迭代扩散
    for it in range(max_iters):
        dist_new = dist.copy()
        
        # 四邻域传播（向量化）
        dist_new[1:, :] = np.minimum(dist_new[1:, :], dist[:-1, :] + 1)
        dist_new[:-1, :] = np.minimum(dist_new[:-1, :], dist[1:, :] + 1)
        dist_new[:, 1:] = np.minimum(dist_new[:, 1:], dist[:, :-1] + 1)
        dist_new[:, :-1] = np.minimum(dist_new[:, :-1], dist[:, 1:] + 1)
        
        # 非可行区域保持大值
        dist_new[~mask] = INF
        # sink 保持 0
        dist_new[sink_mask] = 0
        
        # 检查收敛
        diff = np.abs(dist_new[mask] - dist[mask]).max()
        dist = dist_new
        
        if diff < 0.5:
            print(f"  距离场收敛于迭代 {it+1}")
            break
        
        if (it + 1) % 500 == 0:
            print(f"  迭代 {it+1}, max_diff={diff:.2f}")
    
    return dist


def compute_score_field(field: np.ndarray, eps: float = 1e-6, normalize: bool = True):
    """
    计算梯度与 score（梯度/场强）。
    返回 (grad_y, grad_x), (score_y, score_x)
    """
    grad_y, grad_x = np.gradient(field)
    score_y = grad_y / (field + eps)
    score_x = grad_x / (field + eps)
    if normalize:
        mag = np.hypot(score_y, score_x) + 1e-9
        score_y = score_y / mag
        score_x = score_x / mag
    return (grad_y, grad_x), (score_y, score_x)


def compute_navigation_field(distance_field: np.ndarray, walkable_mask: np.ndarray):
    """
    从距离场计算导航方向场。
    - walkable 区域：方向指向距离最小的 walkable 邻居（即指向 sink）
    - 非 walkable 区域：方向指向最近的 walkable 区域（用于掉网恢复）
    
    使用基于最小邻居的方法（而非梯度），避免边界效应。
    返回归一化的 (nav_y, nav_x)。
    """
    H, W = distance_field.shape
    walkable_bool = walkable_mask > 0
    
    # === 1. Walkable 区域：指向距离最小的 walkable 邻居 ===
    # 初始化为当前距离
    min_dist = distance_field.copy()
    nav_y = np.zeros((H, W), dtype=np.float32)
    nav_x = np.zeros((H, W), dtype=np.float32)
    
    # 4邻域偏移: (dy, dx)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for dy, dx in offsets:
        # 创建邻居距离数组（如果邻居不可行则设为 inf）
        neighbor_dist = np.full((H, W), np.inf, dtype=np.float32)
        
        # 根据偏移提取邻居距离
        if dy == -1:  # 上方邻居
            neighbor_dist[1:, :] = np.where(walkable_mask[:-1, :] > 0, distance_field[:-1, :], np.inf)
        elif dy == 1:  # 下方邻居
            neighbor_dist[:-1, :] = np.where(walkable_mask[1:, :] > 0, distance_field[1:, :], np.inf)
        elif dx == -1:  # 左方邻居
            neighbor_dist[:, 1:] = np.where(walkable_mask[:, :-1] > 0, distance_field[:, :-1], np.inf)
        elif dx == 1:  # 右方邻居
            neighbor_dist[:, :-1] = np.where(walkable_mask[:, 1:] > 0, distance_field[:, 1:], np.inf)
        
        # 如果这个邻居距离更小，更新方向
        update_mask = walkable_bool & (neighbor_dist < min_dist)
        min_dist[update_mask] = neighbor_dist[update_mask]
        nav_y[update_mask] = dy
        nav_x[update_mask] = dx
    
    # 归一化（4邻域方向已经是单位长度）
    mag = np.sqrt(nav_y**2 + nav_x**2) + 1e-9
    nav_y = nav_y / mag
    nav_x = nav_x / mag
    
    # 在 sink 区域（距离=0）或没有更好邻居的点设为0
    no_direction = (nav_y == 0) & (nav_x == 0)
    # 这些点可能是 sink 或局部极小值
    
    # === 2. 非 walkable 区域：指向最近的 walkable 点 ===
    dist_to_walkable = distance_transform_edt(~walkable_bool)
    
    # 负梯度指向 walkable 区域
    off_grad_y, off_grad_x = np.gradient(dist_to_walkable)
    off_nav_y = -off_grad_y
    off_nav_x = -off_grad_x
    
    # 归一化
    off_mag = np.sqrt(off_nav_y**2 + off_nav_x**2) + 1e-9
    off_nav_y = off_nav_y / off_mag
    off_nav_x = off_nav_x / off_mag
    
    # 在非 walkable 区域使用恢复方向
    non_walkable = ~walkable_bool
    nav_y[non_walkable] = off_nav_y[non_walkable]
    nav_x[non_walkable] = off_nav_x[non_walkable]
    
    return nav_y.astype(np.float32), nav_x.astype(np.float32)


def compute_potential_field(
    target_density: np.ndarray,
    walkable_mask: np.ndarray,
    decay_rate: float = 0.05,
) -> np.ndarray:
    """
    计算加权势能场：每个 sink 产生加权引力场（距离衰减），叠加后得到全局势能场。
    
    势能公式: potential(x) = sum_i ( weight_i / (1 + decay * dist_i(x)) )
    
    这种方法：
    1. 保留每个 sink 的流量权重
    2. 梯度随距离衰减但不会消失
    3. 方向始终指向"综合吸引力"最强的方向（权重大且距离近的 sink）
    
    返回：势能场（sink 处高，远离 sink 处低）
    """
    from scipy.ndimage import distance_transform_edt
    
    print(f"  计算势能场: 加权引力场叠加 (decay_rate={decay_rate})...")
    
    H, W = target_density.shape
    
    # 识别 sink 点（target_density > 0 的位置）
    sink_threshold = target_density.max() * 0.001
    sink_mask = target_density > sink_threshold
    
    # 获取 sink 坐标和权重
    sink_ys, sink_xs = np.where(sink_mask)
    sink_weights = target_density[sink_mask]
    
    print(f"    找到 {len(sink_ys)} 个 sink 点，权重范围: [{sink_weights.min():.4f}, {sink_weights.max():.4f}]")
    
    # 创建坐标网格
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # 对高权重 sink 逐个计算引力场（为效率只处理权重最高的 100 个点）
    top_k = min(100, len(sink_ys))
    top_indices = np.argsort(sink_weights)[-top_k:]
    
    field = np.zeros((H, W), dtype=np.float64)
    
    for idx in top_indices:
        sy, sx = sink_ys[idx], sink_xs[idx]
        weight = sink_weights[idx]
        
        # 欧几里得距离
        dist = np.sqrt((yy - sy)**2 + (xx - sx)**2)
        
        # 引力场：weight / (1 + decay * dist)
        contribution = weight / (1.0 + decay_rate * dist)
        field += contribution
    
    # 归一化到 [0, 1]
    field = field / (field.max() + 1e-9)
    
    # 统计
    mask = walkable_mask > 0
    walkable_coverage = (field[mask] > 0.001).sum() / mask.sum()
    
    # 检查梯度强度
    grad_y, grad_x = np.gradient(field)
    grad_mag = np.sqrt(grad_y**2 + grad_x**2)
    
    print(f"    完成: max={field.max():.4f}, "
          f"walkable mean={field[mask].mean():.4f}, coverage(>0.001)={walkable_coverage*100:.1f}%")
    print(f"    梯度: mean={grad_mag[mask].mean():.6f}, max={grad_mag.max():.4f}")
    
    return field.astype(np.float32)


def compute_potential_navigation(
    potential_field: np.ndarray,
    walkable_mask: np.ndarray,
) -> tuple:
    """
    从势能场计算导航方向（保留流量权重信息）。
    
    使用势能场的梯度方向。由于使用引力场叠加方法，梯度在所有位置都非零。
    
    返回：(nav_y, nav_x) 归一化的导航方向
    """
    H, W = potential_field.shape
    walkable_bool = walkable_mask > 0
    
    # === 1. 计算势能场的梯度 ===
    grad_y, grad_x = np.gradient(potential_field)
    
    # 梯度方向（指向势能增大的方向，即 sink）
    nav_y = grad_y.copy()
    nav_x = grad_x.copy()
    
    # 归一化
    mag = np.sqrt(nav_y**2 + nav_x**2) + 1e-9
    nav_y = nav_y / mag
    nav_x = nav_x / mag
    
    # === 2. 非 walkable 区域：指向最近的 walkable 点（用于脱困）===
    dist_to_walkable = distance_transform_edt(~walkable_bool)
    off_grad_y, off_grad_x = np.gradient(dist_to_walkable)
    off_nav_y = -off_grad_y
    off_nav_x = -off_grad_x
    off_mag = np.sqrt(off_nav_y**2 + off_nav_x**2) + 1e-9
    off_nav_y = off_nav_y / off_mag
    off_nav_x = off_nav_x / off_mag
    
    non_walkable = ~walkable_bool
    nav_y[non_walkable] = off_nav_y[non_walkable]
    nav_x[non_walkable] = off_nav_x[non_walkable]
    
    # 统计
    grad_mag = np.sqrt(grad_y**2 + grad_x**2)
    print(f"  势能导航场: 梯度 mean={grad_mag[walkable_bool].mean():.6f}, "
          f"min={grad_mag[walkable_bool].min():.6f}")
    
    walkable_nav_mag = np.sqrt(nav_y[walkable_bool]**2 + nav_x[walkable_bool]**2)
    valid_ratio = (walkable_nav_mag > 0.5).sum() / walkable_bool.sum()
    print(f"  势能导航场: walkable 区域有效方向比例 = {valid_ratio*100:.1f}%")
    
    return nav_y.astype(np.float32), nav_x.astype(np.float32)


def solve_field(
    walkable_mask: np.ndarray,
    target_density: np.ndarray,
    num_iters: int = 400,
    alpha: float = 0.15,
    base_diffusivity: float = 1e-3,
    clamp_min: float | None = 0.0,
    clamp_max: float | None = None,
    normalize: bool = True,
    use_distance_field: bool = False,
    use_potential_field: bool = True,
):
    """
    便捷封装：扩散 -> 梯度 -> score。
    
    导航场计算方式：
    - use_potential_field=True (默认): 使用势能场，保留流量权重
    - use_distance_field=True: 使用距离场，只考虑最近 sink
    
    返回 dict，包括 smooth_field、grad、score、nav 等。
    """
    smooth = diffuse_density(
        target_density,
        walkable_mask,
        num_iters=num_iters,
        alpha=alpha,
        base_diffusivity=base_diffusivity,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        normalize=normalize,
    )
    grad, score = compute_score_field(smooth, normalize=True)
    
    result = {"smooth_field": smooth, "grad": grad, "score": score}
    
    if use_potential_field:
        # 使用势能场计算导航方向（保留流量权重）
        potential = compute_potential_field(target_density, walkable_mask)
        nav_y, nav_x = compute_potential_navigation(potential, walkable_mask)
        result["potential_field"] = potential
        result["nav"] = (nav_y, nav_x)
    elif use_distance_field:
        # 使用距离场计算导航方向（只考虑最近 sink）
        dist_field = compute_distance_field(target_density, walkable_mask, threshold=0.001)
        nav_y, nav_x = compute_navigation_field(dist_field, walkable_mask)
        result["distance_field"] = dist_field
        result["nav"] = (nav_y, nav_x)
    
    return result
