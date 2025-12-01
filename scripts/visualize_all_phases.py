"""
GeoExplicit SFM 全阶段可视化分析

生成高质量的科学图表，适合学术期刊发表。
包括：Phase 1 (Sink识别)、Phase 2 (场计算)、Phase 3 (轨迹生成)、Phase 4 (Diffusion Policy)

Author: GeoExplicit SFM Team
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ================== 全局样式设置 ==================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# 自定义色彩
COLORS = {
    'primary': '#2C3E50',
    'secondary': '#E74C3C',
    'accent': '#3498DB',
    'success': '#27AE60',
    'warning': '#F39C12',
    'road': '#7F8C8D',
    'sink': '#E74C3C',
    'trajectory': '#3498DB',
}

# 自定义 colormap
def create_scientific_cmap():
    """创建适合科学出版的渐变色"""
    colors = ['#FFFFFF', '#E3F2FD', '#90CAF9', '#42A5F5', '#1976D2', '#0D47A1']
    return LinearSegmentedColormap.from_list('scientific_blue', colors)


# ================== Phase 1: Sink 识别 ==================
def visualize_phase1(output_path: Path = None):
    """
    Phase 1: 可视化 Sink 点识别结果
    
    展示：
    1. 道路网络与人口权重
    2. Sink 点分布与权重热力图
    3. 流量分布直方图
    """
    import pandas as pd
    
    # 加载数据
    mask_path = PROJECT_ROOT / "data" / "processed" / "walkable_mask.npy"
    sinks_path = PROJECT_ROOT / "data" / "processed" / "sinks_phase1.csv"
    density_path = PROJECT_ROOT / "data" / "processed" / "target_density.npy"
    
    walkable_mask = np.load(mask_path)
    target_density = np.load(density_path)
    sinks_df = pd.read_csv(sinks_path)
    
    H, W = walkable_mask.shape
    
    # 创建图形
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.8], wspace=0.25)
    
    # --- Panel A: 道路网络 ---
    ax1 = fig.add_subplot(gs[0])
    
    # 道路背景
    road_display = np.where(walkable_mask, 0.9, 0.0)
    ax1.imshow(road_display.T, cmap='gray', origin='lower', vmin=0, vmax=1)
    
    # Sink 点标记（按权重着色）
    if 'grid_x' in sinks_df.columns and 'grid_y' in sinks_df.columns:
        sink_x = sinks_df['grid_x'].values
        sink_y = sinks_df['grid_y'].values
        weights = sinks_df['weight'].values if 'weight' in sinks_df.columns else np.ones(len(sink_x))
        
        # 归一化权重用于着色
        weights_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)
        
        scatter = ax1.scatter(sink_x, sink_y, c=weights_norm, cmap='YlOrRd', 
                             s=20 + 80 * weights_norm, alpha=0.8, edgecolors='white', linewidths=0.5)
        
        # colorbar
        cbar = plt.colorbar(scatter, ax=ax1, shrink=0.7, pad=0.02)
        cbar.set_label('Normalized Flow Weight', fontsize=9)
    
    ax1.set_title('(a) Road Network & Sink Distribution', fontweight='bold')
    ax1.set_xlabel('Grid X (100m/pixel)')
    ax1.set_ylabel('Grid Y (100m/pixel)')
    ax1.set_aspect('equal')
    
    # --- Panel B: 目标密度场 ---
    ax2 = fig.add_subplot(gs[1])
    
    # 平滑密度场用于可视化
    density_smooth = gaussian_filter(target_density, sigma=3)
    
    # 显示密度场
    im = ax2.imshow(density_smooth.T, cmap=create_scientific_cmap(), origin='lower',
                    norm=Normalize(vmin=0, vmax=np.percentile(density_smooth[density_smooth > 0], 99)))
    
    # 叠加道路轮廓
    road_contour = np.where(walkable_mask, 1, np.nan)
    ax2.contour(road_contour.T, levels=[0.5], colors='gray', linewidths=0.5, alpha=0.5)
    
    cbar2 = plt.colorbar(im, ax=ax2, shrink=0.7, pad=0.02)
    cbar2.set_label('Target Density', fontsize=9)
    
    ax2.set_title('(b) Target Density Field', fontweight='bold')
    ax2.set_xlabel('Grid X (100m/pixel)')
    ax2.set_ylabel('Grid Y (100m/pixel)')
    ax2.set_aspect('equal')
    
    # --- Panel C: 流量分布 ---
    ax3 = fig.add_subplot(gs[2])
    
    if 'weight' in sinks_df.columns:
        weights = sinks_df['weight'].values
        weights_log = np.log10(weights + 1)
        
        ax3.hist(weights_log, bins=30, color=COLORS['accent'], edgecolor='white', alpha=0.8)
        ax3.axvline(np.median(weights_log), color=COLORS['secondary'], linestyle='--', 
                   linewidth=1.5, label=f'Median: {10**np.median(weights_log):.0f}')
        
        ax3.set_xlabel('log₁₀(Flow Weight + 1)')
        ax3.set_ylabel('Count')
        ax3.legend(frameon=False)
    
    ax3.set_title('(c) Flow Weight Distribution', fontweight='bold')
    
    # 添加统计信息
    stats_text = (f"Total Sinks: {len(sinks_df)}\n"
                  f"Grid: {H} × {W}\n"
                  f"Resolution: 100m")
    ax3.text(0.95, 0.95, stats_text, transform=ax3.transAxes, fontsize=8,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Phase 1: Sink Identification from OD Flow Data', fontsize=14, fontweight='bold', y=1.02)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    plt.close()
    return fig


# ================== Phase 2: 场计算 ==================
def visualize_phase2(output_path: Path = None):
    """
    Phase 2: 可视化场计算结果
    
    展示：
    1. 势能场（加权引力叠加）
    2. 导航方向场（流线图）
    3. 场强度剖面
    """
    # 加载数据
    mask_path = PROJECT_ROOT / "data" / "processed" / "walkable_mask.npy"
    potential_path = PROJECT_ROOT / "data" / "processed" / "potential_field.npy"
    nav_path = PROJECT_ROOT / "data" / "processed" / "nav_baseline.npz"
    
    walkable_mask = np.load(mask_path)
    
    if not potential_path.exists():
        print("Warning: potential_field.npy not found, generating...")
        return None
    
    potential_field = np.load(potential_path)
    nav_data = np.load(nav_path)
    nav_y, nav_x = nav_data['nav_y'], nav_data['nav_x']
    
    H, W = walkable_mask.shape
    
    # 创建图形
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.8], wspace=0.25)
    
    # --- Panel A: 势能场 ---
    ax1 = fig.add_subplot(gs[0])
    
    # 势能场（只在道路上显示）
    potential_display = np.ma.masked_where(~(walkable_mask > 0), potential_field)
    
    im1 = ax1.imshow(potential_display.T, cmap='plasma', origin='lower',
                     norm=Normalize(vmin=0, vmax=potential_field.max()))
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.7, pad=0.02)
    cbar1.set_label('Potential (Weighted Gravity)', fontsize=9)
    
    ax1.set_title('(a) Potential Field\n(Gravity Superposition)', fontweight='bold')
    ax1.set_xlabel('Grid X (100m/pixel)')
    ax1.set_ylabel('Grid Y (100m/pixel)')
    ax1.set_aspect('equal')
    
    # --- Panel B: 导航流线图 ---
    ax2 = fig.add_subplot(gs[1])
    
    # 背景：道路
    road_display = np.where(walkable_mask, 0.95, 0.3)
    ax2.imshow(road_display.T, cmap='gray', origin='lower', vmin=0, vmax=1)
    
    # 流线图（降采样）- 使用 quiver 替代 streamplot
    step = 25
    Y, X = np.mgrid[0:H:step, 0:W:step]
    U = nav_x[::step, ::step]
    V = nav_y[::step, ::step]
    
    # 速度大小
    speed = np.sqrt(U**2 + V**2)
    
    # 只在道路上绘制箭头
    mask_sub = walkable_mask[::step, ::step] > 0
    
    # 使用 quiver 绘制向量场
    ax2.quiver(X[mask_sub], Y[mask_sub], U[mask_sub], V[mask_sub],
               speed[mask_sub], cmap='viridis', scale=30, width=0.003,
               headwidth=4, headlength=5, alpha=0.8)
    
    ax2.set_title('(b) Navigation Field\n(Streamlines)', fontweight='bold')
    ax2.set_xlabel('Grid X (100m/pixel)')
    ax2.set_ylabel('Grid Y (100m/pixel)')
    ax2.set_aspect('equal')
    ax2.set_xlim(0, W)
    ax2.set_ylim(0, H)
    
    # --- Panel C: 梯度强度分析 ---
    ax3 = fig.add_subplot(gs[2])
    
    # 计算梯度强度
    grad_y, grad_x = np.gradient(potential_field)
    grad_mag = np.sqrt(grad_y**2 + grad_x**2)
    
    # 只看道路上的梯度
    road_mask = walkable_mask > 0
    grad_road = grad_mag[road_mask]
    
    # 梯度分布
    ax3.hist(grad_road, bins=50, color=COLORS['accent'], edgecolor='white', alpha=0.8, density=True)
    ax3.axvline(np.mean(grad_road), color=COLORS['secondary'], linestyle='--', 
               linewidth=1.5, label=f'Mean: {np.mean(grad_road):.4f}')
    ax3.axvline(np.median(grad_road), color=COLORS['success'], linestyle=':', 
               linewidth=1.5, label=f'Median: {np.median(grad_road):.4f}')
    
    ax3.set_xlabel('Gradient Magnitude')
    ax3.set_ylabel('Density')
    ax3.set_title('(c) Gradient Distribution\n(On-Road)', fontweight='bold')
    ax3.legend(frameon=False, fontsize=8)
    
    # 添加技术说明
    tech_text = (
        "Method: Weighted Gravity\n"
        f"$\\phi(x) = \\sum_i \\frac{{w_i}}{{1 + \\alpha \\cdot d_i(x)}}$\n"
        f"Decay: α = 0.05"
    )
    ax3.text(0.95, 0.5, tech_text, transform=ax3.transAxes, fontsize=8,
             verticalalignment='center', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('Phase 2: Navigation Field Computation', fontsize=14, fontweight='bold', y=1.02)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    plt.close()
    return fig


# ================== Phase 3: 轨迹生成 ==================
def visualize_phase3(output_path: Path = None, num_trajectories: int = 50):
    """
    Phase 3: 可视化轨迹生成结果
    
    展示：
    1. 轨迹分布与密度热力图
    2. 速度场分析
    3. 行为统计
    """
    # 加载数据
    mask_path = PROJECT_ROOT / "data" / "processed" / "walkable_mask.npy"
    traj_h5_path = PROJECT_ROOT / "data" / "output" / "trajectories.h5"
    
    walkable_mask = np.load(mask_path)
    H, W = walkable_mask.shape
    
    if not traj_h5_path.exists():
        print("Warning: trajectories.h5 not found")
        return None
    
    # 读取轨迹
    with h5py.File(traj_h5_path, 'r') as f:
        positions = f['positions'][:]
        velocities = f['velocities'][:] if 'velocities' in f else None
    
    T, N, _ = positions.shape
    
    # 创建图形
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.2, 1, 0.8, 0.8], wspace=0.3)
    
    # --- Panel A: 轨迹与密度热力图 ---
    ax1 = fig.add_subplot(gs[0])
    
    # 计算位置密度热力图
    all_pos = positions.reshape(-1, 2)
    valid_mask = (all_pos[:, 0] > 0) & (all_pos[:, 1] > 0)
    all_pos = all_pos[valid_mask]
    
    # 2D 直方图
    density_map, xedges, yedges = np.histogram2d(
        all_pos[:, 1], all_pos[:, 0],  # 注意 x, y 顺序
        bins=[W//4, H//4],
        range=[[0, W], [0, H]]
    )
    density_map = gaussian_filter(density_map.T, sigma=2)
    
    # 背景：密度热力图
    extent = [0, W, 0, H]
    im = ax1.imshow(density_map.T, cmap='hot', origin='lower', extent=extent, alpha=0.7)
    
    # 叠加道路轮廓
    ax1.contour(walkable_mask.T, levels=[0.5], colors='white', linewidths=0.3, alpha=0.5,
               extent=extent, origin='lower')
    
    # 抽样绘制轨迹
    sample_indices = np.random.choice(N, min(num_trajectories, N), replace=False)
    
    for idx in sample_indices:
        traj = positions[:, idx, :]
        # 只绘制有效部分
        valid = (traj[:, 0] > 0) & (traj[:, 1] > 0)
        if valid.sum() > 10:
            traj_valid = traj[valid]
            
            # 使用 LineCollection 按速度着色
            points = np.array([traj_valid[:, 1], traj_valid[:, 0]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # 速度（位移距离）
            speeds = np.sqrt(np.sum(np.diff(traj_valid, axis=0)**2, axis=1))
            
            lc = LineCollection(segments, cmap='winter', alpha=0.5, linewidths=0.5)
            lc.set_array(speeds)
            ax1.add_collection(lc)
    
    cbar = plt.colorbar(im, ax=ax1, shrink=0.7, pad=0.02)
    cbar.set_label('Trajectory Density', fontsize=9)
    
    ax1.set_title('(a) Trajectory Distribution & Density', fontweight='bold')
    ax1.set_xlabel('Grid X (100m/pixel)')
    ax1.set_ylabel('Grid Y (100m/pixel)')
    ax1.set_xlim(0, W)
    ax1.set_ylim(0, H)
    ax1.set_aspect('equal')
    
    # --- Panel B: 速度分析 ---
    ax2 = fig.add_subplot(gs[1])
    
    if velocities is not None:
        # 计算速度大小
        speeds = np.sqrt(velocities[:, :, 0]**2 + velocities[:, :, 1]**2)
        speeds_flat = speeds.flatten()
        speeds_valid = speeds_flat[(speeds_flat > 0) & (speeds_flat < 10)]
        
        # 速度分布
        ax2.hist(speeds_valid, bins=50, color=COLORS['accent'], edgecolor='white', 
                alpha=0.8, density=True)
        
        # 标记均值和中位数
        mean_speed = np.mean(speeds_valid)
        median_speed = np.median(speeds_valid)
        
        ax2.axvline(mean_speed, color=COLORS['secondary'], linestyle='--', 
                   linewidth=1.5, label=f'Mean: {mean_speed:.2f} px/step')
        ax2.axvline(median_speed, color=COLORS['success'], linestyle=':', 
                   linewidth=1.5, label=f'Median: {median_speed:.2f} px/step')
        
        ax2.set_xlabel('Speed (pixels/step)')
        ax2.set_ylabel('Density')
        ax2.legend(frameon=False, fontsize=8)
    
    ax2.set_title('(b) Speed Distribution', fontweight='bold')
    
    # --- Panel C: 方向变化 ---
    ax3 = fig.add_subplot(gs[2])
    
    if velocities is not None:
        # 计算方向变化
        angles = np.arctan2(velocities[:, :, 0], velocities[:, :, 1])
        angle_changes = np.abs(np.diff(angles, axis=0))
        angle_changes = np.minimum(angle_changes, 2*np.pi - angle_changes)  # 处理周期性
        angle_changes_deg = np.degrees(angle_changes.flatten())
        
        valid_changes = angle_changes_deg[~np.isnan(angle_changes_deg) & (angle_changes_deg > 0)]
        valid_changes = valid_changes[valid_changes < 180]
        
        ax3.hist(valid_changes, bins=40, color=COLORS['warning'], edgecolor='white', 
                alpha=0.8, density=True)
        ax3.axvline(np.mean(valid_changes), color=COLORS['secondary'], linestyle='--',
                   linewidth=1.5, label=f'Mean: {np.mean(valid_changes):.1f}°')
        
        ax3.set_xlabel('Direction Change (degrees)')
        ax3.set_ylabel('Density')
        ax3.legend(frameon=False, fontsize=8)
    
    ax3.set_title('(c) Direction Change', fontweight='bold')
    
    # --- Panel D: 统计摘要 ---
    ax4 = fig.add_subplot(gs[3])
    ax4.axis('off')
    
    # 计算统计数据
    on_road_count = 0
    total_count = 0
    for t in range(T):
        for n in range(N):
            y, x = positions[t, n]
            if y > 0 and x > 0:
                yi, xi = int(y), int(x)
                if 0 <= yi < H and 0 <= xi < W:
                    total_count += 1
                    if walkable_mask[yi, xi] > 0:
                        on_road_count += 1
    
    on_road_ratio = on_road_count / total_count if total_count > 0 else 0
    
    # 轨迹效率（直线距离 / 实际距离）
    efficiencies = []
    for n in range(min(N, 1000)):
        traj = positions[:, n, :]
        valid = (traj[:, 0] > 0) & (traj[:, 1] > 0)
        if valid.sum() > 10:
            traj_valid = traj[valid]
            straight_dist = np.linalg.norm(traj_valid[-1] - traj_valid[0])
            path_dist = np.sum(np.sqrt(np.sum(np.diff(traj_valid, axis=0)**2, axis=1)))
            if path_dist > 0:
                efficiencies.append(straight_dist / path_dist)
    
    stats_text = (
        f"━━━ Simulation Statistics ━━━\n\n"
        f"Agents: {N:,}\n"
        f"Time Steps: {T:,}\n"
        f"Total Samples: {T * N:,}\n\n"
        f"━━━ Quality Metrics ━━━\n\n"
        f"On-Road Ratio: {on_road_ratio*100:.1f}%\n"
        f"Path Efficiency: {np.mean(efficiencies)*100:.1f}%\n"
        f"Mean Speed: {mean_speed:.2f} px/step\n\n"
        f"━━━ Physics Parameters ━━━\n\n"
        f"V₀ = 1.5, σ = 0.05\n"
        f"Momentum = 0.85\n"
        f"dt = 1.0"
    )
    
    ax4.text(0.1, 0.95, stats_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='left',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='#F8F9FA', edgecolor='#DEE2E6'))
    
    plt.suptitle('Phase 3: Langevin Dynamics Trajectory Simulation', fontsize=14, fontweight='bold', y=1.02)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    plt.close()
    return fig


# ================== 技术路线总结图 ==================
def visualize_methodology(output_path: Path = None):
    """
    生成技术路线方法论总结图
    """
    fig = plt.figure(figsize=(16, 10))
    
    # 创建 2x2 网格
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.2)
    
    # 加载数据
    mask_path = PROJECT_ROOT / "data" / "processed" / "walkable_mask.npy"
    density_path = PROJECT_ROOT / "data" / "processed" / "target_density.npy"
    potential_path = PROJECT_ROOT / "data" / "processed" / "potential_field.npy"
    traj_path = PROJECT_ROOT / "data" / "output" / "trajectories.h5"
    
    walkable_mask = np.load(mask_path) if mask_path.exists() else None
    target_density = np.load(density_path) if density_path.exists() else None
    potential_field = np.load(potential_path) if potential_path.exists() else None
    
    H, W = walkable_mask.shape if walkable_mask is not None else (100, 100)
    
    # --- Phase 1: Sink 识别 ---
    ax1 = fig.add_subplot(gs[0, 0])
    
    if walkable_mask is not None and target_density is not None:
        # 道路背景
        ax1.imshow(np.where(walkable_mask, 0.9, 0.2).T, cmap='gray', origin='lower')
        
        # Sink 热力图
        density_smooth = gaussian_filter(target_density, sigma=5)
        density_masked = np.ma.masked_where(density_smooth < 0.001, density_smooth)
        ax1.imshow(density_masked.T, cmap='YlOrRd', origin='lower', alpha=0.7)
    
    ax1.set_title('Phase 1: Sink Identification\nOD Flow → Weighted Sink Points', fontweight='bold', fontsize=11)
    ax1.set_xlabel('X (100m/px)')
    ax1.set_ylabel('Y (100m/px)')
    
    # 添加公式
    ax1.text(0.02, 0.98, r'$w_i = \sum_{j} F_{j \rightarrow i}$', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # --- Phase 2: 势能场 ---
    ax2 = fig.add_subplot(gs[0, 1])
    
    if potential_field is not None and walkable_mask is not None:
        potential_masked = np.ma.masked_where(~(walkable_mask > 0), potential_field)
        ax2.imshow(potential_masked.T, cmap='plasma', origin='lower')
        
        # 添加向量场
        nav_path = PROJECT_ROOT / "data" / "processed" / "nav_baseline.npz"
        if nav_path.exists():
            nav_data = np.load(nav_path)
            nav_y, nav_x = nav_data['nav_y'], nav_data['nav_x']
            
            step = 40
            Y, X = np.mgrid[0:H:step, 0:W:step]
            U = nav_x[::step, ::step]
            V = nav_y[::step, ::step]
            mask_sub = walkable_mask[::step, ::step] > 0
            
            ax2.quiver(X[mask_sub], Y[mask_sub], U[mask_sub], V[mask_sub],
                      color='white', scale=35, width=0.003, alpha=0.8)
    
    ax2.set_title('Phase 2: Potential Field\nWeighted Gravity Superposition', fontweight='bold', fontsize=11)
    ax2.set_xlabel('X (100m/px)')
    ax2.set_ylabel('Y (100m/px)')
    
    # 添加公式
    ax2.text(0.02, 0.98, r'$\phi(x) = \sum_i \frac{w_i}{1 + \alpha \cdot d_i(x)}$', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # --- Phase 3: 轨迹仿真 ---
    ax3 = fig.add_subplot(gs[1, 0])
    
    traj_h5 = PROJECT_ROOT / "data" / "output" / "trajectories.h5"
    if traj_h5.exists() and walkable_mask is not None:
        with h5py.File(traj_h5, 'r') as f:
            positions = f['positions'][:]
        
        # 道路背景
        ax3.imshow(np.where(walkable_mask, 0.95, 0.3).T, cmap='gray', origin='lower')
        
        # 抽样轨迹
        T, N, _ = positions.shape
        sample_indices = np.random.choice(N, min(30, N), replace=False)
        
        cmap = plt.cm.rainbow(np.linspace(0, 1, len(sample_indices)))
        for i, idx in enumerate(sample_indices):
            traj = positions[:, idx, :]
            valid = (traj[:, 0] > 0) & (traj[:, 1] > 0)
            if valid.sum() > 10:
                traj_valid = traj[valid]
                ax3.plot(traj_valid[:, 1], traj_valid[:, 0], color=cmap[i], 
                        alpha=0.6, linewidth=0.8)
                ax3.scatter(traj_valid[0, 1], traj_valid[0, 0], color=cmap[i], 
                           s=20, marker='o', zorder=5)
                ax3.scatter(traj_valid[-1, 1], traj_valid[-1, 0], color=cmap[i], 
                           s=20, marker='x', zorder=5)
    
    ax3.set_title('Phase 3: Langevin Dynamics\nMomentum + Road Constraint', fontweight='bold', fontsize=11)
    ax3.set_xlabel('X (100m/px)')
    ax3.set_ylabel('Y (100m/px)')
    ax3.set_xlim(0, W)
    ax3.set_ylim(0, H)
    
    # 添加公式
    ax3.text(0.02, 0.98, 
             r'$v_{t+1} = \beta v_t + (1-\beta)(\nabla\phi + \eta)$' + '\n' + r'$\beta = 0.85$', 
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # --- Phase 4: Diffusion Policy ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # 绘制 Diffusion Policy 示意图
    ax4.text(0.5, 0.9, 'Phase 4: Diffusion Policy', fontsize=12, fontweight='bold',
             ha='center', va='top', transform=ax4.transAxes)
    
    # 架构示意
    architecture = """
┌─────────────────────────────────────────┐
│         Conditional 1D-UNet             │
│  ┌─────────────────────────────────┐    │
│  │   Observation (History States)  │    │
│  │        pos, vel (T×4)           │    │
│  └────────────┬────────────────────┘    │
│               ↓                         │
│  ┌─────────────────────────────────┐    │
│  │   Time Embedding (Sinusoidal)   │    │
│  └────────────┬────────────────────┘    │
│               ↓                         │
│  ┌─────────────────────────────────┐    │
│  │    Encoder → Bottleneck → Decoder│   │
│  │    (ResBlocks + Skip Connections)│   │
│  └────────────┬────────────────────┘    │
│               ↓                         │
│  ┌─────────────────────────────────┐    │
│  │    Predicted Noise ε_θ          │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘

Training: L = E[||ε - ε_θ(x_t, t, c)||²]
Inference: DDPM/DDIM Denoising (100→20 steps)
"""
    
    ax4.text(0.5, 0.75, architecture, fontsize=8, family='monospace',
             ha='center', va='top', transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='#F0F0F0', edgecolor='#CCCCCC'))
    
    plt.suptitle('GeoExplicit SFM: Technical Pipeline Overview', fontsize=16, fontweight='bold', y=0.98)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    plt.close()
    return fig


# ================== 主函数 ==================
def main():
    """生成所有可视化"""
    output_dir = PROJECT_ROOT / "data" / "output" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("GeoExplicit SFM - Publication Quality Visualizations")
    print("=" * 60)
    
    # Phase 1
    print("\n[1/4] Generating Phase 1 visualization...")
    visualize_phase1(output_dir / "phase1_sink_identification.png")
    
    # Phase 2
    print("\n[2/4] Generating Phase 2 visualization...")
    visualize_phase2(output_dir / "phase2_field_computation.png")
    
    # Phase 3
    print("\n[3/4] Generating Phase 3 visualization...")
    visualize_phase3(output_dir / "phase3_trajectory_simulation.png")
    
    # Methodology
    print("\n[4/4] Generating methodology overview...")
    visualize_methodology(output_dir / "methodology_overview.png")
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
