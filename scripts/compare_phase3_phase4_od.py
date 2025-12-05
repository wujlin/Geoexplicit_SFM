"""
Phase 3 vs Phase 4 OD Flow 对比分析

指标:
1. 目的地分布: 各 sink 的到达比例
2. 方向一致性: 轨迹方向与目标 sink 的一致性
3. 到达率: 成功到达 sink 附近的比例
4. 速度分布: 平均速度和方差
"""

import numpy as np
import torch
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_data():
    """加载所有需要的数据"""
    # 地图数据
    mask = np.load('data/processed/walkable_mask.npy')
    nav_data = np.load('data/processed/nav_baseline.npz')
    nav_field = np.stack([nav_data['nav_y'], nav_data['nav_x']], axis=0)
    dist_field = np.load('data/processed/distance_field.npy')
    pixel_to_sink = np.load('data/processed/pixel_to_sink.npy')
    
    # 如果 pixel_to_sink 是 1D，reshape 成 2D
    if pixel_to_sink.ndim == 1:
        pixel_to_sink = pixel_to_sink.reshape(mask.shape)
    
    # Sink 数据
    sinks_df = pd.read_csv('data/processed/sinks_phase1.csv')
    
    # Phase 3 轨迹
    with h5py.File('data/output/trajectories.h5', 'r') as f:
        phase3_pos = f['positions'][:]  # (T, N, 2)
        phase3_vel = f['velocities'][:]
        phase3_dest = f['destinations'][:]  # (T, N)
    
    return {
        'mask': mask,
        'nav_field': nav_field,
        'dist_field': dist_field,
        'pixel_to_sink': pixel_to_sink,
        'sinks_df': sinks_df,
        'phase3_pos': phase3_pos,
        'phase3_vel': phase3_vel,
        'phase3_dest': phase3_dest,
    }


def load_phase4_model():
    """加载 Phase 4 模型"""
    from phase4.model.unet1d import UNet1D
    from phase4.data.normalizer import ObsNormalizer, ActionNormalizer
    from phase4.diffusion.scheduler import DDIMScheduler
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load('data/output/phase4_checkpoints/best.pt', map_location=device, weights_only=False)
    
    model = UNet1D(obs_dim=12, act_dim=2, base_channels=128, cond_dim=64, time_dim=64).to(device)
    model.load_state_dict(ckpt['ema_state_dict'])
    model.eval()
    
    obs_norm = ObsNormalizer()
    obs_norm.load_state_dict(ckpt['obs_normalizer'])
    
    act_norm = ActionNormalizer()
    act_norm.load_state_dict(ckpt['action_normalizer'])
    
    scheduler = DDIMScheduler(num_diffusion_steps=100, num_inference_steps=20)
    
    return model, obs_norm, act_norm, scheduler, device


def simulate_phase4(data, model, obs_norm, act_norm, scheduler, device, 
                    n_agents=100, max_steps=300, guidance_scale=3.0):
    """用 Phase 4 模型生成轨迹"""
    mask = data['mask']
    nav_field = data['nav_field']
    H, W = mask.shape
    
    def get_nav(pos):
        y, x = int(np.clip(pos[0], 0, H-1)), int(np.clip(pos[1], 0, W-1))
        nav = nav_field[:, y, x]
        norm = np.linalg.norm(nav)
        return nav / norm if norm > 1e-6 else np.zeros(2)
    
    # 选择起点 (从 Phase 3 采样)
    phase3_origins = data['phase3_pos'][0, :, :]  # (N, 2)
    np.random.seed(42)
    selected_agents = np.random.choice(len(phase3_origins), n_agents, replace=False)
    start_positions = phase3_origins[selected_agents]
    
    trajectories = []
    
    for i, start_pos in enumerate(start_positions):
        pos = start_pos.copy()
        vel = get_nav(pos) * 0.8
        pos_hist = [pos.copy(), pos.copy()]
        vel_hist = [vel.copy(), vel.copy()]
        nav_hist = [get_nav(pos), get_nav(pos)]
        traj = [pos.copy()]
        
        for step in range(0, max_steps, 4):
            obs = np.concatenate([
                np.stack(pos_hist[-2:]),
                np.stack(vel_hist[-2:]),
                np.stack(nav_hist[-2:])
            ], axis=-1).astype(np.float32)
            
            obs_t = torch.tensor(obs).unsqueeze(0).to(device)
            obs_t = obs_norm.transform(obs_t)
            obs_flat = obs_t.reshape(1, -1)
            
            with torch.no_grad():
                samples = scheduler.sample_cfg(model, (1, 8, 2), obs_flat, device, guidance_scale)
                samples = act_norm.inverse_transform(samples)
            actions = samples[0].cpu().numpy()
            
            for j in range(min(4, max_steps - step)):
                vel = actions[j]
                new_pos = pos + vel
                new_pos = np.clip(new_pos, [0, 0], [H-1, W-1])
                iy, ix = int(new_pos[0]), int(new_pos[1])
                if mask[iy, ix]:
                    pos = new_pos
                nav = get_nav(pos)
                pos_hist.append(pos.copy())
                vel_hist.append(vel.copy())
                nav_hist.append(nav)
                traj.append(pos.copy())
        
        trajectories.append(np.array(traj))
        if (i + 1) % 20 == 0:
            print(f"  Generated {i+1}/{n_agents} trajectories")
    
    return trajectories, selected_agents


def analyze_trajectories(trajectories, data, name):
    """分析轨迹的各项指标"""
    dist_field = data['dist_field']
    pixel_to_sink = data['pixel_to_sink']
    H, W = data['mask'].shape
    
    metrics = {}
    
    # 1. 距离变化
    dist_changes = []
    for traj in trajectories:
        y0, x0 = int(np.clip(traj[0, 0], 0, H-1)), int(np.clip(traj[0, 1], 0, W-1))
        yf, xf = int(np.clip(traj[-1, 0], 0, H-1)), int(np.clip(traj[-1, 1], 0, W-1))
        dist_changes.append(dist_field[yf, xf] - dist_field[y0, x0])
    
    metrics['dist_change_mean'] = np.mean(dist_changes)
    metrics['dist_change_std'] = np.std(dist_changes)
    metrics['approaching_rate'] = np.mean(np.array(dist_changes) < 0)
    
    # 2. 到达率 (距离 sink < 10 px)
    arrival_count = 0
    for traj in trajectories:
        yf, xf = int(np.clip(traj[-1, 0], 0, H-1)), int(np.clip(traj[-1, 1], 0, W-1))
        if dist_field[yf, xf] < 10:
            arrival_count += 1
    metrics['arrival_rate'] = arrival_count / len(trajectories)
    
    # 3. 目的地分布 (最终位置对应的 sink)
    dest_counts = {}
    for traj in trajectories:
        yf, xf = int(np.clip(traj[-1, 0], 0, H-1)), int(np.clip(traj[-1, 1], 0, W-1))
        sink_id = pixel_to_sink[yf, xf]
        dest_counts[sink_id] = dest_counts.get(sink_id, 0) + 1
    metrics['dest_distribution'] = dest_counts
    
    # 4. 速度统计
    speeds = []
    for traj in trajectories:
        vel = np.diff(traj, axis=0)
        speed = np.linalg.norm(vel, axis=1)
        speeds.extend(speed)
    metrics['speed_mean'] = np.mean(speeds)
    metrics['speed_std'] = np.std(speeds)
    
    # 5. 方向平滑度
    angle_changes = []
    for traj in trajectories:
        vel = np.diff(traj, axis=0)
        speed = np.linalg.norm(vel, axis=1)
        valid = speed > 0.1
        if valid.sum() < 2:
            continue
        vel_valid = vel[valid]
        angles = np.arctan2(vel_valid[:, 1], vel_valid[:, 0])
        ad = np.abs(np.diff(angles))
        ad = np.minimum(ad, 2*np.pi - ad)
        angle_changes.extend(ad)
    metrics['angle_change_mean'] = np.degrees(np.mean(angle_changes))
    
    return metrics


def extract_phase3_trajectories(data, agent_indices, max_steps=300):
    """
    提取 Phase 3 对应 agent 的轨迹
    
    改进：找到每个 agent 从远处出发的轨迹段（respawn 后到下一次 respawn 前）
    """
    phase3_pos = data['phase3_pos']  # (T, N, 2)
    phase3_dest = data['phase3_dest']  # (T, N)
    dist_field = data['dist_field']
    H, W = dist_field.shape
    T_total = phase3_pos.shape[0]
    
    trajectories = []
    for agent in agent_indices:
        # 找到这个 agent 的所有 respawn 点（目的地变化的时刻）
        dest_seq = phase3_dest[:, agent]
        respawn_times = np.where(np.diff(dest_seq) != 0)[0] + 1  # respawn 发生的时刻
        
        # 添加起点和终点
        segment_starts = np.concatenate([[0], respawn_times])
        segment_ends = np.concatenate([respawn_times, [T_total]])
        
        # 找到一个从远处出发的段（起点距离 > 30）
        best_traj = None
        best_start_dist = 0
        
        for start, end in zip(segment_starts, segment_ends):
            if end - start < 50:  # 太短的段跳过
                continue
            
            y0 = int(np.clip(phase3_pos[start, agent, 0], 0, H-1))
            x0 = int(np.clip(phase3_pos[start, agent, 1], 0, W-1))
            start_dist = dist_field[y0, x0]
            
            if start_dist > best_start_dist:
                best_start_dist = start_dist
                seg_len = min(end - start, max_steps)
                best_traj = phase3_pos[start:start+seg_len, agent, :]
        
        # 如果没找到好的段，用原始方式
        if best_traj is None:
            best_traj = phase3_pos[:max_steps, agent, :]
        
        trajectories.append(best_traj)
    
    return trajectories


def plot_comparison(phase3_metrics, phase4_metrics, output_path):
    """绘制对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 距离变化对比
    ax = axes[0, 0]
    methods = ['Phase 3\n(Simulation)', 'Phase 4\n(Diffusion)']
    dist_means = [phase3_metrics['dist_change_mean'], phase4_metrics['dist_change_mean']]
    dist_stds = [phase3_metrics['dist_change_std'], phase4_metrics['dist_change_std']]
    bars = ax.bar(methods, dist_means, yerr=dist_stds, capsize=5, color=['#3498db', '#e74c3c'])
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Distance Change (px)')
    ax.set_title('Distance to Sink Change\n(negative = approaching)')
    for bar, val in zip(bars, dist_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{val:.1f}', ha='center')
    
    # 2. 靠近率和到达率
    ax = axes[0, 1]
    x = np.arange(2)
    width = 0.35
    approaching = [phase3_metrics['approaching_rate']*100, phase4_metrics['approaching_rate']*100]
    arrival = [phase3_metrics['arrival_rate']*100, phase4_metrics['arrival_rate']*100]
    ax.bar(x - width/2, approaching, width, label='Approaching Rate', color='#2ecc71')
    ax.bar(x + width/2, arrival, width, label='Arrival Rate', color='#9b59b6')
    ax.set_ylabel('Rate (%)')
    ax.set_title('Approaching & Arrival Rates')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 105)
    
    # 3. 速度分布
    ax = axes[0, 2]
    speed_means = [phase3_metrics['speed_mean'], phase4_metrics['speed_mean']]
    speed_stds = [phase3_metrics['speed_std'], phase4_metrics['speed_std']]
    bars = ax.bar(methods, speed_means, yerr=speed_stds, capsize=5, color=['#3498db', '#e74c3c'])
    ax.set_ylabel('Speed (px/step)')
    ax.set_title('Average Speed')
    for bar, val in zip(bars, speed_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.2f}', ha='center')
    
    # 4. 方向平滑度
    ax = axes[1, 0]
    angle_changes = [phase3_metrics['angle_change_mean'], phase4_metrics['angle_change_mean']]
    bars = ax.bar(methods, angle_changes, color=['#3498db', '#e74c3c'])
    ax.set_ylabel('Angle Change (deg/step)')
    ax.set_title('Direction Smoothness\n(lower = smoother)')
    for bar, val in zip(bars, angle_changes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}°', ha='center')
    
    # 5. 目的地分布 (Top 10 sinks)
    ax = axes[1, 1]
    p3_dest = phase3_metrics['dest_distribution']
    p4_dest = phase4_metrics['dest_distribution']
    all_sinks = set(p3_dest.keys()) | set(p4_dest.keys())
    top_sinks = sorted(all_sinks, key=lambda x: p3_dest.get(x, 0) + p4_dest.get(x, 0), reverse=True)[:10]
    
    x = np.arange(len(top_sinks))
    width = 0.35
    p3_counts = [p3_dest.get(s, 0) for s in top_sinks]
    p4_counts = [p4_dest.get(s, 0) for s in top_sinks]
    ax.bar(x - width/2, p3_counts, width, label='Phase 3', color='#3498db')
    ax.bar(x + width/2, p4_counts, width, label='Phase 4', color='#e74c3c')
    ax.set_xlabel('Sink ID')
    ax.set_ylabel('Count')
    ax.set_title('Destination Distribution (Top 10)')
    ax.set_xticks(x)
    ax.set_xticklabels(top_sinks)
    ax.legend()
    
    # 6. 总结表格
    ax = axes[1, 2]
    ax.axis('off')
    table_data = [
        ['Metric', 'Phase 3', 'Phase 4'],
        ['Dist Change', f'{phase3_metrics["dist_change_mean"]:.1f}', f'{phase4_metrics["dist_change_mean"]:.1f}'],
        ['Approaching %', f'{phase3_metrics["approaching_rate"]*100:.1f}%', f'{phase4_metrics["approaching_rate"]*100:.1f}%'],
        ['Arrival %', f'{phase3_metrics["arrival_rate"]*100:.1f}%', f'{phase4_metrics["arrival_rate"]*100:.1f}%'],
        ['Speed', f'{phase3_metrics["speed_mean"]:.2f}', f'{phase4_metrics["speed_mean"]:.2f}'],
        ['Smoothness', f'{phase3_metrics["angle_change_mean"]:.1f}°', f'{phase4_metrics["angle_change_mean"]:.1f}°'],
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    # 设置表头样式
    for i in range(3):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    ax.set_title('Summary Comparison', fontsize=12, fontweight='bold', y=0.95)
    
    plt.suptitle('Phase 3 (Simulation) vs Phase 4 (Diffusion Policy) Comparison', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Phase 3 vs Phase 4 OD Flow Comparison")
    print("=" * 60)
    
    # 加载数据
    print("\n[1/5] Loading data...")
    data = load_data()
    print(f"  Map size: {data['mask'].shape}")
    print(f"  Phase 3 trajectories: {data['phase3_pos'].shape}")
    
    # 加载 Phase 4 模型
    print("\n[2/5] Loading Phase 4 model...")
    model, obs_norm, act_norm, scheduler, device = load_phase4_model()
    print(f"  Device: {device}")
    
    # 生成 Phase 4 轨迹
    print("\n[3/5] Generating Phase 4 trajectories (CFG=3.0)...")
    phase4_trajs, agent_indices = simulate_phase4(
        data, model, obs_norm, act_norm, scheduler, device,
        n_agents=100, max_steps=300, guidance_scale=3.0
    )
    
    # 提取对应的 Phase 3 轨迹
    print("\n[4/5] Extracting Phase 3 trajectories...")
    phase3_trajs = extract_phase3_trajectories(data, agent_indices, max_steps=300)
    
    # 分析
    print("\n[5/5] Analyzing trajectories...")
    phase3_metrics = analyze_trajectories(phase3_trajs, data, "Phase 3")
    phase4_metrics = analyze_trajectories(phase4_trajs, data, "Phase 4")
    
    # 打印结果
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    
    print("\n  Phase 3 (Simulation):")
    print(f"    Distance change: {phase3_metrics['dist_change_mean']:.1f} ± {phase3_metrics['dist_change_std']:.1f} px")
    print(f"    Approaching rate: {phase3_metrics['approaching_rate']*100:.1f}%")
    print(f"    Arrival rate: {phase3_metrics['arrival_rate']*100:.1f}%")
    print(f"    Speed: {phase3_metrics['speed_mean']:.2f} ± {phase3_metrics['speed_std']:.2f}")
    print(f"    Direction smoothness: {phase3_metrics['angle_change_mean']:.1f}°/step")
    
    print("\n  Phase 4 (Diffusion Policy):")
    print(f"    Distance change: {phase4_metrics['dist_change_mean']:.1f} ± {phase4_metrics['dist_change_std']:.1f} px")
    print(f"    Approaching rate: {phase4_metrics['approaching_rate']*100:.1f}%")
    print(f"    Arrival rate: {phase4_metrics['arrival_rate']*100:.1f}%")
    print(f"    Speed: {phase4_metrics['speed_mean']:.2f} ± {phase4_metrics['speed_std']:.2f}")
    print(f"    Direction smoothness: {phase4_metrics['angle_change_mean']:.1f}°/step")
    
    # 绘制对比图
    output_path = Path('data/output/phase4_validation/phase3_vs_phase4_comparison.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(phase3_metrics, phase4_metrics, output_path)
    
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
