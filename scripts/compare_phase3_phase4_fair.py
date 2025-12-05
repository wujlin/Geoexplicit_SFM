"""
Phase 3 vs Phase 4 公平对比评估

关键修正：
1. 使用相同的起点和目标 sink
2. Phase 4 推理时用目标 sink 的导航场（而非全局导航场）
3. 评估用目标 sink 的距离场（而非最近 sink 的距离场）

运行方式:
    python scripts/compare_phase3_phase4_fair.py [--n_samples 100] [--guidance 3.0]

前置条件:
    1. 先运行 scripts/filter_eval_samples.py 生成有效评估样本
    2. 需要 GPU 环境 (torch + cuda)
"""
import numpy as np
import torch
import h5py
from pathlib import Path
import sys
import argparse
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_data():
    """加载所有需要的数据"""
    print("[1/5] Loading data...")
    
    mask = np.load('data/processed/walkable_mask.npy')
    H, W = mask.shape
    print(f"  Map size: {H} x {W}")
    
    # 加载每个 sink 的导航场和距离场
    nav_fields_dir = Path('data/processed/nav_fields')
    nav_fields = {}
    dist_fields = {}
    
    for i in range(35):
        path = nav_fields_dir / f'nav_field_{i:03d}.npz'
        if path.exists():
            data = np.load(path)
            nav_fields[i] = np.stack([data['nav_y'], data['nav_x']], axis=0)  # (2, H, W)
            dist_fields[i] = data['distance_field']
    
    print(f"  Loaded {len(nav_fields)} sink navigation/distance fields")
    
    # 加载 Phase 3 轨迹
    with h5py.File('data/output/trajectories.h5', 'r') as f:
        phase3_pos = f['positions'][:]
        phase3_dest = f['destinations'][:]
    
    print(f"  Phase 3 trajectories: {phase3_pos.shape}")
    
    # 加载有效评估样本
    valid_trips = np.load('data/output/valid_eval_trips.npy', allow_pickle=True)
    print(f"  Valid evaluation trips: {len(valid_trips)}")
    
    return {
        'mask': mask,
        'nav_fields': nav_fields,
        'dist_fields': dist_fields,
        'phase3_pos': phase3_pos,
        'phase3_dest': phase3_dest,
        'valid_trips': valid_trips,
        'H': H, 'W': W
    }


def load_phase4_model():
    """加载 Phase 4 模型"""
    print("\n[2/5] Loading Phase 4 model...")
    
    from phase4.model.unet1d import UNet1D
    from phase4.data.normalizer import ObsNormalizer, ActionNormalizer
    from phase4.diffusion.scheduler import DDIMScheduler
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
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


def simulate_phase4_with_destination(data, model, obs_norm, act_norm, scheduler, device,
                                      start_pos, target_sink, max_steps=300, guidance_scale=3.0):
    """
    用 Phase 4 模型生成单条轨迹
    
    关键：使用目标 sink 的导航场
    """
    mask = data['mask']
    nav_field = data['nav_fields'][target_sink]  # 使用目标 sink 的导航场
    H, W = data['H'], data['W']
    
    def get_nav(pos):
        y, x = int(np.clip(pos[0], 0, H-1)), int(np.clip(pos[1], 0, W-1))
        nav = nav_field[:, y, x]
        norm = np.linalg.norm(nav)
        return nav / norm if norm > 1e-6 else np.zeros(2)
    
    # 初始化
    pos = start_pos.copy()
    vel = get_nav(pos) * 0.8  # 初始速度沿导航方向
    
    pos_hist = [pos.copy(), pos.copy()]
    vel_hist = [vel.copy(), vel.copy()]
    nav_hist = [get_nav(pos), get_nav(pos)]
    
    traj = [pos.copy()]
    
    for step in range(0, max_steps, 4):  # 每次预测 8 步，执行 4 步
        # 构建观测
        obs = np.concatenate([
            np.stack(pos_hist[-2:]),
            np.stack(vel_hist[-2:]),
            np.stack(nav_hist[-2:])
        ], axis=-1).astype(np.float32)  # (2, 6)
        
        obs_t = torch.tensor(obs).unsqueeze(0).to(device)
        obs_t = obs_norm.transform(obs_t)
        obs_flat = obs_t.reshape(1, -1)
        
        with torch.no_grad():
            samples = scheduler.sample_cfg(model, (1, 8, 2), obs_flat, device, guidance_scale)
            samples = act_norm.inverse_transform(samples)
        
        actions = samples[0].cpu().numpy()  # (8, 2)
        
        # 执行 4 步
        for j in range(min(4, max_steps - step)):
            vel = actions[j]
            new_pos = pos + vel
            new_pos = np.clip(new_pos, [0, 0], [H-1, W-1])
            
            # 道路约束
            iy, ix = int(new_pos[0]), int(new_pos[1])
            if mask[iy, ix]:
                pos = new_pos
            
            nav = get_nav(pos)
            pos_hist.append(pos.copy())
            vel_hist.append(vel.copy())
            nav_hist.append(nav)
            traj.append(pos.copy())
    
    return np.array(traj)


def evaluate_trips(data, model, obs_norm, act_norm, scheduler, device, 
                   n_samples=100, max_steps=300, guidance_scale=3.0):
    """评估 Phase 3 和 Phase 4"""
    
    print(f"\n[3/5] Evaluating {n_samples} trips...")
    
    valid_trips = data['valid_trips']
    phase3_pos = data['phase3_pos']
    dist_fields = data['dist_fields']
    H, W = data['H'], data['W']
    
    # 随机采样
    np.random.seed(42)
    sample_indices = np.random.choice(len(valid_trips), min(n_samples, len(valid_trips)), replace=False)
    
    phase3_results = []
    phase4_results = []
    
    for idx, trip_idx in enumerate(sample_indices):
        trip = valid_trips[trip_idx]
        agent = trip['agent']
        t_start = trip['t_start']
        target = trip['target']
        start_dist = trip['start_dist']
        
        # Phase 3: 直接从数据读取
        t_end = min(trip['t_end'], t_start + max_steps)
        phase3_traj = phase3_pos[t_start:t_end, agent, :]
        
        # Phase 3 终点距离
        yf, xf = phase3_traj[-1]
        yf, xf = int(np.clip(yf, 0, H-1)), int(np.clip(xf, 0, W-1))
        phase3_end_dist = dist_fields[target][yf, xf]
        
        # Phase 4: 从相同起点出发
        start_pos = phase3_pos[t_start, agent, :].copy()
        phase4_traj = simulate_phase4_with_destination(
            data, model, obs_norm, act_norm, scheduler, device,
            start_pos, target, max_steps, guidance_scale
        )
        
        # Phase 4 终点距离
        yf, xf = phase4_traj[-1]
        yf, xf = int(np.clip(yf, 0, H-1)), int(np.clip(xf, 0, W-1))
        phase4_end_dist = dist_fields[target][yf, xf]
        
        phase3_results.append({
            'start_dist': start_dist,
            'end_dist': phase3_end_dist,
            'change': phase3_end_dist - start_dist,
            'traj': phase3_traj,
            'target': target
        })
        
        phase4_results.append({
            'start_dist': start_dist,
            'end_dist': phase4_end_dist,
            'change': phase4_end_dist - start_dist,
            'traj': phase4_traj,
            'target': target
        })
        
        if (idx + 1) % 20 == 0:
            print(f"  Evaluated {idx+1}/{len(sample_indices)} trips")
    
    return phase3_results, phase4_results


def compute_metrics(results, name):
    """计算评估指标"""
    start_dists = [r['start_dist'] for r in results]
    end_dists = [r['end_dist'] for r in results]
    changes = [r['change'] for r in results]
    
    # 速度和平滑度
    speeds = []
    angle_changes = []
    for r in results:
        traj = r['traj']
        vel = np.diff(traj, axis=0)
        speed = np.linalg.norm(vel, axis=1)
        speeds.extend(speed[speed > 0.01])
        
        # 方向变化
        valid = speed > 0.1
        if valid.sum() >= 2:
            vel_valid = vel[valid]
            angles = np.arctan2(vel_valid[:, 1], vel_valid[:, 0])
            ad = np.abs(np.diff(angles))
            ad = np.minimum(ad, 2*np.pi - ad)
            angle_changes.extend(ad)
    
    metrics = {
        'name': name,
        'start_dist': np.mean(start_dists),
        'end_dist': np.mean(end_dists),
        'end_dist_std': np.std(end_dists),
        'change': np.mean(changes),
        'change_std': np.std(changes),
        'approaching_rate': np.mean(np.array(changes) < 0) * 100,
        'arrival_rate': np.mean(np.array(end_dists) < 10) * 100,
        'speed': np.mean(speeds),
        'speed_std': np.std(speeds),
        'smoothness': np.degrees(np.mean(angle_changes)) if angle_changes else 0
    }
    
    return metrics


def print_comparison(phase3_metrics, phase4_metrics):
    """打印对比结果"""
    print("\n" + "="*70)
    print("Results (using target sink distance):")
    print("="*70)
    
    print(f"\n  {'Metric':<25} {'Phase 3':>15} {'Phase 4':>15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")
    
    metrics_to_show = [
        ('Start Distance', 'start_dist', '{:.1f}'),
        ('End Distance', 'end_dist', '{:.1f} ± {:.1f}'),
        ('Distance Change', 'change', '{:.1f} ± {:.1f}'),
        ('Approaching Rate', 'approaching_rate', '{:.1f}%'),
        ('Arrival Rate (<10px)', 'arrival_rate', '{:.1f}%'),
        ('Speed', 'speed', '{:.2f} ± {:.2f}'),
        ('Smoothness', 'smoothness', '{:.1f}°/step'),
    ]
    
    for label, key, fmt in metrics_to_show:
        if '±' in fmt:
            if key + '_std' in phase3_metrics:
                p3_val = fmt.format(phase3_metrics[key], phase3_metrics[key + '_std'])
                p4_val = fmt.format(phase4_metrics[key], phase4_metrics[key + '_std'])
            else:
                p3_val = fmt.format(phase3_metrics[key], phase3_metrics.get(key + '_std', 0))
                p4_val = fmt.format(phase4_metrics[key], phase4_metrics.get(key + '_std', 0))
        else:
            p3_val = fmt.format(phase3_metrics[key])
            p4_val = fmt.format(phase4_metrics[key])
        
        print(f"  {label:<25} {p3_val:>15} {p4_val:>15}")


def plot_comparison(phase3_results, phase4_results, phase3_metrics, phase4_metrics, output_path):
    """绘制对比图"""
    print(f"\n[5/5] Saving comparison plot...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Phase 3 (Simulation) vs Phase 4 (Diffusion Policy) - Fair Comparison', fontsize=14)
    
    # 1. 距离变化
    ax = axes[0, 0]
    methods = ['Phase 3', 'Phase 4']
    changes = [phase3_metrics['change'], phase4_metrics['change']]
    stds = [phase3_metrics['change_std'], phase4_metrics['change_std']]
    colors = ['#3498db', '#e74c3c']
    bars = ax.bar(methods, changes, yerr=stds, capsize=5, color=colors)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Distance Change (px)')
    ax.set_title('Distance to Target Change\n(negative = approaching)')
    for bar, val in zip(bars, changes):
        y_pos = val + (5 if val >= 0 else -15)
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.1f}', ha='center')
    
    # 2. 靠近率和到达率
    ax = axes[0, 1]
    x = np.arange(2)
    width = 0.35
    approaching = [phase3_metrics['approaching_rate'], phase4_metrics['approaching_rate']]
    arrival = [phase3_metrics['arrival_rate'], phase4_metrics['arrival_rate']]
    ax.bar(x - width/2, approaching, width, label='Approaching Rate', color='#2ecc71')
    ax.bar(x + width/2, arrival, width, label='Arrival Rate', color='#9b59b6')
    ax.set_ylabel('Rate (%)')
    ax.set_title('Approaching & Arrival Rates')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 105)
    
    # 3. 速度
    ax = axes[0, 2]
    speeds = [phase3_metrics['speed'], phase4_metrics['speed']]
    speed_stds = [phase3_metrics['speed_std'], phase4_metrics['speed_std']]
    ax.bar(methods, speeds, yerr=speed_stds, capsize=5, color=colors)
    ax.set_ylabel('Speed (px/step)')
    ax.set_title('Average Speed')
    for i, (m, s) in enumerate(zip(methods, speeds)):
        ax.text(i, s + 0.1, f'{s:.2f}', ha='center')
    
    # 4. 平滑度
    ax = axes[1, 0]
    smoothness = [phase3_metrics['smoothness'], phase4_metrics['smoothness']]
    ax.bar(methods, smoothness, color=colors)
    ax.set_ylabel('Angle Change (deg/step)')
    ax.set_title('Direction Smoothness\n(lower = smoother)')
    for i, (m, s) in enumerate(zip(methods, smoothness)):
        ax.text(i, s + 1, f'{s:.1f}°', ha='center')
    
    # 5. 终点距离分布
    ax = axes[1, 1]
    p3_end = [r['end_dist'] for r in phase3_results]
    p4_end = [r['end_dist'] for r in phase4_results]
    ax.hist(p3_end, bins=30, alpha=0.6, label='Phase 3', color='#3498db')
    ax.hist(p4_end, bins=30, alpha=0.6, label='Phase 4', color='#e74c3c')
    ax.axvline(x=10, color='green', linestyle='--', label='Arrival threshold')
    ax.set_xlabel('End Distance to Target (px)')
    ax.set_ylabel('Count')
    ax.set_title('End Distance Distribution')
    ax.legend()
    
    # 6. 汇总表格
    ax = axes[1, 2]
    ax.axis('off')
    table_data = [
        ['Metric', 'Phase 3', 'Phase 4'],
        ['Dist Change', f"{phase3_metrics['change']:.1f}", f"{phase4_metrics['change']:.1f}"],
        ['Approaching %', f"{phase3_metrics['approaching_rate']:.1f}%", f"{phase4_metrics['approaching_rate']:.1f}%"],
        ['Arrival %', f"{phase3_metrics['arrival_rate']:.1f}%", f"{phase4_metrics['arrival_rate']:.1f}%"],
        ['Speed', f"{phase3_metrics['speed']:.2f}", f"{phase4_metrics['speed']:.2f}"],
        ['Smoothness', f"{phase3_metrics['smoothness']:.1f}°", f"{phase4_metrics['smoothness']:.1f}°"],
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    # 设置表头样式
    for j in range(3):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    ax.set_title('Summary Comparison', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Phase 3 vs Phase 4 Fair Comparison')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples to evaluate')
    parser.add_argument('--guidance', type=float, default=3.0, help='CFG guidance scale')
    parser.add_argument('--max_steps', type=int, default=300, help='Max simulation steps')
    args = parser.parse_args()
    
    # 加载数据
    data = load_data()
    
    # 加载模型
    model, obs_norm, act_norm, scheduler, device = load_phase4_model()
    
    # 评估
    phase3_results, phase4_results = evaluate_trips(
        data, model, obs_norm, act_norm, scheduler, device,
        n_samples=args.n_samples,
        max_steps=args.max_steps,
        guidance_scale=args.guidance
    )
    
    # 计算指标
    print("\n[4/5] Computing metrics...")
    phase3_metrics = compute_metrics(phase3_results, 'Phase 3')
    phase4_metrics = compute_metrics(phase4_results, 'Phase 4')
    
    # 打印对比
    print_comparison(phase3_metrics, phase4_metrics)
    
    # 绘图
    output_path = 'data/output/phase4_validation/phase3_vs_phase4_fair_comparison.png'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(phase3_results, phase4_results, phase3_metrics, phase4_metrics, output_path)
    
    print("\nDone!")


if __name__ == '__main__':
    main()