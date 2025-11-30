"""Phase 4 数据质量分析脚本"""
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    traj = h5py.File('data/output/trajectories.h5', 'r')
    pos = np.array(traj['positions']).transpose(1, 0, 2)
    vel = np.array(traj['velocities']).transpose(1, 0, 2)

    N, T, _ = pos.shape

    # === Phase 4 训练关键：动作的分布是否"有信息量" ===
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. 速度方向分布
    ax = axes[0, 0]
    angles = np.arctan2(vel[:,:,0], vel[:,:,1]).flatten()
    speeds = np.sqrt((vel**2).sum(axis=2)).flatten()
    valid = speeds > 0.1
    ax.hist(angles[valid] * 180 / np.pi, bins=72, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Direction (degrees)')
    ax.set_ylabel('Count')
    ax.set_title('Velocity Direction Distribution')

    # 2. 加速度分布
    ax = axes[0, 1]
    acc = np.diff(vel, axis=1)
    acc_mag = np.sqrt((acc**2).sum(axis=2)).flatten()
    ax.hist(acc_mag, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(acc_mag.mean(), color='r', linestyle='--', label=f'Mean: {acc_mag.mean():.4f}')
    ax.set_xlabel('Acceleration Magnitude')
    ax.set_ylabel('Count')
    ax.set_title('Acceleration Distribution')
    ax.legend()

    # 3. 速度时序
    ax = axes[0, 2]
    sample_id = 0
    speed = np.sqrt((vel[sample_id]**2).sum(axis=1))
    ax.plot(speed[:500], 'b-', lw=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Speed')
    ax.set_title('Speed over Time (Sample Trajectory)')

    # 4. 局部速度变化
    ax = axes[1, 0]
    window = 10
    local_stds = []
    for i in range(N):
        for t in range(0, T - window, window):
            win_vel = vel[i, t:t+window]
            local_stds.append(np.std(np.sqrt((win_vel**2).sum(axis=1))))
    ax.hist(local_stds, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Speed Std in 10-step Window')
    ax.set_ylabel('Count')
    ax.set_title('Local Speed Variation')

    # 5. 方向变化率
    ax = axes[1, 1]
    dir_changes = []
    for i in range(min(100, N)):
        for t in range(1, T-1):
            v1 = vel[i, t-1]
            v2 = vel[i, t]
            mag1, mag2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if mag1 > 0.1 and mag2 > 0.1:
                cos_angle = np.dot(v1, v2) / (mag1 * mag2)
                cos_angle = np.clip(cos_angle, -1, 1)
                dir_changes.append(np.arccos(cos_angle) * 180 / np.pi)

    ax.hist(dir_changes, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(dir_changes), color='r', linestyle='--', label=f'Mean: {np.mean(dir_changes):.1f}deg')
    ax.set_xlabel('Direction Change (degrees)')
    ax.set_ylabel('Count')
    ax.set_title('Step-to-Step Direction Change')
    ax.legend()

    # 6. 起点分布
    ax = axes[1, 2]
    starts = pos[:, 0, :]
    heatmap, xedges, yedges = np.histogram2d(starts[:,1], starts[:,0], bins=50)
    ax.imshow(heatmap.T, origin='lower', cmap='hot')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Starting Position Distribution')

    plt.tight_layout()
    plt.savefig('data/output/phase4_data_quality.png', dpi=150)
    print('保存: data/output/phase4_data_quality.png')

    # 打印分析结果
    print(f'\n{"="*50}')
    print('Phase 4 数据质量评估')
    print(f'{"="*50}')
    
    print(f'\n📊 数据规模:')
    print(f'   轨迹数: {N}')
    print(f'   每轨迹步数: {T}')
    print(f'   总 (state, action) 对: {N * T:,}')
    
    print(f'\n📈 动作多样性:')
    print(f'   方向覆盖: {len(np.unique(np.round(angles[valid]*10)))} 个不同方向 (理想: >100)')
    print(f'   速度范围: [{speeds.min():.3f}, {speeds.max():.3f}]')
    print(f'   加速度范围: [{acc_mag.min():.4f}, {acc_mag.max():.4f}]')
    
    print(f'\n🔄 时序依赖:')
    print(f'   步间方向变化: mean={np.mean(dir_changes):.1f}°, std={np.std(dir_changes):.1f}°')
    print(f'   局部速度变化: mean={np.mean(local_stds):.4f}')
    
    print(f'\n{"="*50}')
    print('Phase 4 适用性判断')
    print(f'{"="*50}')
    
    issues = []
    
    # 检查1: 方向变化
    if np.mean(dir_changes) > 90:
        issues.append('⚠️ 方向变化剧烈 (mean > 90°): 轨迹震荡严重，模型难以学习平滑策略')
    else:
        print('✅ 方向变化合理 (mean < 90°)')
    
    # 检查2: 加速度
    if acc_mag.mean() < 0.01:
        issues.append('⚠️ 加速度过小: 动作空间缺乏变化，模型可能学习常数输出')
    else:
        print('✅ 加速度分布有变化')
    
    # 检查3: 数据量
    if N * T < 100000:
        issues.append('⚠️ 数据量不足: 建议至少 100K 样本用于 Diffusion Policy 训练')
    else:
        print(f'✅ 数据量充足 ({N*T:,} 样本)')
    
    # 检查4: 速度分布
    if speeds.std() < 0.1:
        issues.append('⚠️ 速度分布过于集中: 模型可能过拟合到单一速度')
    else:
        print('✅ 速度分布有多样性')
    
    if issues:
        print('\n🔴 发现的问题:')
        for issue in issues:
            print(f'   {issue}')
        print('\n💡 建议: 先解决上述问题再进行 Phase 4 训练')
    else:
        print('\n🟢 数据质量良好，可以进行 Phase 4 训练!')

if __name__ == '__main__':
    main()
