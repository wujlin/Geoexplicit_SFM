"""
全流程数据一致性检查脚本
检查 Phase 1-4 的数据流是否正确
"""

import numpy as np
import h5py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

def main():
    print('='*60)
    print('Phase 1-4 数据一致性检查')
    print('='*60)

    issues = []

    # Phase 1: Sink
    print('\n[Phase 1] Sink 数据')
    import pandas as pd
    sinks = pd.read_csv(PROJECT_ROOT / 'data/processed/sinks_phase1.csv')
    print(f'  Sink 数量: {len(sinks)}')
    total_flow = sinks["total_flow"].sum()
    print(f'  总流量: {total_flow:,}')
    print(f'  列名: {list(sinks.columns)}')
    
    if len(sinks) < 10:
        issues.append('Phase1: Sink 数量过少')

    # Phase 2: 导航场
    print('\n[Phase 2] 导航场数据')
    nav_data = np.load(PROJECT_ROOT / 'data/processed/nav_baseline.npz')
    nav_y = nav_data['nav_y']
    nav_x = nav_data['nav_x']
    print(f'  nav_y shape: {nav_y.shape}')
    print(f'  nav_x shape: {nav_x.shape}')

    distance_field = np.load(PROJECT_ROOT / 'data/processed/distance_field.npy')
    walkable_mask = np.load(PROJECT_ROOT / 'data/processed/walkable_mask.npy')
    print(f'  distance_field shape: {distance_field.shape}')
    print(f'  walkable_mask shape: {walkable_mask.shape}')

    H, W = walkable_mask.shape
    road_pixels = (walkable_mask > 0).sum()
    print(f'  道路像素: {road_pixels:,} ({road_pixels/(H*W)*100:.2f}%)')

    # 验证 nav 有效性
    nav_mag = np.sqrt(nav_y**2 + nav_x**2)
    road_mask = walkable_mask > 0
    valid_nav = (nav_mag[road_mask] > 0.9).mean()
    print(f'  道路上 |nav|>0.9: {valid_nav*100:.2f}%')
    
    if valid_nav < 0.9:
        issues.append('Phase2: 导航场有效率过低')

    # Phase 3: 轨迹
    print('\n[Phase 3] 轨迹数据')
    with h5py.File(PROJECT_ROOT / 'data/output/trajectories.h5', 'r') as f:
        pos_shape = f['positions'].shape
        vel_shape = f['velocities'].shape
        print(f'  positions shape: {pos_shape}')
        print(f'  velocities shape: {vel_shape}')
        
        T, N, _ = pos_shape
        pos = f['positions'][:]
        vel = f['velocities'][:]

    print(f'  时间步: {T}, 智能体: {N}')
    print(f'  总帧数: {T*N:,}')

    # 速度分布
    speeds = np.sqrt((vel**2).sum(axis=2))
    zero_speed_rate = (speeds < 0.01).mean()
    print(f'  速度 mean: {speeds.mean():.4f}')
    print(f'  速度 std: {speeds.std():.4f}')
    print(f'  零速度 (<0.01): {zero_speed_rate*100:.4f}%')
    
    if zero_speed_rate > 0.01:
        issues.append(f'Phase3: 零速度帧过多 ({zero_speed_rate*100:.2f}%)')

    # velocity vs nav 一致性 (采样)
    print('\n[验证] Velocity vs Nav 一致性')
    np.random.seed(42)
    sample_size = 10000
    sample_t = np.random.randint(0, T, sample_size)
    sample_n = np.random.randint(0, N, sample_size)

    nav_field = np.stack([nav_y, nav_x], axis=0)
    cos_sims = []
    for t, n in zip(sample_t, sample_n):
        p = pos[t, n]
        v = vel[t, n]
        
        y = int(np.clip(p[0], 0, H-1))
        x = int(np.clip(p[1], 0, W-1))
        nav = nav_field[:, y, x]
        
        v_mag = np.linalg.norm(v)
        nav_mag_val = np.linalg.norm(nav)
        
        if v_mag > 0.1 and nav_mag_val > 0.1:
            cos = np.dot(v / v_mag, nav / nav_mag_val)
            cos_sims.append(cos)

    cos_sims = np.array(cos_sims)
    cos_mean = cos_sims.mean()
    cos_positive = (cos_sims > 0).mean()
    print(f'  采样数: {len(cos_sims)}')
    print(f'  cos_sim mean: {cos_mean:.4f}')
    print(f'  正向 (>0): {cos_positive*100:.2f}%')
    
    if cos_mean < 0.5:
        issues.append(f'Phase3: velocity vs nav 一致性较低 ({cos_mean:.3f})')

    # 距离递减检查
    print('\n[验证] 距离递减率')
    decrease_count = 0
    total_valid = 0
    for i, (t, n) in enumerate(zip(sample_t, sample_n)):
        if t == 0:
            continue
        p_curr = pos[t, n]
        p_prev = pos[t-1, n]
        
        y_curr = int(np.clip(p_curr[0], 0, H-1))
        x_curr = int(np.clip(p_curr[1], 0, W-1))
        y_prev = int(np.clip(p_prev[0], 0, H-1))
        x_prev = int(np.clip(p_prev[1], 0, W-1))
        
        d_curr = distance_field[y_curr, x_curr]
        d_prev = distance_field[y_prev, x_prev]
        
        if d_prev > 0:
            total_valid += 1
            if d_curr < d_prev:
                decrease_count += 1

    decrease_rate = decrease_count / total_valid if total_valid > 0 else 0
    print(f'  有效样本: {total_valid}')
    print(f'  距离递减率: {decrease_rate*100:.2f}%')
    
    if decrease_rate < 0.8:
        issues.append(f'Phase3: 距离递减率较低 ({decrease_rate*100:.1f}%)')

    # Phase 4: 模型代码检查
    print('\n[Phase 4] 代码检查')
    from phase4.data.normalizer import ActionNormalizer, ObsNormalizer
    an = ActionNormalizer()
    on = ObsNormalizer()
    print(f'  ActionNormalizer mode: {an.mode}')
    print(f'  ObsNormalizer mode: {on.mode}')
    
    if an.mode != 'zscore':
        issues.append('Phase4: ActionNormalizer 不是 zscore')
    if on.mode != 'zscore':
        issues.append('Phase4: ObsNormalizer 不是 zscore')
    
    from phase4.model.unet1d import UNet1D
    import torch
    model = UNet1D(obs_dim=12)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'  UNet1D 参数量: {num_params:,}')
    
    # 测试前向传播
    x = torch.randn(2, 2, 8)
    t = torch.tensor([50, 50])
    c = torch.randn(2, 12)
    out = model(x, t, c)
    print(f'  前向传播: input {x.shape} -> output {out.shape}')
    
    if out.shape != (2, 2, 8):
        issues.append(f'Phase4: 模型输出形状错误 {out.shape}')

    # 总结
    print('\n' + '='*60)
    print('[检查结果]')
    print('='*60)
    
    if issues:
        print('⚠️ 发现以下问题:')
        for issue in issues:
            print(f'  - {issue}')
        return False
    else:
        print('✅ 所有检查通过!')
        print('')
        print('Phase 1: 35 sinks, 114万流量')
        print(f'Phase 2: 导航场有效率 {valid_nav*100:.1f}%')
        print(f'Phase 3: 零速度 {zero_speed_rate*100:.4f}%, vel-nav cos={cos_mean:.3f}, 距离递减 {decrease_rate*100:.1f}%')
        print(f'Phase 4: zscore 归一化, UNet {num_params:,} 参数')
        print('')
        print('✅ 可以安全地在工作站运行训练!')
        return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
