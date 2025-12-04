"""
数据一致性验证脚本

参照 CODE_DATA_STRUCTURE.md 进行系统性验证
"""

import sys
import json
from pathlib import Path
import numpy as np
import h5py

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def verify_phase3_data():
    """验证 Phase 3 轨迹数据一致性"""
    print("=" * 60)
    print("验证 1: Phase 3 轨迹数据一致性")
    print("=" * 60)
    print()
    
    h5_path = PROJECT_ROOT / "data" / "output" / "trajectories.h5"
    
    with h5py.File(h5_path, "r") as f:
        # 参照文档: 格式是 (T, N, 2) [y, x]
        pos = f["positions"][:]    # (T, N, 2)
        vel = f["velocities"][:]   # (T, N, 2)
        dest = f["destinations"][:]
        
        T, N, _ = pos.shape
        print(f"数据形状: T={T}, N={N}")
        print(f"  positions: {pos.shape}, dtype={pos.dtype}")
        print(f"  velocities: {vel.shape}, dtype={vel.dtype}")
        print(f"  destinations: {dest.shape}, dtype={dest.dtype}")
        print()
        
        # 核心检查 1: vel[t] 应该等于 pos[t+1] - pos[t]
        print("核心检查 1: vel[t] == pos[t+1] - pos[t]")
        print("-" * 40)
        
        np.random.seed(42)
        test_cases = [(np.random.randint(0, T - 1), np.random.randint(0, N)) for _ in range(1000)]
        
        match_count = 0
        errors = []
        for t, agent in test_cases:
            actual_disp = pos[t + 1, agent] - pos[t, agent]
            recorded_vel = vel[t, agent]
            
            diff = np.linalg.norm(actual_disp - recorded_vel)
            if diff < 0.01:
                match_count += 1
            else:
                errors.append({
                    "t": t, "agent": agent,
                    "actual_disp": actual_disp.tolist(),
                    "recorded_vel": recorded_vel.tolist(),
                    "diff": float(diff)
                })
        
        print(f"  一致: {match_count}/1000 ({match_count/10:.1f}%)")
        if errors:
            print(f"  ❌ 不一致样本: {len(errors)}")
            for e in errors[:3]:
                print(f"    t={e['t']}, agent={e['agent']}: diff={e['diff']:.4f}")
        else:
            print("  ✅ 所有采样点一致")
        print()
        
        # 核心检查 2: 坐标范围
        print("核心检查 2: 坐标范围")
        print("-" * 40)
        valid_pos = pos[pos != 0].reshape(-1, 2)
        print(f"  pos[0] (y): [{valid_pos[:, 0].min():.1f}, {valid_pos[:, 0].max():.1f}]")
        print(f"  pos[1] (x): [{valid_pos[:, 1].min():.1f}, {valid_pos[:, 1].max():.1f}]")
        
        valid_vel = vel[np.linalg.norm(vel, axis=-1) > 0.01]
        print(f"  vel[0] (vy): [{valid_vel[:, 0].min():.2f}, {valid_vel[:, 0].max():.2f}]")
        print(f"  vel[1] (vx): [{valid_vel[:, 1].min():.2f}, {valid_vel[:, 1].max():.2f}]")
        print()
        
        return match_count == 1000


def verify_nav_fields():
    """验证导航场数据"""
    print("=" * 60)
    print("验证 2: 导航场数据")
    print("=" * 60)
    print()
    
    nav_fields_dir = PROJECT_ROOT / "data" / "processed" / "nav_fields"
    
    # 加载索引
    with open(nav_fields_dir / "nav_fields_index.json") as f:
        index = json.load(f)
    
    print(f"导航场数量: {index['num_sinks']}")
    print(f"Sink IDs: {index['sink_ids'][:5]}... (共 {len(index['sink_ids'])} 个)")
    print()
    
    # 检查单个导航场
    sink_id = index["sink_ids"][0]
    data = np.load(nav_fields_dir / f"nav_field_{sink_id:03d}.npz")
    
    print(f"单个导航场 (sink {sink_id}):")
    print(f"  nav_y: {data['nav_y'].shape}, dtype={data['nav_y'].dtype}")
    print(f"  nav_x: {data['nav_x'].shape}, dtype={data['nav_x'].dtype}")
    
    # 参照文档: 组合为 (2, H, W) 格式
    nav_field = np.stack([data["nav_y"], data["nav_x"]], axis=0)
    print(f"  组合后: {nav_field.shape} [nav_y, nav_x]")
    
    # 检查是否为单位向量
    nav_mag = np.sqrt(nav_field[0]**2 + nav_field[1]**2)
    valid_mag = nav_mag[nav_mag > 0.01]
    print(f"  方向向量模长: mean={valid_mag.mean():.4f}, std={valid_mag.std():.4f}")
    is_unit = np.abs(valid_mag - 1.0).max() < 0.01
    print(f"  是否单位向量: {'✅ 是' if is_unit else '⚠️ 否'}")
    print()
    
    return True


def verify_phase4_dataset():
    """验证 Phase 4 数据集构建"""
    print("=" * 60)
    print("验证 3: Phase 4 数据集构建")
    print("=" * 60)
    print()
    
    import importlib.util
    
    def _import_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    phase4_root = PROJECT_ROOT / "src" / "phase4"
    dataset_mod = _import_module("dataset", phase4_root / "data" / "dataset.py")
    TrajectorySlidingWindow = dataset_mod.TrajectorySlidingWindow
    
    dataset = TrajectorySlidingWindow(
        h5_path=PROJECT_ROOT / "data" / "output" / "trajectories.h5",
        history=2,
        future=8,
        nav_fields_dir=PROJECT_ROOT / "data" / "processed" / "nav_fields",
    )
    
    print(f"数据集大小: {len(dataset):,}")
    print()
    
    # 检查单个样本
    sample = dataset[1000]
    print("单个样本 (idx=1000):")
    print(f"  obs shape: {sample['obs'].shape}")  # 应该是 (history, 6)
    print(f"  action shape: {sample['action'].shape}")  # 应该是 (future, 2)
    print(f"  agent: {sample['agent']}, t0: {sample['t0']}, dest: {sample['dest']}")
    print()
    
    # 参照文档: obs = [pos_y, pos_x, vel_y, vel_x, nav_y, nav_x]
    obs = sample["obs"].numpy()
    print("obs 内容分解 (应为 [pos_y, pos_x, vel_y, vel_x, nav_y, nav_x]):")
    print(f"  frame 0: pos=[{obs[0, 0]:.2f}, {obs[0, 1]:.2f}], vel=[{obs[0, 2]:.2f}, {obs[0, 3]:.2f}], nav=[{obs[0, 4]:.2f}, {obs[0, 5]:.2f}]")
    print(f"  frame 1: pos=[{obs[1, 0]:.2f}, {obs[1, 1]:.2f}], vel=[{obs[1, 2]:.2f}, {obs[1, 3]:.2f}], nav=[{obs[1, 4]:.2f}, {obs[1, 5]:.2f}]")
    print()
    
    # 验证 obs 中的 velocity 与 action 的关系
    # 参照文档: action = vel[t0+history : t0+history+future]
    # obs 中的 vel = vel[t0 : t0+history]
    agent, t0 = sample["agent"], sample["t0"]
    
    with h5py.File(PROJECT_ROOT / "data" / "output" / "trajectories.h5", "r") as f:
        h5_pos = f["positions"][t0:t0+2+8, agent]  # (history+future, 2)
        h5_vel = f["velocities"][t0:t0+2+8, agent]
    
    print("与 HDF5 原始数据对比:")
    print(f"  obs pos[0] vs h5_pos[0]: {obs[0, :2]} vs {h5_pos[0]} -> {'✅' if np.allclose(obs[0, :2], h5_pos[0]) else '❌'}")
    print(f"  obs vel[0] vs h5_vel[0]: {obs[0, 2:4]} vs {h5_vel[0]} -> {'✅' if np.allclose(obs[0, 2:4], h5_vel[0]) else '❌'}")
    
    action = sample["action"].numpy()
    print(f"  action[0] vs h5_vel[2]: {action[0]} vs {h5_vel[2]} -> {'✅' if np.allclose(action[0], h5_vel[2]) else '❌'}")
    print()
    
    # 关键检查: action[0] 应该是从 obs 最后位置出发的速度
    # 即 action[0] ≈ h5_pos[3] - h5_pos[2]
    expected_action0 = h5_pos[3] - h5_pos[2]
    print("关键关系验证:")
    print(f"  action[0] 应等于 pos[t0+2+1] - pos[t0+2]")
    print(f"  action[0] = {action[0]}")
    print(f"  h5_pos[3] - h5_pos[2] = {expected_action0}")
    print(f"  -> {'✅ 一致' if np.allclose(action[0], expected_action0, atol=0.01) else '❌ 不一致'}")
    print()
    
    return True


def verify_model_inference():
    """验证模型推理流程"""
    print("=" * 60)
    print("验证 4: 模型推理流程")
    print("=" * 60)
    print()
    
    import torch
    import importlib.util
    
    # 兼容 numpy 版本
    if not hasattr(np, '_core'):
        sys.modules['numpy._core'] = np.core
        sys.modules['numpy._core.multiarray'] = np.core.multiarray
    
    def _import_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    phase4_root = PROJECT_ROOT / "src" / "phase4"
    normalizer_mod = _import_module("normalizer", phase4_root / "data" / "normalizer.py")
    scheduler_mod = _import_module("scheduler", phase4_root / "diffusion" / "scheduler.py")
    unet_mod = _import_module("unet1d", phase4_root / "model" / "unet1d.py")
    dataset_mod = _import_module("dataset", phase4_root / "data" / "dataset.py")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = PROJECT_ROOT / "data" / "output" / "phase4_checkpoints" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    
    print(f"模型配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    # 加载模型
    model = unet_mod.UNet1D(
        obs_dim=config["obs_dim"],
        act_dim=config["act_dim"],
        base_channels=config["base_channels"],
        cond_dim=config["cond_dim"],
        time_dim=config["time_dim"],
    ).to(device)
    model.load_state_dict(ckpt["ema_state_dict"])
    model.eval()
    
    # 加载归一化器
    action_normalizer = normalizer_mod.ActionNormalizer()
    action_normalizer.load_state_dict(ckpt["action_normalizer"])
    obs_normalizer = normalizer_mod.ObsNormalizer()
    obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
    
    # 创建数据集
    dataset = dataset_mod.TrajectorySlidingWindow(
        h5_path=PROJECT_ROOT / "data" / "output" / "trajectories.h5",
        history=config["history"],
        future=config["future"],
        nav_fields_dir=PROJECT_ROOT / "data" / "processed" / "nav_fields",
    )
    
    # 取一个样本
    sample = dataset[1000]
    obs = sample["obs"].unsqueeze(0).to(device)  # (1, history, 6)
    action = sample["action"].unsqueeze(0).to(device)  # (1, future, 2)
    
    print("推理流程检查:")
    print(f"  1. obs 原始形状: {obs.shape}")  # (1, history, 6)
    
    # 归一化
    obs_normed = obs_normalizer.transform(obs)
    print(f"  2. obs 归一化后: {obs_normed.shape}")
    
    # 展平为 global_cond
    global_cond = obs_normed.reshape(1, -1)
    print(f"  3. global_cond: {global_cond.shape}")  # (1, history*6) = (1, 12)
    
    # 参照文档: Conv1D 期望 (B, C, T) 格式
    scheduler = scheduler_mod.DDIMScheduler(
        num_diffusion_steps=config["num_diffusion_steps"],
        num_inference_steps=20,
    )
    
    print(f"  4. DDIM 采样...")
    shape = (1, config["future"], config["act_dim"])  # (1, 8, 2)
    
    with torch.no_grad():
        pred = scheduler.sample(model, shape, global_cond, device)
    
    print(f"  5. 预测输出形状: {pred.shape}")  # (1, 8, 2)
    
    # 反归一化
    pred_denorm = action_normalizer.inverse_transform(pred)
    print(f"  6. 反归一化后: {pred_denorm.shape}")
    
    # 对比 GT
    gt = sample["action"].numpy()
    pred_np = pred_denorm.cpu().numpy()[0]
    
    print()
    print("预测 vs GT (前 3 步):")
    for i in range(3):
        p, g = pred_np[i], gt[i]
        cos_sim = np.dot(p, g) / (np.linalg.norm(p) * np.linalg.norm(g) + 1e-8)
        print(f"  step {i}: pred={p}, gt={g}, cos_sim={cos_sim:.3f}")
    
    print()
    return True


def main():
    print("\n" + "=" * 60)
    print("数据一致性验证 (参照 CODE_DATA_STRUCTURE.md)")
    print("=" * 60 + "\n")
    
    results = {}
    
    # 验证 1
    results["phase3_data"] = verify_phase3_data()
    
    # 验证 2
    results["nav_fields"] = verify_nav_fields()
    
    # 验证 3
    results["phase4_dataset"] = verify_phase4_dataset()
    
    # 验证 4
    results["model_inference"] = verify_model_inference()
    
    # 汇总
    print("=" * 60)
    print("验证汇总")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
