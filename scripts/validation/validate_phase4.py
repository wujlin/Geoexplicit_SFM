"""
Phase 4 验证：Diffusion Policy 推理质量评估

使用实际的推理流程（DDIM 采样）评估预测质量
"""

import sys
import importlib.util
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import h5py

from . import PathConfig, Phase4Metrics, get_path_config, load_nav_fields


def _import_module(name: str, path: Path):
    """动态导入模块"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_phase4(
    paths: PathConfig = None, 
    save_figure: bool = True,
    n_samples: int = 50000,
    device: str = None
) -> Phase4Metrics:
    """
    验证 Phase 4 Diffusion Policy 推理质量
    
    检查项：
    1. 预测速度与 GT 速度的余弦相似度
    2. 预测速度与导航场方向的一致性
    3. 方向正确率（与导航场点积 > 0）
    """
    if paths is None:
        paths = get_path_config()
    
    paths.ensure_dirs()
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 动态导入 phase4 模块（避免包结构问题）
    phase4_root = paths.project_root / "src" / "phase4"
    
    normalizer_module = _import_module("normalizer", phase4_root / "data" / "normalizer.py")
    scheduler_module = _import_module("scheduler", phase4_root / "diffusion" / "scheduler.py")
    unet_module = _import_module("unet1d", phase4_root / "model" / "unet1d.py")
    
    ActionNormalizer = normalizer_module.ActionNormalizer
    ObsNormalizer = normalizer_module.ObsNormalizer
    DDIMScheduler = scheduler_module.DDIMScheduler
    UNet1D = unet_module.UNet1D
    
    # 加载检查点
    print(f"Loading checkpoint from {paths.best_checkpoint}...")
    
    # 兼容 numpy 版本
    if not hasattr(np, '_core'):
        sys.modules['numpy._core'] = np.core
        sys.modules['numpy._core.multiarray'] = np.core.multiarray
    
    ckpt = torch.load(paths.best_checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    
    print(f"  Model config: history={config['history']}, future={config['future']}, obs_dim={config['obs_dim']}")
    
    # 创建模型
    model = UNet1D(
        obs_dim=config["obs_dim"],
        act_dim=config["act_dim"],
        base_channels=config["base_channels"],
        cond_dim=config["cond_dim"],
        time_dim=config["time_dim"],
    ).to(device)
    
    # 加载 EMA 权重
    if "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
        print("  Loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        print("  Loaded model weights")
    model.eval()
    
    # 加载归一化器
    action_normalizer = ActionNormalizer()
    action_normalizer.load_state_dict(ckpt["action_normalizer"])
    
    obs_normalizer = None
    if "obs_normalizer" in ckpt:
        obs_normalizer = ObsNormalizer()
        obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
        print("  Loaded obs normalizer")
    
    # 创建调度器（DDIM 更快）
    scheduler = DDIMScheduler(
        num_diffusion_steps=config["num_diffusion_steps"],
        num_inference_steps=20,  # DDIM 加速
    )
    
    # 加载轨迹数据 - 注意 HDF5 格式是 (T, N, 2)
    print(f"Loading trajectories from {paths.trajectories_h5}...")
    with h5py.File(paths.trajectories_h5, "r") as f:
        # 格式: (T, N, 2) -> 需要转置获取 agent-wise 数据
        positions = f["positions"][:]      # (T, N, 2)
        velocities = f["velocities"][:]    # (T, N, 2)
        destinations = f["destinations"][:]  # (T, N) or (N,)
        
        T_max, N_agents, _ = positions.shape
        print(f"  Loaded: T={T_max}, N_agents={N_agents}")
    
    # 处理 destinations 格式
    if destinations.ndim == 2:
        # 取第一个时刻的 destination（假设不变）
        agent_destinations = destinations[0]  # (N,)
    else:
        agent_destinations = destinations
    
    # 加载导航场
    nav_fields = load_nav_fields(paths.nav_fields_dir)
    print(f"  Loaded {len(nav_fields)} navigation fields")
    
    # 参数
    history = config["history"]
    future = config["future"]
    
    # 采样验证集 - 需要足够的历史和未来数据
    print(f"Sampling validation data (need history={history}, future={future})...")
    valid_samples = []  # (agent_idx, time_idx)
    
    for agent_idx in range(N_agents):
        # 找到有效时间范围（位置非零）
        agent_pos = positions[:, agent_idx, :]  # (T, 2)
        valid_mask = np.any(agent_pos != 0, axis=1)
        valid_times = np.where(valid_mask)[0]
        
        if len(valid_times) < history + future:
            continue
        
        # 采样时刻需要确保有 history 帧历史和 future 帧未来
        t_start = valid_times[0] + history - 1
        t_end = valid_times[-1] - future
        
        if t_end <= t_start:
            continue
        
        # 每个 agent 采样一些时间点
        sample_times = np.arange(t_start, t_end, max(1, (t_end - t_start) // 100))
        for t in sample_times[:100]:  # 每个 agent 最多 100 个样本
            valid_samples.append((agent_idx, t))
    
    print(f"  Found {len(valid_samples)} valid samples")
    
    if len(valid_samples) > n_samples:
        sample_idx = np.random.choice(len(valid_samples), n_samples, replace=False)
        valid_samples = [valid_samples[i] for i in sample_idx]
    
    print(f"Validating with {len(valid_samples)} samples...")
    
    # 批量推理
    batch_size = 512
    all_cos_sim_gt = []
    all_cos_sim_nav = []
    all_nav_dot = []
    
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(valid_samples), batch_size), desc="Inference"):
            batch_end = min(batch_start + batch_size, len(valid_samples))
            batch_samples = valid_samples[batch_start:batch_end]
            
            obs_list = []
            gt_actions_list = []
            nav_actions_list = []
            
            for agent_idx, t in batch_samples:
                dest_id = int(agent_destinations[agent_idx])
                
                # 获取导航场
                if dest_id not in nav_fields:
                    continue
                nav_field = nav_fields[dest_id]  # (2, H, W)
                H, W = nav_field.shape[1], nav_field.shape[2]
                
                # 构建历史观测 (history, 6): [pos, vel, nav_dir]
                obs_frames = []
                for h in range(history):
                    t_h = t - history + 1 + h
                    pos = positions[t_h, agent_idx]  # (2,)
                    vel = velocities[t_h, agent_idx]  # (2,)
                    
                    # 获取导航方向
                    iy = int(np.clip(pos[0], 0, H - 1))
                    ix = int(np.clip(pos[1], 0, W - 1))
                    nav_dir = nav_field[:, iy, ix]
                    # 归一化
                    nav_norm = np.linalg.norm(nav_dir)
                    if nav_norm > 1e-6:
                        nav_dir = nav_dir / nav_norm
                    else:
                        nav_dir = np.array([0.0, 0.0])
                    
                    obs_frames.append(np.concatenate([pos, vel, nav_dir]))
                
                obs = np.stack(obs_frames, axis=0)  # (history, 6)
                obs_list.append(obs)
                
                # GT 未来速度 (future, 2)
                gt_vel = velocities[t+1:t+1+future, agent_idx]  # (future, 2)
                gt_actions_list.append(gt_vel)
                
                # 当前导航方向（用于评估）
                curr_pos = positions[t, agent_idx]
                iy = int(np.clip(curr_pos[0], 0, H - 1))
                ix = int(np.clip(curr_pos[1], 0, W - 1))
                nav_dir = nav_field[:, iy, ix]
                nav_norm = np.linalg.norm(nav_dir)
                if nav_norm > 1e-6:
                    nav_dir = nav_dir / nav_norm
                nav_actions_list.append(nav_dir)
            
            if len(obs_list) == 0:
                continue
            
            # 转换为 tensor
            obs_batch = torch.from_numpy(np.array(obs_list, dtype=np.float32)).to(device)  # (B, history, 6)
            gt_actions = np.array(gt_actions_list)  # (B, future, 2)
            nav_actions = np.array(nav_actions_list)  # (B, 2)
            
            B = obs_batch.shape[0]
            
            # obs 归一化
            if obs_normalizer is not None:
                obs_batch = obs_normalizer.transform(obs_batch)
            
            # 展平为 global_cond
            global_cond = obs_batch.reshape(B, -1)  # (B, history * 6)
            
            # DDIM 采样预测动作
            shape = (B, future, config["act_dim"])
            pred_actions = scheduler.sample(
                model=model,
                shape=shape,
                condition=global_cond,
                device=device,
            )  # (B, future, 2)
            
            # 反归一化
            pred_actions = action_normalizer.inverse_transform(pred_actions)
            pred_actions = pred_actions.cpu().numpy()  # (B, future, 2)
            
            # 计算余弦相似度（使用第一步预测）
            for j in range(len(pred_actions)):
                pred = pred_actions[j, 0]  # 第一步预测速度
                gt = gt_actions[j, 0]      # GT 第一步速度
                nav = nav_actions[j]       # 导航方向
                
                pred_norm = np.linalg.norm(pred)
                gt_norm = np.linalg.norm(gt)
                nav_norm = np.linalg.norm(nav)
                
                # 只有当所有 norm 都有效时才计算
                if pred_norm > 1e-6 and gt_norm > 1e-6 and nav_norm > 1e-6:
                    cos_sim_gt = np.dot(pred, gt) / (pred_norm * gt_norm)
                    cos_sim_nav = np.dot(pred, nav) / (pred_norm * nav_norm)
                    nav_dot = np.dot(pred, nav)
                    
                    all_cos_sim_gt.append(cos_sim_gt)
                    all_cos_sim_nav.append(cos_sim_nav)
                    all_nav_dot.append(nav_dot)
    
    # 计算统计
    all_cos_sim_gt = np.array(all_cos_sim_gt)
    all_cos_sim_nav = np.array(all_cos_sim_nav)
    all_nav_dot = np.array(all_nav_dot)
    
    if len(all_cos_sim_gt) == 0:
        print("❌ No valid samples found for evaluation!")
        return Phase4Metrics(
            n_samples=0,
            mean_cos_sim_gt=0.0,
            std_cos_sim_gt=0.0,
            mean_cos_sim_nav=0.0,
            std_cos_sim_nav=0.0,
            direction_correct_ratio=0.0,
            aligned_ratio=0.0,
        )
    
    metrics = Phase4Metrics(
        n_samples=len(all_cos_sim_gt),
        mean_cos_sim_gt=float(np.mean(all_cos_sim_gt)),
        std_cos_sim_gt=float(np.std(all_cos_sim_gt)),
        mean_cos_sim_nav=float(np.mean(all_cos_sim_nav)),
        std_cos_sim_nav=float(np.std(all_cos_sim_nav)),
        direction_correct_ratio=float((all_nav_dot > 0).sum() / len(all_nav_dot)),
        aligned_ratio=float((all_cos_sim_gt > 0.5).sum() / len(all_cos_sim_gt)),
    )
    
    # 可视化
    if save_figure:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Pred vs GT 余弦相似度分布
        ax = axes[0, 0]
        ax.hist(all_cos_sim_gt, bins=50, density=True, alpha=0.7, edgecolor="black")
        ax.axvline(metrics.mean_cos_sim_gt, color="r", linestyle="--", 
                   label=f"Mean: {metrics.mean_cos_sim_gt:.3f}")
        ax.set_xlabel("Cosine Similarity (Pred vs GT)")
        ax.set_ylabel("Density")
        ax.set_title("Prediction vs Ground Truth Alignment")
        ax.legend()
        
        # 2. Pred vs Nav 余弦相似度分布
        ax = axes[0, 1]
        ax.hist(all_cos_sim_nav, bins=50, density=True, alpha=0.7, edgecolor="black", color="green")
        ax.axvline(metrics.mean_cos_sim_nav, color="r", linestyle="--",
                   label=f"Mean: {metrics.mean_cos_sim_nav:.3f}")
        ax.set_xlabel("Cosine Similarity (Pred vs Nav)")
        ax.set_ylabel("Density")
        ax.set_title("Prediction vs Navigation Field Alignment")
        ax.legend()
        
        # 3. 方向正确率（散点图）
        ax = axes[1, 0]
        ax.scatter(all_cos_sim_nav, all_cos_sim_gt, alpha=0.1, s=1)
        ax.axhline(0, color="gray", linestyle="-", alpha=0.5)
        ax.axvline(0, color="gray", linestyle="-", alpha=0.5)
        ax.set_xlabel("Cos Sim (Pred vs Nav)")
        ax.set_ylabel("Cos Sim (Pred vs GT)")
        ax.set_title("Prediction Alignment Scatter")
        
        # 4. 统计摘要
        ax = axes[1, 1]
        ax.axis("off")
        summary_text = f"""
Phase 4 Diffusion Policy Validation Summary

Samples evaluated: {metrics.n_samples:,}

Prediction vs GT:
  Mean cos_sim: {metrics.mean_cos_sim_gt:.4f} ± {metrics.std_cos_sim_gt:.4f}
  Aligned (>0.5): {metrics.aligned_ratio * 100:.1f}%

Prediction vs Navigation:
  Mean cos_sim: {metrics.mean_cos_sim_nav:.4f} ± {metrics.std_cos_sim_nav:.4f}
  Correct direction (>0): {metrics.direction_correct_ratio * 100:.1f}%

Quality Assessment:
  {"✅ Excellent" if metrics.mean_cos_sim_gt > 0.8 else "✅ Good" if metrics.mean_cos_sim_gt > 0.6 else "⚠️ Moderate" if metrics.mean_cos_sim_gt > 0.4 else "❌ Poor"}
"""
        ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment="center",
                fontfamily="monospace", transform=ax.transAxes)
        
        plt.tight_layout()
        fig.savefig(paths.figures_dir / "phase4_diffusion.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {paths.figures_dir / 'phase4_diffusion.png'}")
    
    # 保存指标
    np.savez(
        paths.metrics_dir / "phase4_metrics.npz",
        n_samples=metrics.n_samples,
        mean_cos_sim_gt=metrics.mean_cos_sim_gt,
        std_cos_sim_gt=metrics.std_cos_sim_gt,
        mean_cos_sim_nav=metrics.mean_cos_sim_nav,
        std_cos_sim_nav=metrics.std_cos_sim_nav,
        direction_correct_ratio=metrics.direction_correct_ratio,
        aligned_ratio=metrics.aligned_ratio,
        cos_sim_gt=all_cos_sim_gt,
        cos_sim_nav=all_cos_sim_nav,
    )
    
    # 打印结果
    print("\n" + "=" * 60)
    print("Phase 4 Validation Results")
    print("=" * 60)
    print(f"  Samples: {metrics.n_samples:,}")
    print(f"  Pred vs GT cos_sim: {metrics.mean_cos_sim_gt:.4f} ± {metrics.std_cos_sim_gt:.4f}")
    print(f"  Pred vs Nav cos_sim: {metrics.mean_cos_sim_nav:.4f} ± {metrics.std_cos_sim_nav:.4f}")
    print(f"  Direction correct (dot > 0): {metrics.direction_correct_ratio * 100:.1f}%")
    print(f"  Aligned with GT (cos > 0.5): {metrics.aligned_ratio * 100:.1f}%")
    
    if metrics.mean_cos_sim_gt > 0.8:
        print("  ✅ Diffusion Policy quality excellent")
    elif metrics.mean_cos_sim_gt > 0.6:
        print("  ✅ Diffusion Policy quality good")
    elif metrics.mean_cos_sim_gt > 0.4:
        print("  ⚠️ Diffusion Policy quality moderate")
    else:
        print("  ❌ Diffusion Policy quality needs improvement")
    
    return metrics


if __name__ == "__main__":
    validate_phase4()
