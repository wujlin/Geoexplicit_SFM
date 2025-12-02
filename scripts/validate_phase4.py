"""
Phase 4 Diffusion Policy 标准化验证脚本

功能：
1. 模型基本信息检查
2. 单步预测质量评估 (Pred vs GT)
3. 条件响应测试 (模型是否对 nav 条件敏感)
4. 闭环仿真评估 (轨迹是否朝向 Sink)
5. 生成可视化图表

用法：
    python scripts/validate_phase4.py
    python scripts/validate_phase4.py --checkpoint path/to/model.pt
    python scripts/validate_phase4.py --quick  # 快速模式，减少样本数
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any
import json
import time

import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 导入 Phase 4 模块
import importlib.util

def _import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

PHASE4_ROOT = PROJECT_ROOT / "src" / "phase4"
normalizer_module = _import_module("normalizer", PHASE4_ROOT / "data" / "normalizer.py")
scheduler_module = _import_module("scheduler", PHASE4_ROOT / "diffusion" / "scheduler.py")
unet_module = _import_module("unet1d", PHASE4_ROOT / "model" / "unet1d.py")

ActionNormalizer = normalizer_module.ActionNormalizer
ObsNormalizer = normalizer_module.ObsNormalizer
DDIMScheduler = scheduler_module.DDIMScheduler
UNet1D = unet_module.UNet1D


class Phase4Validator:
    """Phase 4 模型验证器"""
    
    def __init__(self, checkpoint_path: Path, device: str = "auto"):
        self.checkpoint_path = checkpoint_path
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 结果存储 (必须先初始化)
        self.results: Dict[str, Any] = {}
        
        # 加载模型
        self._load_model()
        
        # 加载数据
        self._load_data()
    
    def _load_model(self):
        """加载模型和配置"""
        print(f"\n{'='*60}")
        print("1. 加载模型")
        print(f"{'='*60}")
        
        # numpy 兼容性补丁
        if not hasattr(np, '_core'):
            sys.modules['numpy._core'] = np.core
            sys.modules['numpy._core.multiarray'] = np.core.multiarray
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        self.config = checkpoint["config"]
        self.epoch = checkpoint.get("epoch", "unknown")
        self.loss = checkpoint.get("loss", "unknown")
        
        # 创建模型
        self.model = UNet1D(
            obs_dim=self.config["obs_dim"],
            act_dim=self.config["act_dim"],
            base_channels=self.config["base_channels"],
            cond_dim=self.config["cond_dim"],
            time_dim=self.config["time_dim"],
        ).to(self.device)
        
        # 优先使用 EMA 权重
        if "ema_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["ema_state_dict"])
            self.weight_type = "EMA"
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.weight_type = "Normal"
        
        self.model.eval()
        
        # 归一化器
        self.action_normalizer = ActionNormalizer()
        self.action_normalizer.load_state_dict(checkpoint["action_normalizer"])
        
        self.obs_normalizer = None
        if "obs_normalizer" in checkpoint:
            self.obs_normalizer = ObsNormalizer()
            self.obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
        
        # DDIM 调度器
        self.scheduler = DDIMScheduler(
            num_diffusion_steps=self.config["num_diffusion_steps"],
            num_inference_steps=20,
        )
        
        # 模型参数量
        num_params = sum(p.numel() for p in self.model.parameters())
        model_size_mb = num_params * 4 / (1024 * 1024)
        
        print(f"  Checkpoint: {self.checkpoint_path.name}")
        print(f"  Epoch: {self.epoch}")
        print(f"  Loss: {self.loss:.6f}" if isinstance(self.loss, float) else f"  Loss: {self.loss}")
        print(f"  Weight Type: {self.weight_type}")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {num_params:,} ({model_size_mb:.2f} MB)")
        print(f"  Config: history={self.config['history']}, future={self.config['future']}, obs_dim={self.config['obs_dim']}")
        
        self.results["model_info"] = {
            "checkpoint": str(self.checkpoint_path),
            "epoch": self.epoch,
            "loss": float(self.loss) if isinstance(self.loss, float) else self.loss,
            "weight_type": self.weight_type,
            "num_params": num_params,
            "model_size_mb": model_size_mb,
            "config": self.config,
        }
    
    def _load_data(self):
        """加载轨迹数据和导航场"""
        print(f"\n{'='*60}")
        print("2. 加载数据")
        print(f"{'='*60}")
        
        # 轨迹数据
        traj_path = PROJECT_ROOT / "data" / "output" / "trajectories.h5"
        with h5py.File(traj_path, "r") as f:
            self.positions = f["positions"][:]  # (T, N, 2)
            self.velocities = f["velocities"][:]  # (T, N, 2)
        
        self.T, self.N, _ = self.positions.shape
        print(f"  Trajectories: {self.T} steps × {self.N} agents")
        
        # 导航场
        nav_path = PROJECT_ROOT / "data" / "processed" / "nav_baseline.npz"
        nav_data = np.load(nav_path)
        self.nav_field = np.stack([nav_data["nav_y"], nav_data["nav_x"]], axis=0)  # (2, H, W)
        self.H, self.W = self.nav_field.shape[1], self.nav_field.shape[2]
        print(f"  Nav Field: {self.H} × {self.W}")
        
        # 距离场
        dist_path = PROJECT_ROOT / "data" / "processed" / "distance_field.npy"
        self.distance_field = np.load(dist_path)
        print(f"  Distance Field: {self.distance_field.shape}")
        
        # Walkable mask
        mask_path = PROJECT_ROOT / "data" / "processed" / "walkable_mask.npy"
        self.walkable_mask = np.load(mask_path)
        print(f"  Walkable Mask: {self.walkable_mask.shape}")
        
        self.results["data_info"] = {
            "trajectories_shape": [self.T, self.N, 2],
            "nav_field_shape": list(self.nav_field.shape),
            "map_size": [self.H, self.W],
        }
    
    def _get_nav_at_pos(self, pos: np.ndarray) -> np.ndarray:
        """获取位置处的导航方向"""
        y = int(np.clip(pos[0], 0, self.H - 1))
        x = int(np.clip(pos[1], 0, self.W - 1))
        nav = self.nav_field[:, y, x]
        norm = np.linalg.norm(nav)
        if norm > 1e-6:
            return nav / norm
        return np.array([0.0, 0.0])
    
    def _build_obs(self, t: int, agent: int) -> np.ndarray:
        """构建 observation: (history, 6) = [pos, vel, nav]"""
        history = self.config["history"]
        obs_list = []
        for h in range(history):
            ti = t + h
            pos = self.positions[ti, agent]
            vel = self.velocities[ti, agent]
            nav = self._get_nav_at_pos(pos)
            obs_list.append(np.concatenate([pos, vel, nav]))
        return np.stack(obs_list, axis=0)  # (history, 6)
    
    @torch.no_grad()
    def _predict_batch(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """批量预测动作"""
        B = obs_batch.shape[0]
        
        # 归一化 obs
        if self.obs_normalizer is not None:
            obs_batch = self.obs_normalizer.transform(obs_batch)
        
        # Flatten obs: (B, history, 6) -> (B, history * 6)
        obs_flat = obs_batch.reshape(B, -1)
        
        # 采样
        shape = (B, self.config["future"], self.config["act_dim"])
        samples = self.scheduler.sample(
            model=self.model,
            shape=shape,
            condition=obs_flat,
            device=self.device,
        )
        
        # 反归一化
        samples = self.action_normalizer.inverse_transform(samples)
        return samples
    
    def evaluate_prediction_quality(self, num_samples: int = 10000, batch_size: int = 512):
        """
        评估 1: 单步预测质量
        比较模型预测的动作与 Ground Truth 的一致性
        """
        print(f"\n{'='*60}")
        print("3. 评估单步预测质量")
        print(f"{'='*60}")
        
        history = self.config["history"]
        future = self.config["future"]
        
        # 采样有效的 (t, agent) 对
        valid_t_max = self.T - history - future
        if valid_t_max <= 0:
            print("  错误: 轨迹长度不足")
            return
        
        np.random.seed(42)
        sample_t = np.random.randint(0, valid_t_max, num_samples)
        sample_agent = np.random.randint(0, self.N, num_samples)
        
        # 收集结果
        all_pred_vel = []
        all_gt_vel = []
        all_nav = []
        
        # 批量处理
        for start in tqdm(range(0, num_samples, batch_size), desc="  预测中"):
            end = min(start + batch_size, num_samples)
            batch_t = sample_t[start:end]
            batch_agent = sample_agent[start:end]
            
            # 构建 obs batch
            obs_list = []
            gt_list = []
            nav_list = []
            for t, agent in zip(batch_t, batch_agent):
                obs = self._build_obs(t, agent)
                gt = self.velocities[t + history : t + history + future, agent]  # (future, 2)
                nav = self._get_nav_at_pos(self.positions[t + history, agent])
                
                obs_list.append(obs)
                gt_list.append(gt)
                nav_list.append(nav)
            
            obs_batch = torch.tensor(np.stack(obs_list), dtype=torch.float32, device=self.device)
            pred = self._predict_batch(obs_batch).cpu().numpy()  # (B, future, 2)
            
            all_pred_vel.extend(pred)
            all_gt_vel.extend(gt_list)
            all_nav.extend(nav_list)
        
        all_pred_vel = np.array(all_pred_vel)  # (N, future, 2)
        all_gt_vel = np.array(all_gt_vel)
        all_nav = np.array(all_nav)  # (N, 2)
        
        # === 计算指标 ===
        
        # 1. Pred vs GT 的 cosine similarity (取第一帧)
        pred_first = all_pred_vel[:, 0]  # (N, 2)
        gt_first = all_gt_vel[:, 0]
        
        pred_norm = pred_first / (np.linalg.norm(pred_first, axis=1, keepdims=True) + 1e-8)
        gt_norm = gt_first / (np.linalg.norm(gt_first, axis=1, keepdims=True) + 1e-8)
        
        cos_pred_gt = (pred_norm * gt_norm).sum(axis=1)
        
        # 2. GT vs Nav 的 cosine similarity (作为参考)
        nav_norm = all_nav / (np.linalg.norm(all_nav, axis=1, keepdims=True) + 1e-8)
        cos_gt_nav = (gt_norm * nav_norm).sum(axis=1)
        
        # 3. Pred vs Nav
        cos_pred_nav = (pred_norm * nav_norm).sum(axis=1)
        
        # 4. 速度幅度对比
        pred_speed = np.linalg.norm(pred_first, axis=1)
        gt_speed = np.linalg.norm(gt_first, axis=1)
        
        # 5. MSE
        mse_first = ((pred_first - gt_first) ** 2).mean()
        mse_all = ((all_pred_vel - all_gt_vel) ** 2).mean()
        
        # 输出结果
        print(f"\n  样本数: {num_samples}")
        print(f"\n  [Cosine Similarity]")
        print(f"    Pred vs GT:  mean={cos_pred_gt.mean():.4f}, >0: {(cos_pred_gt > 0).mean()*100:.1f}%")
        print(f"    GT vs Nav:   mean={cos_gt_nav.mean():.4f}, >0: {(cos_gt_nav > 0).mean()*100:.1f}%")
        print(f"    Pred vs Nav: mean={cos_pred_nav.mean():.4f}, >0: {(cos_pred_nav > 0).mean()*100:.1f}%")
        print(f"\n  [Speed]")
        print(f"    GT speed:   mean={gt_speed.mean():.4f}, std={gt_speed.std():.4f}")
        print(f"    Pred speed: mean={pred_speed.mean():.4f}, std={pred_speed.std():.4f}")
        print(f"\n  [MSE]")
        print(f"    First frame: {mse_first:.6f}")
        print(f"    All frames:  {mse_all:.6f}")
        
        self.results["prediction_quality"] = {
            "num_samples": num_samples,
            "cos_pred_gt_mean": float(cos_pred_gt.mean()),
            "cos_pred_gt_positive_rate": float((cos_pred_gt > 0).mean()),
            "cos_gt_nav_mean": float(cos_gt_nav.mean()),
            "cos_gt_nav_positive_rate": float((cos_gt_nav > 0).mean()),
            "cos_pred_nav_mean": float(cos_pred_nav.mean()),
            "cos_pred_nav_positive_rate": float((cos_pred_nav > 0).mean()),
            "gt_speed_mean": float(gt_speed.mean()),
            "pred_speed_mean": float(pred_speed.mean()),
            "mse_first_frame": float(mse_first),
            "mse_all_frames": float(mse_all),
        }
        
        # 保存分布数据用于绘图
        self._pred_quality_data = {
            "cos_pred_gt": cos_pred_gt,
            "cos_pred_nav": cos_pred_nav,
            "cos_gt_nav": cos_gt_nav,
            "pred_speed": pred_speed,
            "gt_speed": gt_speed,
        }
    
    def evaluate_condition_response(self, num_tests: int = 100):
        """
        评估 2: 条件响应测试
        测试模型是否对不同的 nav 条件产生不同的输出
        """
        print(f"\n{'='*60}")
        print("4. 评估条件响应")
        print(f"{'='*60}")
        
        history = self.config["history"]
        
        # 固定位置和速度，变化 nav 方向
        # 使用 4 个正交方向: (1,0), (0,1), (-1,0), (0,-1)
        test_navs = np.array([
            [1.0, 0.0],   # 向 y+
            [0.0, 1.0],   # 向 x+
            [-1.0, 0.0],  # 向 y-
            [0.0, -1.0],  # 向 x-
        ])
        
        # 随机选择一些起点
        np.random.seed(123)
        valid_t = self.T // 2
        test_agents = np.random.choice(self.N, num_tests, replace=False)
        
        responses = {i: [] for i in range(4)}  # 4 个方向的响应
        
        for agent in tqdm(test_agents, desc="  条件测试"):
            pos = self.positions[valid_t, agent]
            vel = self.velocities[valid_t, agent]
            
            for nav_idx, nav in enumerate(test_navs):
                # 构建 obs
                obs = np.zeros((history, 6), dtype=np.float32)
                for h in range(history):
                    obs[h, 0:2] = pos
                    obs[h, 2:4] = vel
                    obs[h, 4:6] = nav
                
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                pred = self._predict_batch(obs_tensor).cpu().numpy()[0, 0]  # 取第一帧
                
                responses[nav_idx].append(pred)
        
        # 分析响应
        print(f"\n  样本数: {num_tests}")
        print(f"\n  [各方向预测结果均值]")
        
        mean_responses = {}
        for nav_idx, nav in enumerate(test_navs):
            preds = np.array(responses[nav_idx])  # (num_tests, 2)
            mean_pred = preds.mean(axis=0)
            mean_responses[nav_idx] = mean_pred
            
            # 计算与输入 nav 的一致性
            cos_sim = np.dot(mean_pred / (np.linalg.norm(mean_pred) + 1e-8), 
                           nav / (np.linalg.norm(nav) + 1e-8))
            
            print(f"    nav={nav} → pred_mean={mean_pred} | cos(pred, nav)={cos_sim:.3f}")
        
        # 条件敏感性: 不同 nav 导致的输出差异
        # 计算 4 个方向响应的方差
        all_means = np.array([mean_responses[i] for i in range(4)])
        condition_variance = all_means.var(axis=0).mean()
        
        print(f"\n  [条件敏感性]")
        print(f"    响应方差 (越大越好): {condition_variance:.4f}")
        
        self.results["condition_response"] = {
            "num_tests": num_tests,
            "responses": {str(test_navs[i].tolist()): mean_responses[i].tolist() for i in range(4)},
            "condition_variance": float(condition_variance),
        }
    
    def evaluate_trajectory_quality(self, num_agents: int = 50, max_steps: int = 200):
        """
        评估 3: 闭环轨迹质量
        测试模型生成的轨迹是否朝向 Sink
        """
        print(f"\n{'='*60}")
        print("5. 评估闭环轨迹质量")
        print(f"{'='*60}")
        
        history = self.config["history"]
        
        # 选择起点 (优先选择离 sink 较远的点)
        walkable_points = np.argwhere(self.walkable_mask)
        np.random.seed(456)
        
        # 选择高 nav magnitude 的点
        nav_mag = np.sqrt(self.nav_field[0]**2 + self.nav_field[1]**2)
        point_mags = nav_mag[walkable_points[:, 0], walkable_points[:, 1]]
        good_indices = np.where(point_mags > 0.5)[0]
        
        if len(good_indices) >= num_agents:
            selected = np.random.choice(good_indices, num_agents, replace=False)
        else:
            selected = np.random.choice(len(walkable_points), num_agents, replace=False)
        
        start_positions = walkable_points[selected].astype(np.float32)
        
        # 运行闭环仿真
        trajectories = []
        initial_distances = []
        final_distances = []
        
        for start_pos in tqdm(start_positions, desc="  仿真中"):
            traj = self._simulate_trajectory(start_pos, max_steps)
            trajectories.append(traj)
            
            # 计算到 sink 的距离变化
            y0, x0 = int(traj[0, 0]), int(traj[0, 1])
            yf, xf = int(traj[-1, 0]), int(traj[-1, 1])
            
            y0, x0 = np.clip(y0, 0, self.H-1), np.clip(x0, 0, self.W-1)
            yf, xf = np.clip(yf, 0, self.H-1), np.clip(xf, 0, self.W-1)
            
            initial_distances.append(self.distance_field[y0, x0])
            final_distances.append(self.distance_field[yf, xf])
        
        initial_distances = np.array(initial_distances)
        final_distances = np.array(final_distances)
        distance_change = final_distances - initial_distances
        
        # 计算指标
        approaching_rate = (distance_change < 0).mean()
        mean_distance_change = distance_change.mean()
        
        # 轨迹长度
        traj_lengths = [len(t) for t in trajectories]
        
        print(f"\n  智能体数: {num_agents}")
        print(f"  最大步数: {max_steps}")
        print(f"\n  [距离变化 (负数表示靠近 Sink)]")
        print(f"    靠近率: {approaching_rate*100:.1f}%")
        print(f"    平均变化: {mean_distance_change:.2f} px")
        print(f"    初始距离: {initial_distances.mean():.1f} ± {initial_distances.std():.1f}")
        print(f"    最终距离: {final_distances.mean():.1f} ± {final_distances.std():.1f}")
        print(f"\n  [轨迹长度]")
        print(f"    平均: {np.mean(traj_lengths):.1f}, 最短: {np.min(traj_lengths)}, 最长: {np.max(traj_lengths)}")
        
        self.results["trajectory_quality"] = {
            "num_agents": num_agents,
            "max_steps": max_steps,
            "approaching_rate": float(approaching_rate),
            "mean_distance_change": float(mean_distance_change),
            "initial_distance_mean": float(initial_distances.mean()),
            "final_distance_mean": float(final_distances.mean()),
            "traj_length_mean": float(np.mean(traj_lengths)),
        }
        
        # 保存轨迹用于可视化
        self._trajectories = trajectories
    
    def _simulate_trajectory(self, start_pos: np.ndarray, max_steps: int) -> np.ndarray:
        """闭环仿真单条轨迹"""
        history = self.config["history"]
        
        pos = start_pos.copy()
        nav = self._get_nav_at_pos(pos)
        vel = nav * 0.8
        
        pos_history = [pos.copy() for _ in range(history)]
        vel_history = [vel.copy() for _ in range(history)]
        nav_history = [nav.copy() for _ in range(history)]
        
        trajectory = [pos.copy()]
        action_horizon = 4  # MPC 执行步数
        
        step = 0
        while step < max_steps:
            # 构建 obs
            obs = np.concatenate([
                np.stack(pos_history[-history:], axis=0),
                np.stack(vel_history[-history:], axis=0),
                np.stack(nav_history[-history:], axis=0),
            ], axis=-1)
            
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_seq = self._predict_batch(obs_tensor).cpu().numpy()[0]  # (future, 2)
            
            # 执行动作
            for i in range(min(action_horizon, len(action_seq))):
                if step >= max_steps:
                    break
                
                vel = action_seq[i]
                new_pos = pos + vel
                new_pos = np.clip(new_pos, [0, 0], [self.H-1, self.W-1])
                
                # 可行走检查
                iy, ix = int(new_pos[0]), int(new_pos[1])
                if self.walkable_mask[iy, ix]:
                    pos = new_pos
                
                nav = self._get_nav_at_pos(pos)
                pos_history.append(pos.copy())
                vel_history.append(vel.copy())
                nav_history.append(nav.copy())
                trajectory.append(pos.copy())
                step += 1
        
        return np.array(trajectory)
    
    def generate_report(self, output_dir: Path):
        """生成验证报告和可视化"""
        print(f"\n{'='*60}")
        print("6. 生成报告")
        print(f"{'='*60}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存 JSON 结果
        json_path = output_dir / "validation_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"  结果保存至: {json_path}")
        
        # 生成可视化
        self._plot_prediction_quality(output_dir)
        self._plot_trajectories(output_dir)
        
        print(f"\n{'='*60}")
        print("验证完成!")
        print(f"{'='*60}")
    
    def _plot_prediction_quality(self, output_dir: Path):
        """绘制预测质量分布图"""
        if not hasattr(self, "_pred_quality_data"):
            return
        
        data = self._pred_quality_data
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 1. Cosine similarity 分布
        ax = axes[0]
        ax.hist(data["cos_pred_gt"], bins=50, alpha=0.7, label="Pred vs GT", color="blue")
        ax.hist(data["cos_gt_nav"], bins=50, alpha=0.7, label="GT vs Nav", color="green")
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")
        ax.set_title("Direction Alignment Distribution")
        ax.legend()
        
        # 2. Pred vs Nav
        ax = axes[1]
        ax.hist(data["cos_pred_nav"], bins=50, alpha=0.7, color="purple")
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        ax.axvline(x=data["cos_pred_nav"].mean(), color="orange", linestyle="-", 
                   label=f"mean={data['cos_pred_nav'].mean():.3f}")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")
        ax.set_title("Pred vs Nav Distribution")
        ax.legend()
        
        # 3. 速度对比
        ax = axes[2]
        ax.hist(data["gt_speed"], bins=50, alpha=0.7, label=f"GT (mean={data['gt_speed'].mean():.2f})", color="green")
        ax.hist(data["pred_speed"], bins=50, alpha=0.7, label=f"Pred (mean={data['pred_speed'].mean():.2f})", color="blue")
        ax.set_xlabel("Speed")
        ax.set_ylabel("Count")
        ax.set_title("Speed Distribution")
        ax.legend()
        
        plt.tight_layout()
        fig_path = output_dir / "prediction_quality.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  图表保存至: {fig_path}")
    
    def _plot_trajectories(self, output_dir: Path):
        """绘制闭环轨迹"""
        if not hasattr(self, "_trajectories"):
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 绘制道路
        ax.imshow(self.walkable_mask.T, cmap="gray", origin="lower", alpha=0.5)
        
        # 绘制轨迹
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self._trajectories)))
        for traj, color in zip(self._trajectories, colors):
            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1, alpha=0.7)
            ax.scatter(traj[0, 0], traj[0, 1], color=color, s=30, marker="o", zorder=5)
            ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=30, marker="x", zorder=5)
        
        ax.set_title(f"Closed-Loop Trajectories (n={len(self._trajectories)})")
        ax.set_xlabel("Y")
        ax.set_ylabel("X")
        ax.set_aspect("equal")
        
        plt.tight_layout()
        fig_path = output_dir / "trajectories.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  图表保存至: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Diffusion Policy Validation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (default: best.pt)")
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer samples")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    
    args = parser.parse_args()
    
    # 默认路径
    if args.checkpoint is None:
        args.checkpoint = PROJECT_ROOT / "data" / "output" / "phase4_checkpoints" / "best.pt"
    else:
        args.checkpoint = Path(args.checkpoint)
    
    if args.output is None:
        args.output = PROJECT_ROOT / "data" / "output" / "phase4_validation"
    else:
        args.output = Path(args.output)
    
    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # 配置
    if args.quick:
        num_pred_samples = 2000
        num_cond_tests = 30
        num_traj_agents = 20
        max_traj_steps = 100
    else:
        num_pred_samples = 10000
        num_cond_tests = 100
        num_traj_agents = 50
        max_traj_steps = 200
    
    # 运行验证
    start_time = time.time()
    
    validator = Phase4Validator(args.checkpoint, device=args.device)
    validator.evaluate_prediction_quality(num_samples=num_pred_samples)
    validator.evaluate_condition_response(num_tests=num_cond_tests)
    validator.evaluate_trajectory_quality(num_agents=num_traj_agents, max_steps=max_traj_steps)
    validator.generate_report(args.output)
    
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.1f} 秒")


if __name__ == "__main__":
    main()
