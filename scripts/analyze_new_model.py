"""分析新模型的推理效果"""
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "phase4"))

from inference import DiffusionPolicyInference

def analyze_traj(traj, nav_field):
    """分析单条轨迹"""
    velocities = np.diff(traj, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    
    cos_sims = []
    for i in range(len(velocities)):
        if speeds[i] < 0.01:
            continue
        pos = traj[i]
        iy = int(np.clip(pos[0], 0, nav_field.shape[1]-1))
        ix = int(np.clip(pos[1], 0, nav_field.shape[2]-1))
        nav_d = nav_field[:, iy, ix]
        nav_norm = np.linalg.norm(nav_d)
        if nav_norm > 0.01:
            cos_sim = np.dot(nav_d/nav_norm, velocities[i]/speeds[i])
            cos_sims.append(cos_sim)
    
    return {
        'mean_speed': speeds.mean(),
        'cos_sim': np.mean(cos_sims) if cos_sims else 0,
        'moving_ratio': (speeds > 0.1).mean()
    }


def main():
    # 加载数据
    nav = np.load(PROJECT_ROOT / "data" / "processed" / "nav_baseline.npz")
    nav_field = np.stack([nav['nav_y'], nav['nav_x']], axis=0)
    walkable = np.load(PROJECT_ROOT / "data" / "processed" / "walkable_mask.npy")
    
    inferencer = DiffusionPolicyInference(
        PROJECT_ROOT / "data" / "output" / "phase4_checkpoints" / "best.pt",
        use_ddim=True, ddim_steps=10
    )
    
    # 多个起点
    np.random.seed(42)
    walkable_pts = np.argwhere(walkable)
    nav_mag = np.sqrt(nav_field[0]**2 + nav_field[1]**2)
    good_pts = walkable_pts[nav_mag[walkable_pts[:,0], walkable_pts[:,1]] > 0.5]
    starts = good_pts[np.random.choice(len(good_pts), 10, replace=False)].astype(float)
    
    # Test 1: 纯模型
    print("=== nav_weight=0.0 (纯模型预测方向) ===")
    results_pure = []
    for start in starts:
        traj = inferencer.simulate(start, walkable, nav_field, max_steps=300, nav_weight=0.0)
        results_pure.append(analyze_traj(traj, nav_field))
    
    avg_speed = np.mean([r['mean_speed'] for r in results_pure])
    avg_cos = np.mean([r['cos_sim'] for r in results_pure])
    avg_moving = np.mean([r['moving_ratio'] for r in results_pure])
    print(f"平均速度: {avg_speed:.3f}")
    print(f"方向一致性: {avg_cos:.3f}")
    print(f"移动比例: {avg_moving*100:.1f}%")
    
    # Test 2: 混合模式
    print("\n=== nav_weight=0.7 (混合模式) ===")
    results_hybrid = []
    for start in starts:
        traj = inferencer.simulate(start, walkable, nav_field, max_steps=300, nav_weight=0.7)
        results_hybrid.append(analyze_traj(traj, nav_field))
    
    avg_speed = np.mean([r['mean_speed'] for r in results_hybrid])
    avg_cos = np.mean([r['cos_sim'] for r in results_hybrid])
    avg_moving = np.mean([r['moving_ratio'] for r in results_hybrid])
    print(f"平均速度: {avg_speed:.3f}")
    print(f"方向一致性: {avg_cos:.3f}")
    print(f"移动比例: {avg_moving*100:.1f}%")


if __name__ == "__main__":
    main()
