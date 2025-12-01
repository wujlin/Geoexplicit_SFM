from pathlib import Path

import numpy as np

from src.phase2 import config
from src.phase2.baseline import solve_field
from src.phase2.common.visualizer import plot_field


def ensure_output_dir():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ensure_output_dir()
    print(f"加载 walkable_mask: {config.WALKABLE_MASK_PATH}")
    mask = np.load(config.WALKABLE_MASK_PATH)
    print(f"加载 target_density: {config.TARGET_DENSITY_PATH}")
    density = np.load(config.TARGET_DENSITY_PATH)

    params = {
        "num_iters": 400,
        "alpha": 0.15,
        "base_diffusivity": 1e-3,
        "clamp_min": 0.0,
        "clamp_max": None,
        "normalize": True,
        "use_potential_field": True,   # 用于可视化
        "use_distance_field": True,    # 用于导航（指向最近 sink）
    }
    out = solve_field(mask, density, **params)
    field = out["smooth_field"]
    grad_y, grad_x = out["grad"]
    score_y, score_x = out["score"]

    np.save(config.FIELD_BASELINE_PATH, field)
    np.savez(config.GRAD_BASELINE_PATH, grad_y=grad_y, grad_x=grad_x)
    np.savez(config.SCORE_BASELINE_PATH, score_y=score_y, score_x=score_x)
    
    # 保存导航场
    if "nav" in out:
        nav_y, nav_x = out["nav"]
        np.savez(config.OUTPUT_DIR / "nav_baseline.npz", nav_y=nav_y, nav_x=nav_x)
        nav_mag = np.hypot(nav_y, nav_x)
        print(f"保存导航场: nav_baseline.npz")
        print(f"导航场统计: magnitude range=[{nav_mag.min():.4f}, {nav_mag.max():.4f}], "
              f"mean={nav_mag.mean():.4f}, >0.5 ratio={(nav_mag > 0.5).mean()*100:.1f}%")
    
    # 保存势能场（用于调试）
    if "potential_field" in out:
        np.save(config.OUTPUT_DIR / "potential_field.npy", out["potential_field"])
        pot = out["potential_field"]
        print(f"保存势能场: potential_field.npy, range=[{pot.min():.4f}, {pot.max():.4f}], "
              f"mean={pot.mean():.6f}")
    
    # 保存距离场（如果有）
    if "distance_field" in out:
        np.save(config.OUTPUT_DIR / "distance_field.npy", out["distance_field"])
        print(f"保存距离场: distance_field.npy, range=[{out['distance_field'].min():.1f}, {out['distance_field'].max():.1f}]")

    print(f"保存 baseline 场: {config.FIELD_BASELINE_PATH}")
    print(f"保存梯度: {config.GRAD_BASELINE_PATH}")
    print(f"保存 score: {config.SCORE_BASELINE_PATH}")
    print(f"场统计: shape={field.shape}, min={field.min():.4f}, max={field.max():.4f}, mean={field.mean():.4f}")
    print(f"梯度范数均值: {(np.hypot(grad_y, grad_x)).mean():.4f}")

    try:
        out_img = plot_field(field, mask, score=(score_y, score_x), out_path=config.BASELINE_VIZ_PATH, title="Baseline Field")
        print(f"保存可视化: {out_img}")
    except Exception as exc:  # pragma: no cover - 可视化失败不阻塞
        print(f"可视化失败: {exc}")


if __name__ == "__main__":
    main()
