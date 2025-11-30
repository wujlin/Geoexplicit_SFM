import numpy as np
import matplotlib.pyplot as plt

from src.phase2 import config


def inspect():
    print(f"Loading density from: {config.TARGET_DENSITY_PATH}")
    if not config.TARGET_DENSITY_PATH.exists():
        print("Error: 目标密度文件不存在，请先运行 geo_rasterizer 生成。")
        return

    density = np.load(config.TARGET_DENSITY_PATH)
    mask = np.load(config.WALKABLE_MASK_PATH)

    print(f"Density shape: {density.shape}")
    print(f"Density range: [{density.min():.6f}, {density.max():.6f}], mean={density.mean():.6f}")
    print(f"Mask range: [{mask.min()}, {mask.max()}]")

    if density.max() < 1e-6:
        print("\n[CRITICAL] 目标密度几乎全为 0，可能是投影或 bbox 出错。")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].set_title("Target Density (log1p)")
    im0 = axes[0].imshow(np.log1p(density), cmap="magma", origin="lower")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].set_title("Walkable Mask")
    axes[1].imshow(mask, cmap="gray", origin="lower")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    save_path = config.OUTPUT_DIR / "data_inspection.png"
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"诊断图已保存: {save_path}")
    print("请检查 density 是否有明显光斑、mask 是否有清晰路网。")


if __name__ == "__main__":
    inspect()
