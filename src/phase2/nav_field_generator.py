"""
生成单 sink 导航场：为每个 sink 单独求解 Eikonal 方程。

输出：
- data/processed/nav_fields/nav_field_{sink_id}.npz: 包含 nav_y, nav_x, distance_field
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.ndimage import gaussian_filter

from src.phase2 import config
from src.phase2.baseline.pde_solver import compute_distance_field, compute_navigation_field


def load_raster_meta():
    """加载栅格元数据"""
    meta_path = config.OUTPUT_DIR / "raster_meta.json"
    with open(meta_path) as f:
        return json.load(f)


def create_single_sink_density(
    sink_row: pd.Series,
    shape: tuple,
    meta: dict,
    sigma_m: float = 800.0,  # 高斯核标准差（米）
) -> np.ndarray:
    """
    为单个 sink 创建目标密度场
    
    Args:
        sink_row: sinks DataFrame 的一行
        shape: (H, W) 栅格尺寸
        meta: 栅格元数据
        sigma_m: 高斯核标准差（米）
        
    Returns:
        density: (H, W) 密度场，sink 位置为高斯核
    """
    H, W = shape
    minx, miny, maxx, maxy = meta["bounds_proj"]
    res = meta["grid_res_m"]
    
    # 将 sink 坐标投影到 EPSG:3857
    sink_gdf = gpd.GeoDataFrame(
        [sink_row],
        geometry=gpd.points_from_xy([sink_row["lon"]], [sink_row["lat"]]),
        crs="EPSG:4326",
    ).to_crs(meta["crs"])
    
    sx = sink_gdf.geometry.x.values[0]
    sy = sink_gdf.geometry.y.values[0]
    
    # 转换为像素坐标
    px = (sx - minx) / res
    py = (maxy - sy) / res
    
    # 创建密度场
    density = np.zeros((H, W), dtype=np.float32)
    
    # 在 sink 位置放置一个点
    r, c = int(round(py)), int(round(px))
    if 0 <= r < H and 0 <= c < W:
        density[r, c] = 1.0
    
    # 高斯模糊
    sigma_pix = sigma_m / res
    density = gaussian_filter(density, sigma=sigma_pix)
    
    # 归一化
    if density.max() > 0:
        density = density / density.max()
    
    return density


def generate_single_sink_nav_field(
    sink_id: int,
    sinks: pd.DataFrame,
    walkable_mask: np.ndarray,
    meta: dict,
    output_dir: Path,
) -> dict:
    """
    为单个 sink 生成导航场
    
    Args:
        sink_id: sink 的 cluster_id
        sinks: 完整 sinks DataFrame
        walkable_mask: 可行走掩膜
        meta: 栅格元数据
        output_dir: 输出目录
        
    Returns:
        dict with nav_y, nav_x, distance_field
    """
    sink_id = int(sink_id)  # 确保是整数
    sink_row = sinks[sinks["cluster_id"] == sink_id].iloc[0]
    H, W = walkable_mask.shape
    
    # 创建单 sink 密度场
    density = create_single_sink_density(sink_row, (H, W), meta)
    
    # 计算距离场
    distance_field = compute_distance_field(density, walkable_mask, threshold=0.001)
    
    # 计算导航场
    nav_y, nav_x = compute_navigation_field(distance_field, walkable_mask)
    
    # 保存
    output_path = output_dir / f"nav_field_{sink_id:03d}.npz"
    np.savez_compressed(
        output_path,
        nav_y=nav_y,
        nav_x=nav_x,
        distance_field=distance_field,
    )
    
    return {
        "nav_y": nav_y,
        "nav_x": nav_x,
        "distance_field": distance_field,
    }


def generate_all_nav_fields(
    sinks_path: Path = None,
    output_dir: Path = None,
    verbose: bool = True,
):
    """
    为所有 sink 生成导航场
    
    Args:
        sinks_path: sinks CSV 路径
        output_dir: 输出目录
        verbose: 是否打印进度
    """
    if sinks_path is None:
        sinks_path = config.SINKS_PATH
    if output_dir is None:
        output_dir = config.OUTPUT_DIR / "nav_fields"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    meta = load_raster_meta()
    sinks = pd.read_csv(sinks_path)
    walkable_mask = np.load(config.WALKABLE_MASK_PATH)
    
    print(f"=" * 60)
    print(f"Generating navigation fields for {len(sinks)} sinks")
    print(f"Output directory: {output_dir}")
    print(f"=" * 60)
    
    for i, (_, sink_row) in enumerate(sinks.iterrows()):
        sink_id = sink_row["cluster_id"]
        
        if verbose:
            print(f"\n[{i+1}/{len(sinks)}] Sink {sink_id}: "
                  f"({sink_row['lat']:.4f}, {sink_row['lon']:.4f}), "
                  f"flow={sink_row['total_flow']:,}")
        
        result = generate_single_sink_nav_field(
            sink_id=sink_id,
            sinks=sinks,
            walkable_mask=walkable_mask,
            meta=meta,
            output_dir=output_dir,
        )
        
        if verbose:
            # 统计导航场信息
            nav_mag = np.sqrt(result["nav_y"]**2 + result["nav_x"]**2)
            valid_ratio = (nav_mag[walkable_mask > 0] > 0.5).sum() / (walkable_mask > 0).sum()
            print(f"  Valid nav ratio: {valid_ratio*100:.1f}%")
            print(f"  Distance range: [{result['distance_field'].min():.0f}, "
                  f"{result['distance_field'][result['distance_field'] < 1e6].max():.0f}]")
    
    # 保存索引文件
    index_path = output_dir / "nav_fields_index.json"
    index = {
        "num_sinks": len(sinks),
        "sink_ids": sinks["cluster_id"].tolist(),
        "shape": list(walkable_mask.shape),
        "files": [f"nav_field_{sid:03d}.npz" for sid in sinks["cluster_id"]],
    }
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    
    print(f"\n" + "=" * 60)
    print(f"Done! Generated {len(sinks)} navigation fields")
    print(f"Index saved to {index_path}")


def load_nav_field(sink_id: int, nav_fields_dir: Path = None) -> dict:
    """
    加载单个 sink 的导航场
    
    Args:
        sink_id: sink 的 cluster_id
        nav_fields_dir: 导航场目录
        
    Returns:
        dict with nav_y, nav_x, distance_field
    """
    if nav_fields_dir is None:
        nav_fields_dir = config.OUTPUT_DIR / "nav_fields"
    
    path = nav_fields_dir / f"nav_field_{sink_id:03d}.npz"
    data = np.load(path)
    
    return {
        "nav_y": data["nav_y"],
        "nav_x": data["nav_x"],
        "distance_field": data["distance_field"],
    }


class NavFieldManager:
    """
    导航场管理器：按需加载和缓存导航场
    """
    
    def __init__(self, nav_fields_dir: Path = None, cache_size: int = 10):
        """
        Args:
            nav_fields_dir: 导航场目录
            cache_size: LRU 缓存大小
        """
        if nav_fields_dir is None:
            nav_fields_dir = config.OUTPUT_DIR / "nav_fields"
        
        self.nav_fields_dir = nav_fields_dir
        self.cache_size = cache_size
        self._cache = {}
        self._access_order = []
        
        # 加载索引
        index_path = nav_fields_dir / "nav_fields_index.json"
        with open(index_path) as f:
            self.index = json.load(f)
        
        self.sink_ids = set(self.index["sink_ids"])
    
    def get(self, sink_id: int) -> dict:
        """
        获取指定 sink 的导航场（带 LRU 缓存）
        
        Args:
            sink_id: sink 的 cluster_id
            
        Returns:
            dict with nav_y, nav_x, distance_field
        """
        if sink_id not in self.sink_ids:
            raise ValueError(f"Unknown sink_id: {sink_id}. Valid IDs: {sorted(self.sink_ids)}")
        
        # 检查缓存
        if sink_id in self._cache:
            # 更新访问顺序
            self._access_order.remove(sink_id)
            self._access_order.append(sink_id)
            return self._cache[sink_id]
        
        # 加载
        nav_field = load_nav_field(sink_id, self.nav_fields_dir)
        
        # 缓存管理
        if len(self._cache) >= self.cache_size:
            # 驱逐最久未访问的
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        
        self._cache[sink_id] = nav_field
        self._access_order.append(sink_id)
        
        return nav_field
    
    def get_nav_direction(self, sink_id: int, py: int, px: int) -> tuple:
        """
        获取指定位置的导航方向
        
        Args:
            sink_id: 目标 sink ID
            py, px: 像素坐标
            
        Returns:
            (nav_y, nav_x) 导航方向
        """
        nav_field = self.get(sink_id)
        H, W = nav_field["nav_y"].shape
        
        # 边界检查
        py = np.clip(py, 0, H - 1)
        px = np.clip(px, 0, W - 1)
        
        return nav_field["nav_y"][py, px], nav_field["nav_x"][py, px]
    
    def get_distance(self, sink_id: int, py: int, px: int) -> float:
        """
        获取到指定 sink 的距离
        
        Args:
            sink_id: 目标 sink ID
            py, px: 像素坐标
            
        Returns:
            distance: 距离（像素单位）
        """
        nav_field = self.get(sink_id)
        H, W = nav_field["distance_field"].shape
        
        py = np.clip(py, 0, H - 1)
        px = np.clip(px, 0, W - 1)
        
        return nav_field["distance_field"][py, px]


if __name__ == "__main__":
    generate_all_nav_fields()
