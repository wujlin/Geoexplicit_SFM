"""
统一验证框架：Phase 1-4 全流程验证与可视化

设计原则：
1. 统一的输出路径管理
2. 一致的接口设计
3. 可复用的验证函数
4. 自动生成报告和可视化

输出结构：
data/output/validation/
├── reports/
│   ├── phase1_report.json
│   ├── phase2_report.json
│   ├── phase3_report.json
│   ├── phase4_report.json
│   └── full_report.json
├── figures/
│   ├── phase1_sinks.png
│   ├── phase2_nav_fields.png
│   ├── phase3_trajectories.png
│   ├── phase3_od_validation.png
│   ├── phase4_training_curve.png
│   ├── phase4_inference.png
│   └── summary_dashboard.png
└── metrics/
    ├── phase1_metrics.npz
    ├── phase2_metrics.npz
    ├── phase3_metrics.npz
    └── phase4_metrics.npz
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np


# ============================================================
# 路径管理
# ============================================================

@dataclass
class PathConfig:
    """统一路径配置"""
    base_dir: Path
    
    @property
    def project_root(self) -> Path:
        return self.base_dir
    
    # 输入数据
    @property
    def sinks_csv(self) -> Path:
        return self.base_dir / "data" / "processed" / "sinks_phase1.csv"
    
    @property
    def walkable_mask(self) -> Path:
        return self.base_dir / "data" / "processed" / "walkable_mask.npy"
    
    @property
    def nav_baseline(self) -> Path:
        return self.base_dir / "data" / "processed" / "nav_baseline.npz"
    
    @property
    def nav_fields_dir(self) -> Path:
        return self.base_dir / "data" / "processed" / "nav_fields"
    
    @property
    def trajectories_h5(self) -> Path:
        return self.base_dir / "data" / "output" / "trajectories.h5"
    
    @property
    def valid_indices(self) -> Path:
        return self.base_dir / "data" / "output" / "valid_indices.npy"
    
    @property
    def checkpoint_dir(self) -> Path:
        return self.base_dir / "data" / "output" / "phase4_checkpoints"
    
    @property
    def best_checkpoint(self) -> Path:
        return self.checkpoint_dir / "best.pt"
    
    # OD 数据
    @property
    def od_matrix(self) -> Path:
        return self.base_dir / "data" / "processed" / "sink_od_matrix.csv"
    
    @property
    def od_prob(self) -> Path:
        return self.base_dir / "data" / "processed" / "sink_od_prob.csv"
    
    @property
    def od_real(self) -> Path:
        return self.base_dir / "dataset" / "od_flow" / "mi-tract-od-2020.csv"
    
    # 输出目录
    @property
    def validation_dir(self) -> Path:
        return self.base_dir / "data" / "output" / "validation"
    
    @property
    def reports_dir(self) -> Path:
        return self.validation_dir / "reports"
    
    @property
    def figures_dir(self) -> Path:
        return self.validation_dir / "figures"
    
    @property
    def metrics_dir(self) -> Path:
        return self.validation_dir / "metrics"
    
    def ensure_dirs(self):
        """创建所有输出目录"""
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)


def get_path_config() -> PathConfig:
    """获取路径配置（自动检测项目根目录）"""
    # 尝试从当前文件位置推断
    current = Path(__file__).resolve()
    
    # 向上查找包含 'src' 目录的父目录
    for parent in current.parents:
        if (parent / "src").exists() and (parent / "data").exists():
            return PathConfig(base_dir=parent)
    
    # 回退到当前工作目录
    return PathConfig(base_dir=Path.cwd())


# ============================================================
# 验证结果数据类
# ============================================================

@dataclass
class Phase1Metrics:
    """Phase 1 验证指标"""
    num_sinks: int
    total_sinks_flow: int
    population_covered_ratio: float
    spatial_coverage: float
    lat_range: tuple = None
    lon_range: tuple = None
    flow_distribution: List[int] = None  # 每个 sink 的流量
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Phase2Metrics:
    """Phase 2 验证指标"""
    grid_shape: tuple
    walkable_pixels: int
    walkable_ratio: float
    num_nav_fields: int
    nav_field_coverage: Dict[int, float]  # sink_id -> 有效方向覆盖率
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Phase3Metrics:
    """Phase 3 验证指标"""
    n_agents: int
    n_arrived: int
    arrival_rate: float
    mean_steps: float
    mean_speed: float
    std_speed: float
    od_pearson_r: float
    od_spearman_r: float
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Phase4Metrics:
    """Phase 4 验证指标"""
    n_samples: int
    mean_cos_sim_gt: float
    std_cos_sim_gt: float
    mean_cos_sim_nav: float
    std_cos_sim_nav: float
    direction_correct_ratio: float  # >0
    aligned_ratio: float  # >0.5
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationReport:
    """完整验证报告"""
    timestamp: str
    phase1: Optional[Phase1Metrics] = None
    phase2: Optional[Phase2Metrics] = None
    phase3: Optional[Phase3Metrics] = None
    phase4: Optional[Phase4Metrics] = None
    
    def to_dict(self) -> dict:
        result = {"timestamp": self.timestamp}
        if self.phase1:
            result["phase1"] = self.phase1.to_dict()
        if self.phase2:
            result["phase2"] = self.phase2.to_dict()
        if self.phase3:
            result["phase3"] = self.phase3.to_dict()
        if self.phase4:
            result["phase4"] = self.phase4.to_dict()
        return result
    
    def save(self, path: Path):
        """保存报告为 JSON"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Path) -> "ValidationReport":
        """从 JSON 加载报告"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        report = cls(timestamp=data["timestamp"])
        if "phase1" in data:
            report.phase1 = Phase1Metrics(**data["phase1"])
        if "phase2" in data:
            report.phase2 = Phase2Metrics(**data["phase2"])
        if "phase3" in data:
            report.phase3 = Phase3Metrics(**data["phase3"])
        if "phase4" in data:
            report.phase4 = Phase4Metrics(**data["phase4"])
        return report


# ============================================================
# 工具函数
# ============================================================

def load_nav_fields(nav_fields_dir: Path) -> Dict[int, np.ndarray]:
    """加载所有导航场"""
    index_path = nav_fields_dir / "nav_fields_index.json"
    with open(index_path) as f:
        index = json.load(f)
    
    nav_fields = {}
    for sink_id in index["sink_ids"]:
        path = nav_fields_dir / f"nav_field_{sink_id:03d}.npz"
        data = np.load(path)
        nav_fields[sink_id] = np.stack([data["nav_y"], data["nav_x"]], axis=0)
    
    return nav_fields


def compute_cos_sim(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-6) -> float:
    """计算两个向量的余弦相似度"""
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if mag1 < eps or mag2 < eps:
        return 0.0
    return float(np.dot(v1 / mag1, v2 / mag2))
