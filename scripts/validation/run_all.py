"""
统一验证流程入口
运行所有 Phase 的验证并生成综合报告
"""

import json
import argparse
from datetime import datetime
from pathlib import Path

from . import get_path_config


def run_all_validations(
    phases: list = None,
    save_figures: bool = True,
    n_phase4_samples: int = 50000
):
    """
    运行所有 Phase 的验证
    
    Args:
        phases: 要验证的 Phase 列表，默认 [1, 2, 3, 4]
        save_figures: 是否保存可视化图
        n_phase4_samples: Phase 4 验证采样数
    """
    if phases is None:
        phases = [1, 2, 3, 4]
    
    paths = get_path_config()
    paths.ensure_dirs()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "phases": {},
    }
    
    # Phase 1
    if 1 in phases:
        print("\n" + "=" * 60)
        print("Running Phase 1 Validation...")
        print("=" * 60)
        from .validate_phase1 import validate_phase1
        metrics = validate_phase1(paths, save_figure=save_figures)
        results["phases"]["phase1"] = {
            "num_sinks": metrics.num_sinks,
            "total_sinks_flow": metrics.total_sinks_flow,
            "population_covered_ratio": metrics.population_covered_ratio,
            "spatial_coverage": metrics.spatial_coverage,
        }
    
    # Phase 2
    if 2 in phases:
        print("\n" + "=" * 60)
        print("Running Phase 2 Validation...")
        print("=" * 60)
        from .validate_phase2 import validate_phase2
        metrics = validate_phase2(paths, save_figure=save_figures)
        results["phases"]["phase2"] = {
            "grid_shape": metrics.grid_shape,
            "walkable_pixels": metrics.walkable_pixels,
            "walkable_ratio": metrics.walkable_ratio,
            "num_nav_fields": metrics.num_nav_fields,
            "mean_coverage": float(sum(metrics.nav_field_coverage.values()) / len(metrics.nav_field_coverage)),
        }
    
    # Phase 3
    if 3 in phases:
        print("\n" + "=" * 60)
        print("Running Phase 3 Validation...")
        print("=" * 60)
        from .validate_phase3 import validate_phase3
        metrics = validate_phase3(paths, save_figure=save_figures)
        results["phases"]["phase3"] = {
            "n_agents": metrics.n_agents,
            "n_arrived": metrics.n_arrived,
            "arrival_rate": metrics.arrival_rate,
            "mean_steps": metrics.mean_steps,
            "mean_speed": metrics.mean_speed,
            "od_pearson_r": metrics.od_pearson_r,
            "od_spearman_r": metrics.od_spearman_r,
        }
    
    # Phase 4
    if 4 in phases:
        print("\n" + "=" * 60)
        print("Running Phase 4 Validation...")
        print("=" * 60)
        from .validate_phase4 import validate_phase4
        metrics = validate_phase4(paths, save_figure=save_figures, n_samples=n_phase4_samples)
        results["phases"]["phase4"] = {
            "n_samples": metrics.n_samples,
            "mean_cos_sim_gt": metrics.mean_cos_sim_gt,
            "std_cos_sim_gt": metrics.std_cos_sim_gt,
            "mean_cos_sim_nav": metrics.mean_cos_sim_nav,
            "direction_correct_ratio": metrics.direction_correct_ratio,
            "aligned_ratio": metrics.aligned_ratio,
        }
    
    # 生成综合报告
    report_path = paths.reports_dir / "validation_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary report: {report_path}")
    
    # 打印综合摘要
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if "phase1" in results["phases"]:
        p1 = results["phases"]["phase1"]
        print(f"\nPhase 1 - Sink Identification:")
        print(f"  {p1['num_sinks']} sinks, {p1['population_covered_ratio']*100:.1f}% population covered")
    
    if "phase2" in results["phases"]:
        p2 = results["phases"]["phase2"]
        print(f"\nPhase 2 - Navigation Fields:")
        print(f"  {p2['num_nav_fields']} fields, {p2['mean_coverage']*100:.1f}% avg coverage")
    
    if "phase3" in results["phases"]:
        p3 = results["phases"]["phase3"]
        print(f"\nPhase 3 - Trajectory Simulation:")
        print(f"  {p3['arrival_rate']*100:.1f}% arrival rate, OD r={p3['od_pearson_r']:.3f}")
    
    if "phase4" in results["phases"]:
        p4 = results["phases"]["phase4"]
        print(f"\nPhase 4 - Diffusion Policy:")
        print(f"  cos_sim={p4['mean_cos_sim_gt']:.3f}, {p4['direction_correct_ratio']*100:.1f}% correct direction")
    
    # 质量评估
    print("\n" + "-" * 60)
    print("Quality Assessment:")
    
    all_good = True
    if "phase3" in results["phases"]:
        if results["phases"]["phase3"]["od_pearson_r"] < 0.7:
            print("  ⚠️ Phase 3: OD correlation below 0.7")
            all_good = False
    
    if "phase4" in results["phases"]:
        if results["phases"]["phase4"]["mean_cos_sim_gt"] < 0.6:
            print("  ⚠️ Phase 4: Prediction cos_sim below 0.6")
            all_good = False
    
    if all_good:
        print("  ✅ All phases passed quality checks")
    
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run validation for all phases")
    parser.add_argument("--phases", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="Phases to validate (default: 1 2 3 4)")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip saving visualization figures")
    parser.add_argument("--phase4-samples", type=int, default=50000,
                        help="Number of samples for Phase 4 validation")
    
    args = parser.parse_args()
    
    run_all_validations(
        phases=args.phases,
        save_figures=not args.no_figures,
        n_phase4_samples=args.phase4_samples
    )


if __name__ == "__main__":
    main()
