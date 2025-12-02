"""
全流程验证脚本 - 一键运行 Phase 1-4 所有验证

用法:
    python scripts/validate_all.py
    python scripts/validate_all.py --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import json
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from validate_phase1 import validate_phase1
from validate_phase2 import validate_phase2
from validate_phase3 import validate_phase3


def main():
    parser = argparse.ArgumentParser(description="Run all phase validations")
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer samples")
    parser.add_argument("--skip-phase4", action="store_true", help="Skip Phase 4 (slow)")
    args = parser.parse_args()
    
    output_dir = PROJECT_ROOT / "data" / "output" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    all_results = {}
    
    print("\n" + "="*70)
    print(" GeoExplicit-SFM 全流程验证")
    print("="*70)
    
    # Phase 1
    print("\n" + "-"*70)
    r1 = validate_phase1(output_dir)
    if r1:
        all_results["phase1"] = r1
    
    # Phase 2
    print("\n" + "-"*70)
    num_samples_p2 = 5000 if args.quick else 10000
    r2 = validate_phase2(output_dir, num_samples=num_samples_p2)
    if r2:
        all_results["phase2"] = r2
    
    # Phase 3
    print("\n" + "-"*70)
    sample_ratio = 0.001 if args.quick else 0.01
    r3 = validate_phase3(output_dir, sample_ratio=sample_ratio)
    if r3:
        all_results["phase3"] = r3
    
    # Phase 4 (可选)
    if not args.skip_phase4:
        print("\n" + "-"*70)
        try:
            from validate_phase4 import Phase4Validator
            
            checkpoint_path = PROJECT_ROOT / "data" / "output" / "phase4_checkpoints" / "best.pt"
            if checkpoint_path.exists():
                validator = Phase4Validator(checkpoint_path)
                
                num_pred = 2000 if args.quick else 10000
                num_cond = 30 if args.quick else 100
                num_traj = 20 if args.quick else 50
                max_steps = 100 if args.quick else 200
                
                validator.evaluate_prediction_quality(num_samples=num_pred)
                validator.evaluate_condition_response(num_tests=num_cond)
                validator.evaluate_trajectory_quality(num_agents=num_traj, max_steps=max_steps)
                validator.generate_report(output_dir / "phase4")
                
                all_results["phase4"] = validator.results
            else:
                print(f"  跳过 Phase 4: 模型文件不存在")
        except Exception as e:
            print(f"  Phase 4 验证出错: {e}")
    
    # 汇总
    print("\n" + "="*70)
    print(" 验证汇总")
    print("="*70)
    
    summary = []
    for phase in ["phase1", "phase2", "phase3", "phase4"]:
        if phase in all_results:
            passed = all_results[phase].get("validation_passed", "N/A")
            if passed is True:
                status = "✅ 通过"
            elif passed is False:
                status = "❌ 失败"
            else:
                status = "⚠️ 未知"
            summary.append(f"  {phase.upper()}: {status}")
    
    print("\n".join(summary))
    
    # 保存汇总
    summary_path = output_dir / "validation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n汇总保存至: {summary_path}")
    
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.1f} 秒")


if __name__ == "__main__":
    main()
