"""
ATLAS Core Example: Pluggable Space Comparison
可插拔空间对比实验示例

展示如何使用核心框架进行对比实验
"""

import numpy as np
import sys
import os

# 添加 src 到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.core import Experiment, GeodesicSolver
from atlas.core.registry import create_space, list_available_spaces
from atlas.spaces import *  # 导入所有空间实现


def create_test_scenario(seed: int = 42):
    """创建测试场景"""
    np.random.seed(seed)

    # 创建障碍物
    obstacles = set()
    mid_x = 20
    for y in range(3, 17):
        if y != 10:  # 留出通道
            obstacles.add((mid_x, y))

    return {
        'start': (5, 10),
        'goal': (35, 10),
        'obstacles': obstacles,
        'observations': [
            {'position': (10, 10), 'data': {'obstacles': [(mid_x, 5), (mid_x, 15)]}},
            {'position': (25, 10), 'data': {'goal_position': (35, 10)}},
        ]
    }


def run_comparison_experiment():
    """运行对比实验"""
    print("=" * 70)
    print("ATLAS Core: Pluggable Space Comparison")
    print("=" * 70)
    print()

    # 列出可用空间
    print("Available cognitive spaces:")
    spaces = list_available_spaces()
    for name, desc in spaces.items():
        print(f"  - {name}: {desc}")
    print()

    # 创建实验
    experiment = Experiment(name="space_comparison")

    # 注册空间（使用工厂函数）
    width, height = 40, 20

    space_configs = [
        ("euclidean", {}),
        ("ricci", {"curvature_scale": 2.0}),
        ("conformal", {"base_scale": 1.0}),
        ("fisher", {"temperature": 1.0}),
        ("wasserstein", {"base_cost": 1.0}),
        ("finsler", {"asymmetry": 0.5}),
    ]

    for name, kwargs in space_configs:
        try:
            space = create_space(name, width, height, **kwargs)
            experiment.register_space(name, space)
            print(f"Registered: {name}")
        except Exception as e:
            print(f"Failed to register {name}: {e}")

    # 注册求解器
    experiment.register_solver("astar", GeodesicSolver(None))  # space 会在运行时设置

    # 添加测试场景
    for i in range(3):
        scenario = create_test_scenario(seed=42 + i)
        experiment.add_scenario(scenario)

    print()
    print("Running experiment...")
    print()

    # 运行实验
    def progress_callback(name, current, total):
        print(f"  [{current+1}/{total}] {name}")

    results = experiment.run(num_trials=5, progress_callback=progress_callback)

    # 显示结果
    print()
    print(experiment.get_summary())

    # 空间排名
    print()
    print("=" * 70)
    print("Space Rankings")
    print("=" * 70)

    print("\nBy Success Rate:")
    for name, rate in experiment.compare_spaces('success_rate'):
        print(f"  {name:15s}: {rate:.1%}")

    print("\nBy Mean Steps (lower is better):")
    for name, steps in sorted(experiment.compare_spaces('mean_steps'),
                              key=lambda x: x[1]):
        print(f"  {name:15s}: {steps:.1f}")

    print("\nBy Path Cost:")
    for name, cost in sorted(experiment.compare_spaces('mean_cost'),
                            key=lambda x: x[1]):
        print(f"  {name:15s}: {cost:.1f}")

    return results


def run_ablation_example():
    """运行消融实验示例"""
    print()
    print("=" * 70)
    print("Ablation Study: Ricci Curvature Scale")
    print("=" * 70)
    print()

    from atlas.core.experiment import AblationStudy

    # 基线配置
    base_config = {
        'space_type': 'ricci',
        'width': 40,
        'height': 20,
        'curvature_scale': 1.0,
    }

    study = AblationStudy(base_config)

    # 添加变体
    study.add_variation("no_curvature", {'curvature_scale': 0.0})
    study.add_variation("high_curvature", {'curvature_scale': 3.0})
    study.add_variation("very_high", {'curvature_scale': 5.0})

    # 测试函数
    def test_config(config):
        from atlas.core.registry import create_space
        from atlas.core.solver import GeodesicSolver

        space = create_space(
            config['space_type'],
            config['width'],
            config['height'],
            curvature_scale=config.get('curvature_scale', 1.0)
        )

        solver = GeodesicSolver(space)

        # 简单测试
        scenario = create_test_scenario()
        result = solver.solve(
            scenario['start'],
            scenario['goal'],
            scenario['obstacles']
        )

        # 返回性能分数（成功率 × 效率）
        if result.success:
            euclidean = np.sqrt(
                (scenario['goal'][0] - scenario['start'][0])**2 +
                (scenario['goal'][1] - scenario['start'][1])**2
            )
            efficiency = euclidean / len(result.path) if result.path else 0
            return efficiency
        return 0.0

    # 运行消融研究
    print("Running ablation study...")
    results = study.run(test_fn=test_config, num_runs=10)

    # 分析结果
    print(study.analyze(results))


if __name__ == "__main__":
    # 运行对比实验
    results = run_comparison_experiment()

    # 运行消融实验
    run_ablation_example()

    print()
    print("=" * 70)
    print("Experiment Complete!")
    print("=" * 70)
