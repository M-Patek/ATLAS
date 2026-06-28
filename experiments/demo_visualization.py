"""
演示: ATLAS 可视化工具

展示如何使用可视化模块分析空间和路径
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from atlas.core.registry import create_space
from atlas.core import GeodesicSolver, AdaptiveNavigator

print("=" * 70)
print("ATLAS Visualization Demo")
print("=" * 70)
print()

# 创建不同空间
print("Creating cognitive spaces...")
spaces = {
    'euclidean': create_space('euclidean', width=30, height=20),
    'ricci': create_space('ricci', width=30, height=20, curvature_scale=1.5),
    'conformal': create_space('conformal', width=30, height=20),
}

# 场景设置
start = (5, 10)
goal = (25, 10)
obstacles = {(15, y) for y in range(5, 16) if y != 10}  # 中间有缺口

# 更新空间
for name, space in spaces.items():
    space.update_from_observation(start, {'obstacles': list(obstacles)})
    space.update_from_observation(goal, {'goal_position': goal})

print(f"Spaces: {list(spaces.keys())}")
print(f"Start: {start}, Goal: {goal}")
print(f"Obstacles: {len(obstacles)} cells")
print()

# 规划路径
print("Planning paths...")
paths = {}
for name, space in spaces.items():
    solver = GeodesicSolver(space)
    result = solver.solve(start, goal, obstacles)
    if result.success:
        paths[name] = result.path
        print(f"  {name:12s}: {len(result.path)} steps, cost={result.cost:.2f}")

print()

# 尝试可视化（如果matplotlib可用）
try:
    from atlas.visualization import SpaceVisualizer, ComparisonPlotter

    print("Generating visualizations...")

    viz = SpaceVisualizer()

    # 1. 可视化单个空间
    print("  - Ricci space visualization")
    viz.visualize_space(
        spaces['ricci'],
        fields=['uncertainty', 'curvature', 'familiarity'],
        overlay={'path': paths['ricci'], 'obstacles': obstacles, 'goal': goal},
        save_path='outputs/ricci_space.png',
        show=False
    )

    # 2. 对比多个空间
    print("  - Space comparison")
    viz.compare_spaces(
        list(spaces.values()),
        field='metric',
        titles=list(spaces.keys()),
        save_path='outputs/space_comparison.png',
        show=False
    )

    # 3. 性能对比
    print("  - Performance plot")
    plotter = ComparisonPlotter()

    # 模拟多次运行结果
    results = {}
    for name in spaces.keys():
        results[name] = [
            {'steps': len(paths[name]), 'time_ms': np.random.uniform(1, 10)}
            for _ in range(10)
        ]

    plotter.plot_performance_comparison(
        results,
        metrics=['steps', 'time_ms'],
        save_path='outputs/performance_comparison.png',
        show=False
    )

    print()
    print("Visualizations saved to outputs/")

except ImportError as e:
    print(f"Visualization skipped: {e}")
    print("Install matplotlib/seaborn to enable visualization")

print()
print("=" * 70)
print("Demo completed")
print("=" * 70)
