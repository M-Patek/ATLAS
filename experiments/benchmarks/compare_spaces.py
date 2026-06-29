"""
Experiment: Compare Cognitive Spaces
对比不同认知空间的性能
"""

import numpy as np
import sys
import os

# 添加 src 到路�?sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core import Experiment, GeodesicSolver
from src.core.registry import create_space, list_available_spaces


def create_maze_scenario(seed: int = 42):
    """创建迷宫测试场景"""
    np.random.seed(seed)

    # 简单的墙障�?    obstacles = set()
    for y in range(3, 17):
        if y != 10:  # 留出通道
            obstacles.add((20, y))

    return {
        'start': (5, 10),
        'goal': (35, 10),
        'obstacles': obstacles,
        'observations': [
            {'position': (10, 10), 'data': {'obstacles': [(20, 8), (20, 12)]}},
            {'position': (25, 10), 'data': {'goal_position': (35, 10)}},
        ]
    }


def main():
    print("=" * 70)
    print("ATLAS: Cognitive Space Comparison")
    print("=" * 70)
    print()

    # 显示可用空间
    print("Available cognitive spaces:")
    try:
        spaces = list_available_spaces()
        for name, desc in spaces.items():
            print(f"  - {name}: {desc}")
    except Exception as e:
        print(f"  (Registry not fully initialized: {e})")
    print()

    # 手动测试几个核心空间
    print("Testing core spaces...")
    print()

    width, height = 40, 20

    # 测试场景
    scenario = create_maze_scenario()

    # 测试的空间配�?    space_configs = [
        ("euclidean", {}),
        ("ricci", {"curvature_scale": 1.5}),
        ("conformal", {}),
    ]

    results = []

    for space_name, kwargs in space_configs:
        try:
            print(f"  Testing {space_name}...")

            # 创建空间
            space = create_space(space_name, width, height, **kwargs)

            # 应用观测
            for obs in scenario['observations']:
                space.update_from_observation(obs['position'], obs['data'])

            # 创建求解�?            solver = GeodesicSolver(space)

            # 求解
            result = solver.solve(
                scenario['start'],
                scenario['goal'],
                scenario['obstacles']
            )

            # 记录结果
            results.append({
                'name': space_name,
                'success': result.success,
                'steps': len(result.path) if result.path else 0,
                'cost': result.cost,
                'time_ms': result.time_ms,
                'nodes': result.nodes_expanded
            })

            print(f"    Success: {result.success}, Steps: {len(result.path) if result.path else 0}")

        except Exception as e:
            print(f"    Error: {e}")
            results.append({
                'name': space_name,
                'success': False,
                'error': str(e)
            })

    # 汇�?    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()

    print(f"{'Space':<15} {'Success':<10} {'Steps':<10} {'Cost':<12} {'Time(ms)':<12}")
    print("-" * 70)

    for r in results:
        if 'error' in r:
            print(f"{r['name']:<15} ERROR: {r['error'][:40]}")
        else:
            print(f"{r['name']:<15} {str(r['success']):<10} {r['steps']:<10} "
                  f"{r['cost']:<12.2f} {r['time_ms']:<12.2f}")

    print()


if __name__ == "__main__":
    main()
