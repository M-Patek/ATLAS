"""
结构池压力测试：暴露无限增长的性能退化

设计：
1. 长回合（1000+ 步）
2. 每步都调用 perceive
3. 观察结构池大小和步时间的关系
4. 找到性能退化的临界点
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Set
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from atlas.core.ssfr_enhanced import SSFREnhanced


def benchmark_pool_growth(max_steps: int = 1000, grid_size: int = 30):
    """测试结构池增长对性能的影响"""
    print(f"\n{'='*80}")
    print(f"Structure Pool Growth Test: {grid_size}x{grid_size}, {max_steps} steps")
    print(f"{'='*80}")

    ssfr = SSFREnhanced(
        width=grid_size, height=grid_size,
        space_names=['ricci', 'fisher', 'wasserstein', 'conformal', 'finsler'],
        max_structures=1000,
        evolution_interval=10
    )

    results = []
    for i in range(max_steps):
        pos = (i % grid_size, (i * 3) % grid_size)
        obs = {
            'position': pos,
            'goal_position': (grid_size - 2, grid_size // 2),
            'obstacles': [(pos[0] + dx, pos[1] + dy)
                         for dx in range(-2, 3) for dy in range(-2, 3)
                         if (dx**2 + dy**2) < 5 and (dx, dy) != (0, 0)],
        }

        start = time.time()
        ssfr.perceive(pos, obs, active_space_name='ricci')
        elapsed = (time.time() - start) * 1000

        pool_size = len(ssfr.structure_pool.structures)

        if i % 100 == 0 or i < 20:
            results.append({
                'step': i,
                'pool_size': pool_size,
                'perceive_ms': elapsed,
            })
            print(f"  Step {i:4d}: pool_size={pool_size:4d}, perceive={elapsed:6.2f}ms")

    # 分析增长趋势
    print(f"\n--- Growth Analysis ---")
    print(f"  Initial pool size: {results[0]['pool_size']}")
    print(f"  Final pool size: {results[-1]['pool_size']}")
    print(f"  Growth rate: {(results[-1]['pool_size'] - results[0]['pool_size']) / max_steps:.2f} structures/step")

    # 分析性能退化
    early_times = [r['perceive_ms'] for r in results if r['step'] < 100]
    late_times = [r['perceive_ms'] for r in results if r['step'] >= max_steps - 100]
    print(f"\n--- Performance Degradation ---")
    print(f"  Early avg (steps 0-99): {np.mean(early_times):.2f}ms")
    print(f"  Late avg (last 100): {np.mean(late_times):.2f}ms")
    if np.mean(early_times) > 0:
        print(f"  Degradation factor: {np.mean(late_times) / np.mean(early_times):.2f}x")

    return results


def benchmark_pool_operations(grid_size: int = 50):
    """测试结构池操作（add, get_best, compete）的性能"""
    print(f"\n{'='*80}")
    print(f"Pool Operations Benchmark: {grid_size}x{grid_size}")
    print(f"{'='*80}")

    ssfr = SSFREnhanced(
        width=grid_size, height=grid_size,
        space_names=['ricci', 'fisher', 'wasserstein'],
        max_structures=500,
        evolution_interval=10
    )

    # 先填充结构池
    print("  Filling pool...")
    for i in range(500):
        pos = (i % grid_size, (i * 3) % grid_size)
        obs = {
            'position': pos,
            'goal_position': (grid_size - 2, grid_size // 2),
            'obstacles': [],
        }
        ssfr.perceive(pos, obs, active_space_name='ricci')

    pool_size = len(ssfr.structure_pool.structures)
    print(f"  Pool size: {pool_size}")

    # 测试 get_best
    print("\n  Testing get_best_structures()...")
    times = []
    for n in [1, 5, 10, 20, 50, 100]:
        t = []
        for _ in range(100):
            start = time.time()
            ssfr.structure_pool.get_best(n)
            t.append((time.time() - start) * 1000)
        print(f"    get_best(n={n:3d}): avg={np.mean(t):.3f}ms, max={np.max(t):.3f}ms")
        times.append(np.mean(t))

    # 测试 compete
    print("\n  Testing compete()...")
    obs = {'position': (10, 10), 'goal_position': (40, 25), 'obstacles': []}
    actual = {'position': (11, 10)}
    t = []
    for _ in range(100):
        start = time.time()
        ssfr.structure_pool.compete(obs, actual, timestamp=0)
        t.append((time.time() - start) * 1000)
    print(f"    compete(): avg={np.mean(t):.3f}ms, max={np.max(t):.3f}ms")

    # 测试 evolve
    print("\n  Testing evolve()...")
    t = []
    for _ in range(100):
        start = time.time()
        ssfr.structure_pool.evolve(timestamp=0)
        t.append((time.time() - start) * 1000)
    print(f"    evolve(): avg={np.mean(t):.3f}ms, max={np.max(t):.3f}ms")


if __name__ == "__main__":
    # 测试1: 结构池增长
    benchmark_pool_growth(max_steps=1000, grid_size=30)

    # 测试2: 结构池操作性能
    benchmark_pool_operations(grid_size=50)

    print(f"\n{'='*80}")
    print("Pool Pressure Test Complete")
    print(f"{'='*80}")
