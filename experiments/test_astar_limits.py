"""
A* 极限测试：找到性能断崖点

测试矩阵：
- 网格大小: 20x20 -> 200x200
- 障碍密度: 0.2 -> 0.8
- 空间类型: euclidean, ricci, conformal
- 目标: 找到 A* 失败的临界点
"""

import numpy as np
import time
import sys
import os
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from atlas.core.registry import create_space
from atlas.core.path_planning import astar_path


def generate_maze(width: int, height: int, density: float, seed: int = 42) -> set:
    """生成迷宫"""
    random.seed(seed)
    obstacles = set()
    for x in range(width):
        obstacles.add((x, 0))
        obstacles.add((x, height - 1))
    for y in range(height):
        obstacles.add((0, y))
        obstacles.add((width - 1, y))

    for x in range(2, width - 2):
        for y in range(2, height - 2):
            if random.random() < density:
                obstacles.add((x, y))

    # 保证通路
    path_y = height // 2
    for x in range(1, width - 1):
        for dy in [-1, 0, 1]:
            obstacles.discard((x, path_y + dy))

    return obstacles


def test_astar_limits():
    """测试 A* 的极限"""
    print("=" * 80)
    print("A* LIMIT TEST")
    print("=" * 80)

    # 测试1: 网格大小极限
    print("\n--- Test 1: Grid Size Limit ---")
    for size in [20, 30, 50, 80, 100, 150, 200]:
        space = create_space('euclidean', size, size)
        obstacles = generate_maze(size, size, 0.3)
        start = (1, size // 2)
        goal = (size - 2, size // 2)

        t0 = time.time()
        path = astar_path(space, start, goal, obstacles, size, size, max_steps=50000)
        elapsed = (time.time() - t0) * 1000

        path_len = len(path) if path else 0
        print(f"  {size}x{size}: {elapsed:8.1f}ms, path={path_len:4d}, "
              f"success={path_len > 0}")

    # 测试2: 障碍密度极限
    print("\n--- Test 2: Density Limit (100x100) ---")
    for density in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        space = create_space('euclidean', 100, 100)
        obstacles = generate_maze(100, 100, density)
        start = (1, 50)
        goal = (98, 50)

        t0 = time.time()
        path = astar_path(space, start, goal, obstacles, 100, 100, max_steps=50000)
        elapsed = (time.time() - t0) * 1000

        path_len = len(path) if path else 0
        print(f"  density={density:.1f}: {elapsed:8.1f}ms, path={path_len:4d}, "
              f"success={path_len > 0}")

    # 测试3: 复杂空间极限
    print("\n--- Test 3: Complex Space Limit (50x50) ---")
    for space_name in ['euclidean', 'ricci', 'conformal', 'fisher', 'wasserstein', 'finsler']:
        for size in [20, 30, 50]:
            try:
                space = create_space(space_name, size, size)
                obstacles = generate_maze(size, size, 0.3)
                start = (1, size // 2)
                goal = (size - 2, size // 2)

                # 预热
                for i in range(5):
                    pos = (i * 5 % size, size // 2)
                    space.update_from_observation(pos, {
                        'position': pos, 'goal_position': goal, 'obstacles': list(obstacles)[:10],
                    })

                t0 = time.time()
                path = astar_path(space, start, goal, obstacles, size, size, max_steps=20000)
                elapsed = (time.time() - t0) * 1000

                path_len = len(path) if path else 0
                print(f"  {space_name:12s} {size}x{size}: {elapsed:8.1f}ms, path={path_len:4d}")
            except Exception as e:
                print(f"  {space_name:12s} {size}x{size}: ERROR - {e}")

    # 测试4: max_steps 影响
    print("\n--- Test 4: max_steps Impact (100x100) ---")
    space = create_space('euclidean', 100, 100)
    obstacles = generate_maze(100, 100, 0.3)
    start = (1, 50)
    goal = (98, 50)

    for max_steps in [100, 500, 1000, 5000, 10000, 50000]:
        t0 = time.time()
        path = astar_path(space, start, goal, obstacles, 100, 100, max_steps=max_steps)
        elapsed = (time.time() - t0) * 1000
        path_len = len(path) if path else 0
        print(f"  max_steps={max_steps:6d}: {elapsed:8.1f}ms, path={path_len:4d}, success={path_len > 0}")


if __name__ == "__main__":
    test_astar_limits()
    print("\n" + "=" * 80)
    print("A* Limit Test Complete")
    print("=" * 80)
