"""
瓶颈确认测试：验证首次初始化的开销来源

测试：
1. 测量首次 update_from_observation 的时间
2. 对比：有/无障碍物时的差异
3. 对比：不同空间类型的初始化开销
"""

import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from atlas.core.registry import create_space


def benchmark_first_update():
    """测试首次更新的开销"""
    print("=" * 80)
    print("FIRST UPDATE BENCHMARK")
    print("=" * 80)

    # 测试1: 不同空间类型的首次更新
    print("\n--- Test 1: First Update Overhead by Space ---")
    for space_name in ['euclidean', 'ricci', 'conformal', 'fisher', 'wasserstein', 'finsler']:
        for size in [20, 30, 50, 80]:
            try:
                space = create_space(space_name, size, size)

                # 生成障碍物
                import random
                random.seed(42)
                obstacles = []
                for x in range(size):
                    for y in range(size):
                        if random.random() < 0.3:
                            obstacles.append((x, y))

                obs = {
                    'position': (1, size // 2),
                    'goal_position': (size - 2, size // 2),
                    'obstacles': obstacles,
                }

                # 首次更新
                t0 = time.time()
                space.update_from_observation((1, size // 2), obs)
                first_ms = (time.time() - t0) * 1000

                # 第二次更新（应该快很多）
                t0 = time.time()
                space.update_from_observation((2, size // 2), obs)
                second_ms = (time.time() - t0) * 1000

                print(f"  {space_name:12s} {size}x{size}: "
                      f"first={first_ms:7.1f}ms, second={second_ms:6.2f}ms, "
                      f"ratio={first_ms/max(0.1,second_ms):5.1f}x")
            except Exception as e:
                print(f"  {space_name:12s} {size}x{size}: ERROR - {e}")

    # 测试2: 障碍物数量对首次更新的影响
    print("\n--- Test 2: Obstacle Count Impact (Ricci, 50x50) ---")
    for num_obstacles in [0, 10, 50, 100, 200, 500, 1000]:
        space = create_space('ricci', 50, 50)

        import random
        random.seed(42)
        obstacles = []
        for _ in range(num_obstacles):
            x = random.randint(0, 49)
            y = random.randint(0, 49)
            obstacles.append((x, y))

        obs = {
            'position': (1, 25),
            'goal_position': (48, 25),
            'obstacles': obstacles,
        }

        t0 = time.time()
        space.update_from_observation((1, 25), obs)
        elapsed = (time.time() - t0) * 1000

        print(f"  obstacles={num_obstacles:4d}: {elapsed:7.1f}ms")

    # 测试3: Ricci 的 _update_obstacle_uncertainty 详细分析
    print("\n--- Test 3: Ricci _update_obstacle_uncertainty Breakdown ---")
    space = create_space('ricci', 50, 50)

    import random
    random.seed(42)

    for num_obs in [10, 50, 100, 500]:
        obstacles = [(random.randint(0, 49), random.randint(0, 49)) for _ in range(num_obs)]

        t0 = time.time()
        space._update_obstacle_uncertainty(set(obstacles))
        update_ms = (time.time() - t0) * 1000

        t0 = time.time()
        space._recalculate_curvature()
        recalc_ms = (time.time() - t0) * 1000

        print(f"  obstacles={num_obs:4d}: update={update_ms:6.1f}ms, recalc={recalc_ms:6.1f}ms, total={update_ms+recalc_ms:6.1f}ms")

    # 测试4: Conformal 的首次更新
    print("\n--- Test 4: Conformal First Update Breakdown ---")
    space = create_space('conformal', 50, 50)

    for num_obs in [5, 10, 20, 50]:
        obstacles = [(random.randint(0, 49), random.randint(0, 49)) for _ in range(num_obs)]

        obs = {
            'position': (1, 25),
            'goal_position': (48, 25),
            'obstacles': obstacles,
        }

        t0 = time.time()
        space.update_from_observation((1, 25), obs)
        elapsed = (time.time() - t0) * 1000

        print(f"  obstacles={num_obs:3d}: {elapsed:6.1f}ms")

    # 测试5: 空间创建开销
    print("\n--- Test 5: Space Creation Overhead ---")
    for space_name in ['euclidean', 'ricci', 'conformal', 'fisher', 'wasserstein', 'finsler']:
        for size in [20, 50, 100]:
            try:
                t0 = time.time()
                space = create_space(space_name, size, size)
                elapsed = (time.time() - t0) * 1000
                print(f"  {space_name:12s} {size}x{size}: create={elapsed:6.1f}ms")
            except Exception as e:
                print(f"  {space_name:12s} {size}x{size}: ERROR - {e}")


if __name__ == "__main__":
    benchmark_first_update()
    print("\n" + "=" * 80)
    print("First Update Benchmark Complete")
    print("=" * 80)
