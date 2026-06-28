"""
极端压力测试：暴露深层瓶颈

设计目标：
1. A* + 复杂空间的组合瓶颈
2. 大规模网格（100x100）
3. 高障碍密度（60%）
4. 每步动态变化（动态_interval=1）
5. 高噪声观测（50%）
6. 多空间竞争（6个空间）
7. 长路径（目标在远端）
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Set
import time
import sys
import os
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from atlas.core.registry import create_space
from atlas.core.ssfr_enhanced import SSFREnhanced
from atlas.core.path_planning import astar_path, action_from_positions


# ============================================================================
# 1. 极端迷宫生成
# ============================================================================

def generate_extreme_maze(width: int, height: int, density: float, seed: int = 42) -> Set[Tuple[int, int]]:
    """生成极限密度迷宫，保证有解"""
    random.seed(seed)
    obstacles = set()

    # 边界
    for x in range(width):
        obstacles.add((x, 0))
        obstacles.add((x, height - 1))
    for y in range(height):
        obstacles.add((0, y))
        obstacles.add((width - 1, y))

    # 随机障碍
    for x in range(2, width - 2):
        for y in range(2, height - 2):
            if random.random() < density:
                obstacles.add((x, y))

    # 保证一条蛇形路径
    path_y = height // 2
    for x in range(1, width - 1):
        for dy in [-1, 0, 1]:
            obstacles.discard((x, path_y + dy))

    return obstacles


# ============================================================================
# 2. 极端环境
# ============================================================================

class ExtremeEnv:
    """极限环境：每步都变化"""

    def __init__(self, width: int = 100, height: int = 100,
                 density: float = 0.6, noise_rate: float = 0.5):
        self.width = width
        self.height = height
        self.density = density
        self.noise_rate = noise_rate
        self.step_count = 0

        self.start = (1, height // 2)
        self.goal = (width - 2, height // 2)
        self.position = self.start

        self.static_obstacles = generate_extreme_maze(width, height, density, seed=42)
        self.obstacles = self.static_obstacles.copy()

    def reset(self):
        self.position = self.start
        self.step_count = 0
        self.obstacles = self.static_obstacles.copy()
        return self.get_state()

    def get_state(self):
        obs = list(self.obstacles)
        if self.noise_rate > 0:
            true_obs = set(self.obstacles)
            for o in list(true_obs):
                if random.random() < self.noise_rate:
                    true_obs.discard(o)
            for _ in range(int(len(true_obs) * self.noise_rate * 0.3)):
                x = random.randint(1, self.width - 2)
                y = random.randint(1, self.height - 2)
                true_obs.add((x, y))
            obs = list(true_obs)

        return {
            'position': self.position,
            'goal': self.goal,
            'obstacles': obs,
        }

    def step(self, action: str):
        x, y = self.position
        if action == 'up':
            new_pos = (x, y - 1)
        elif action == 'down':
            new_pos = (x, y + 1)
        elif action == 'left':
            new_pos = (x - 1, y)
        elif action == 'right':
            new_pos = (x + 1, y)
        else:
            new_pos = self.position

        if not (0 <= new_pos[0] < self.width and 0 <= new_pos[1] < self.height):
            new_pos = self.position
        if new_pos in self.obstacles:
            new_pos = self.position

        self.position = new_pos
        self.step_count += 1

        # 每步都随机变化一些障碍
        if self.step_count % 1 == 0:
            # 随机移除一些障碍
            for _ in range(self.width * self.height // 100):
                x = random.randint(1, self.width - 2)
                y = random.randint(1, self.height - 2)
                self.obstacles.discard((x, y))
            # 随机添加一些障碍
            for _ in range(self.width * self.height // 100):
                x = random.randint(1, self.width - 2)
                y = random.randint(1, self.height - 2)
                if (x, y) != self.position and (x, y) != self.goal:
                    self.obstacles.add((x, y))

        dist = abs(self.position[0] - self.goal[0]) + abs(self.position[1] - self.goal[1])
        reward = -0.1 * dist
        done = self.position == self.goal
        if done:
            reward = 1000.0

        return self.get_state(), reward, done


# ============================================================================
# 3. 基准测试：A* + 复杂空间
# ============================================================================

def benchmark_astar_with_space(space_name: str, width: int, height: int,
                                density: float, num_runs: int = 5):
    """基准测试：A* 使用复杂空间的距离计算"""
    space = create_space(space_name, width, height)
    obstacles = generate_extreme_maze(width, height, density, seed=42)
    start = (1, height // 2)
    goal = (width - 2, height // 2)

    times = []
    path_lengths = []
    compute_dist_calls = []

    for _ in range(num_runs):
        # 重置空间
        space = create_space(space_name, width, height)
        for i in range(10):
            pos = (i * 5 % width, height // 2)
            space.update_from_observation(pos, {
                'position': pos,
                'goal_position': goal,
                'obstacles': list(obstacles)[:20],
            })

        start_t = time.time()
        path = astar_path(space, start, goal, obstacles, width, height, max_steps=10000)
        elapsed = time.time() - start_t

        times.append(elapsed)
        path_lengths.append(len(path) if path else 0)

    return {
        'space': space_name,
        'width': width,
        'density': density,
        'avg_ms': np.mean(times) * 1000,
        'max_ms': np.max(times) * 1000,
        'path_length': np.mean(path_lengths) if path_lengths else 0,
        'success': any(p > 0 for p in path_lengths),
    }


# ============================================================================
# 4. 极端压力测试
# ============================================================================

def run_extreme_stress_test():
    """极端压力测试"""
    print("=" * 80)
    print("EXTREME STRESS TEST")
    print("=" * 80)

    # 测试1: A* + 各种空间的组合性能
    print("\n--- Test 1: A* with Complex Spaces ---")
    for space_name in ['euclidean', 'ricci', 'conformal', 'fisher', 'wasserstein', 'finsler']:
        for size in [20, 30, 50]:
            for density in [0.2, 0.4, 0.6]:
                try:
                    result = benchmark_astar_with_space(space_name, size, size, density, num_runs=3)
                    print(f"  {space_name:12s} {size}x{size} d={density:.1f}: "
                          f"avg={result['avg_ms']:8.1f}ms, max={result['max_ms']:8.1f}ms, "
                          f"path={result['path_length']:5.0f}, success={result['success']}")
                except Exception as e:
                    print(f"  {space_name:12s} {size}x{size} d={density:.1f}: ERROR - {e}")

    # 测试2: 极端环境端到端
    print("\n--- Test 2: Extreme Environment (100x100, 60% density, 50% noise) ---")
    env = ExtremeEnv(width=100, height=100, density=0.6, noise_rate=0.5)
    space = create_space('euclidean', 100, 100)

    state = env.reset()
    total_time = 0
    steps = 0
    max_step_time = 0

    for step in range(500):
        step_start = time.time()
        pos = state['position']
        goal = state['goal']
        obstacles = set(state['obstacles'])

        path = astar_path(space, pos, goal, obstacles, 100, 100, max_steps=10000)
        if path and len(path) > 1:
            action = action_from_positions(pos, path[1])
        else:
            action = 'right'

        state, reward, done = env.step(action)
        step_time = (time.time() - step_start) * 1000
        total_time += step_time
        max_step_time = max(max_step_time, step_time)
        steps += 1

        if step % 50 == 0:
            print(f"  Step {step}: pos={pos}, step_time={step_time:.1f}ms, avg={total_time/max(1,steps):.1f}ms")

        if done:
            break

    print(f"\n  Total: {steps} steps, {total_time:.0f}ms, avg={total_time/max(1,steps):.1f}ms/step, max={max_step_time:.1f}ms")
    print(f"  Success: {env.position == env.goal}")

    # 测试3: SSFR 在极端条件下的性能
    print("\n--- Test 3: SSFR Performance under Stress ---")
    ssfr = SSFREnhanced(
        width=50, height=50,
        space_names=['ricci', 'fisher', 'wasserstein', 'conformal', 'finsler'],
        max_structures=200,
        evolution_interval=5
    )

    perceive_times = []
    for i in range(100):
        pos = (i % 50, (i * 3) % 50)
        obs = {
            'position': pos,
            'goal_position': (48, 25),
            'obstacles': [(pos[0] + dx, pos[1] + dy)
                         for dx in range(-2, 3) for dy in range(-2, 3)
                         if random.random() < 0.3],
        }

        start = time.time()
        ssfr.perceive(pos, obs, active_space_name='ricci')
        perceive_times.append(time.time() - start)

    print(f"  SSFR perceive (100 steps, 50x50, active=ricci):")
    print(f"    avg={np.mean(perceive_times)*1000:.2f}ms, max={np.max(perceive_times)*1000:.2f}ms")
    print(f"    total={np.sum(perceive_times)*1000:.0f}ms")

    # 测试4: 结构池增长压力
    print("\n--- Test 4: Structure Pool Growth ---")
    ssfr2 = SSFREnhanced(
        width=30, height=30,
        space_names=['ricci', 'fisher', 'wasserstein'],
        max_structures=500,
        evolution_interval=1
    )

    pool_size_history = []
    for i in range(200):
        pos = (i % 30, (i * 7) % 30)
        obs = {
            'position': pos,
            'goal_position': (28, 15),
            'obstacles': [],
        }
        ssfr2.perceive(pos, obs, active_space_name='ricci')
        if i % 10 == 0:
            pool_size = len(ssfr2.structure_pool.structures)
            pool_size_history.append(pool_size)
            print(f"  Step {i}: pool_size={pool_size}")

    print(f"  Final pool size: {len(ssfr2.structure_pool.structures)}")


# ============================================================================
# 5. 主入口
# ============================================================================

if __name__ == "__main__":
    run_extreme_stress_test()
    print("\n" + "=" * 80)
    print("Extreme Stress Test Complete")
    print("=" * 80)
