"""
极限压力测试：专门暴露深层瓶颈

设计原则：
1. 专门攻击已知弱点
2. 逐步增加压力直到失败
3. 记录每个阶段的性能退化

测试矩阵：
- 网格大小: 30x30 -> 50x50 -> 100x100
- 障碍密度: 20% -> 40% -> 60%
- 动态频率: 每10步 -> 每5步 -> 每步
- 噪声率: 0% -> 20% -> 50%
- 空间数量: 4 -> 6 -> 10
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
from atlas.core.path_planning import astar_path, greedy_step, action_from_positions


# ============================================================================
# 1. 极限场景生成器
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


class ExtremeMazeEnv:
    """极限迷宫环境"""

    def __init__(self, width: int = 50, height: int = 50,
                 density: float = 0.3, noise_rate: float = 0.0,
                 dynamic_interval: int = 1000, seed: int = 42):
        self.width = width
        self.height = height
        self.density = density
        self.noise_rate = noise_rate
        self.dynamic_interval = dynamic_interval
        self.seed = seed
        self.step_count = 0

        self.start = (1, height // 2)
        self.goal = (width - 2, height // 2)
        self.position = self.start

        self.static_obstacles = generate_extreme_maze(width, height, density, seed)
        self.dynamic_obstacles = set()
        self.obstacles = self.static_obstacles.copy()

    def reset(self):
        self.position = self.start
        self.step_count = 0
        self.dynamic_obstacles.clear()
        self.obstacles = self.static_obstacles.copy()
        return self.get_state()

    def get_state(self):
        obs = list(self.obstacles)
        if self.noise_rate > 0:
            # 添加噪声
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

        # 动态障碍
        if self.step_count % self.dynamic_interval == 0 and self.dynamic_interval < 1000:
            self.obstacles -= self.dynamic_obstacles
            self.dynamic_obstacles.clear()
            for _ in range(self.width * self.height // 30):
                x = random.randint(2, self.width - 3)
                y = random.randint(2, self.height - 3)
                if abs(x - self.position[0]) > 2:
                    self.dynamic_obstacles.add((x, y))
            self.obstacles |= self.dynamic_obstacles

        dist = abs(self.position[0] - self.goal[0]) + abs(self.position[1] - self.goal[1])
        reward = -0.1 * dist
        done = self.position == self.goal
        if done:
            reward = 1000.0

        return self.get_state(), reward, done


# ============================================================================
# 2. 微基准测试：隔离各个组件
# ============================================================================

def benchmark_space_update(space_name: str, width: int, height: int, num_steps: int = 100):
    """基准测试：空间更新性能"""
    space = create_space(space_name, width, height)
    times = []

    for i in range(num_steps):
        pos = (i % width, (i * 3) % height)
        obs = {
            'position': pos,
            'goal_position': (width - 2, height // 2),
            'obstacles': [(pos[0] + 1, pos[1]), (pos[0] - 1, pos[1])] if i % 5 == 0 else [],
        }
        start = time.time()
        space.update_from_observation(pos, obs)
        times.append(time.time() - start)

    return {
        'space': space_name,
        'avg_ms': np.mean(times) * 1000,
        'max_ms': np.max(times) * 1000,
        'total_ms': np.sum(times) * 1000,
    }


def benchmark_compute_distance(space_name: str, width: int, height: int, num_calls: int = 1000):
    """基准测试：距离计算性能"""
    space = create_space(space_name, width, height)

    # 先更新一些数据
    for i in range(10):
        pos = (i * 5 % width, height // 2)
        space.update_from_observation(pos, {
            'position': pos,
            'goal_position': (width - 2, height // 2),
            'obstacles': [],
        })

    times = []
    for i in range(num_calls):
        p1 = (random.randint(0, width - 1), random.randint(0, height - 1))
        p2 = (random.randint(0, width - 1), random.randint(0, height - 1))
        start = time.time()
        d = space.compute_distance(p1, p2)
        times.append(time.time() - start)

    return {
        'space': space_name,
        'avg_ms': np.mean(times) * 1000,
        'max_ms': np.max(times) * 1000,
        'total_ms': np.sum(times) * 1000,
    }


def benchmark_astar(space_name: str, width: int, height: int, density: float, num_runs: int = 10):
    """基准测试：A* 性能"""
    space = create_space(space_name, width, height)
    obstacles = generate_extreme_maze(width, height, density, seed=42)
    start = (1, height // 2)
    goal = (width - 2, height // 2)

    times = []
    path_lengths = []
    for _ in range(num_runs):
        start_t = time.time()
        path = astar_path(space, start, goal, obstacles, width, height, max_steps=5000)
        times.append(time.time() - start_t)
        path_lengths.append(len(path) if path else 0)

    return {
        'space': space_name,
        'density': density,
        'avg_ms': np.mean(times) * 1000,
        'max_ms': np.max(times) * 1000,
        'path_length': np.mean(path_lengths) if path_lengths else 0,
        'success_rate': sum(1 for p in path_lengths if p > 0) / len(path_lengths),
    }


def benchmark_ssfr_perceive(width: int, height: int, num_steps: int = 50, active_only: bool = True):
    """基准测试：SSFR perceive 性能"""
    ssfr = SSFREnhanced(
        width=width, height=height,
        space_names=['ricci', 'fisher', 'wasserstein', 'conformal', 'finsler'],
        max_structures=100,
        evolution_interval=10
    )

    times_full = []
    times_active = []

    for i in range(num_steps):
        pos = (i % width, (i * 3) % height)
        obs = {
            'position': pos,
            'goal_position': (width - 2, height // 2),
            'obstacles': [(pos[0] + 2, pos[1])] if i % 3 == 0 else [],
        }

        # 全空间更新
        start = time.time()
        ssfr.perceive(pos, obs, active_space_name=None)
        times_full.append(time.time() - start)

        # 活跃空间更新
        start = time.time()
        ssfr.perceive(pos, obs, active_space_name='ricci')
        times_active.append(time.time() - start)

    return {
        'width': width,
        'height': height,
        'full_avg_ms': np.mean(times_full) * 1000,
        'full_max_ms': np.max(times_full) * 1000,
        'active_avg_ms': np.mean(times_active) * 1000,
        'active_max_ms': np.max(times_active) * 1000,
        'speedup': np.mean(times_full) / np.mean(times_active) if np.mean(times_active) > 0 else 0,
    }


# ============================================================================
# 3. 压力梯度测试
# ============================================================================

def run_pressure_test():
    """压力梯度测试：逐步增加难度直到失败"""
    print("=" * 80)
    print("PRESSURE GRADIENT TEST")
    print("=" * 80)

    # 测试1: 网格大小压力
    print("\n--- Test 1: Grid Size Pressure ---")
    for size in [20, 30, 50, 80, 100]:
        result = benchmark_astar('euclidean', size, size, density=0.2, num_runs=5)
        print(f"  {size}x{size}: avg={result['avg_ms']:.1f}ms, max={result['max_ms']:.1f}ms, "
              f"path={result['path_length']:.0f}, success={result['success_rate']*100:.0f}%")

    # 测试2: 障碍密度压力
    print("\n--- Test 2: Obstacle Density Pressure (50x50) ---")
    for density in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        result = benchmark_astar('euclidean', 50, 50, density=density, num_runs=5)
        print(f"  density={density:.1f}: avg={result['avg_ms']:.1f}ms, max={result['max_ms']:.1f}ms, "
              f"success={result['success_rate']*100:.0f}%")

    # 测试3: 空间更新压力
    print("\n--- Test 3: Space Update Pressure ---")
    for size in [20, 30, 50, 80, 100]:
        result = benchmark_space_update('ricci', size, size, num_steps=50)
        print(f"  {size}x{size}: avg={result['avg_ms']:.2f}ms, max={result['max_ms']:.2f}ms")

    # 测试4: 距离计算压力
    print("\n--- Test 4: Compute Distance Pressure ---")
    for space in ['euclidean', 'ricci', 'conformal', 'fisher', 'wasserstein', 'finsler']:
        try:
            result = benchmark_compute_distance(space, 50, 50, num_calls=1000)
            print(f"  {space:12s}: avg={result['avg_ms']:.3f}ms, max={result['max_ms']:.3f}ms")
        except Exception as e:
            print(f"  {space:12s}: ERROR - {e}")

    # 测试5: SSFR perceive 压力
    print("\n--- Test 5: SSFR Perceive Pressure ---")
    for size in [20, 30, 50]:
        result = benchmark_ssfr_perceive(size, size, num_steps=50)
        print(f"  {size}x{size}: full={result['full_avg_ms']:.1f}ms, active={result['active_avg_ms']:.1f}ms, "
              f"speedup={result['speedup']:.1f}x")

    # 测试6: 端到端压力（综合所有因素）
    print("\n--- Test 6: End-to-End Pressure Test ---")
    configs = [
        (20, 0.2, 0.0, 1000),
        (30, 0.3, 0.0, 1000),
        (50, 0.3, 0.1, 1000),
        (50, 0.4, 0.2, 1000),
        (80, 0.3, 0.2, 2000),
        (100, 0.3, 0.3, 2000),
    ]

    for width, density, noise, max_steps in configs:
        env = ExtremeMazeEnv(width=width, height=width, density=density,
                              noise_rate=noise, dynamic_interval=1000)
        space = create_space('euclidean', width, width)

        state = env.reset()
        start_time = time.time()
        steps = 0

        for step in range(max_steps):
            pos = state['position']
            goal = state['goal']
            obstacles = set(state['obstacles'])

            # 简单策略：A* + Euclidean
            path = astar_path(space, pos, goal, obstacles, width, width, max_steps=5000)
            if path and len(path) > 1:
                action = action_from_positions(pos, path[1])
            else:
                action = 'right'

            state, reward, done = env.step(action)
            steps += 1
            if done:
                break

        elapsed = (time.time() - start_time) * 1000
        success = env.position == env.goal

        print(f"  {width}x{width} d={density:.1f} n={noise:.1f}: "
              f"steps={steps}, time={elapsed:.0f}ms, "
              f"avg={elapsed/max(1,steps):.1f}ms/step, success={success}")


# ============================================================================
# 4. 主入口
# ============================================================================

if __name__ == "__main__":
    run_pressure_test()
    print("\n" + "=" * 80)
    print("Pressure Test Complete")
    print("=" * 80)
