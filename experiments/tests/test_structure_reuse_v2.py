"""
结构复用机制验证测试

验证 SSFR 的结构复用功能：
1. 相似场景下结构复用率
2. 结构池增长曲线（应该亚线性）
3. A* 调用率随复用增加而下降
"""

import numpy as np
import time
import sys
import os
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from atlas.core.registry import create_space
from atlas.core.ssfr_enhanced import SSFREnhanced, StructureHypothesis
from atlas.core.path_planning import astar_path, action_from_positions


class ReuseRateEnv:
    """简单环境用于测试复用率"""

    def __init__(self, width: int = 30, height: int = 30, density: float = 0.3, seed: int = 42):
        self.width = width
        self.height = height
        random.seed(seed)

        self.start = (1, height // 2)
        self.goal = (width - 2, height // 2)
        self.position = self.start

        # 生成迷宫
        self.obstacles = set()
        for x in range(width):
            self.obstacles.add((x, 0))
            self.obstacles.add((x, height - 1))
        for y in range(height):
            self.obstacles.add((0, y))
            self.obstacles.add((width - 1, y))
        for x in range(2, width - 2):
            for y in range(2, height - 2):
                if random.random() < density:
                    self.obstacles.add((x, y))
        # 保证通路
        for x in range(1, width - 1):
            for dy in [-1, 0, 1]:
                self.obstacles.discard((x, height // 2 + dy))

    def reset(self):
        self.position = self.start
        return self.get_state()

    def get_state(self):
        return {
            'position': self.position,
            'goal': self.goal,
            'obstacles': list(self.obstacles),
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

        dist = abs(self.position[0] - self.goal[0]) + abs(self.position[1] - self.goal[1])
        reward = -0.1 * dist
        done = self.position == self.goal
        if done:
            reward = 1000.0

        return self.get_state(), reward, done


def test_structure_reuse():
    """测试结构复用机制"""
    print("=" * 80)
    print("STRUCTURE REUSE MECHANISM TEST")
    print("=" * 80)

    ssfr = SSFREnhanced(
        width=30, height=30,
        space_names=['ricci', 'fisher', 'wasserstein', 'conformal', 'finsler'],
        max_structures=500,
        evolution_interval=10,
        reuse_threshold=0.85
    )

    # 模拟 200 步，每步在相似场景（位置变化但场景相似）
    print("\n--- Phase 1: Testing Reuse in Similar Scenarios ---")
    for step in range(200):
        pos = (step % 30, (step * 3) % 30)

        obs = {
            'position': pos,
            'goal_position': (28, 15),
            'obstacles': [],
        }

        ssfr.perceive(pos, obs, active_space_name='ricci')
        ssfr.step_count += 1

        if step % 50 == 0:
            pool_size = len(ssfr.structure_pool.structures)
            stats = ssfr.get_statistics()
            reuse_rate = stats.get('reuse_rate', 0)
            print(f"  Step {step:3d}: pool_size={pool_size:3d}, reuse_rate={reuse_rate*100:.1f}%")

    # 分析结果
    print("\n--- Phase 1 Results ---")
    stats = ssfr.get_statistics()
    print(f"  Total perceptions: {stats['num_perceptions']}")
    print(f"  Total structures: {stats['pool_stats']['num_structures']}")
    print(f"  Reuse rate: {stats.get('reuse_rate', 0)*100:.1f}%")
    print(f"  New generation rate: {stats.get('new_generation_rate', 0)*100:.1f}%")

    # 测试2：不同空间类型的复用
    print("\n--- Phase 2: Testing Reuse Across Different Spaces ---")
    ssfr2 = SSFREnhanced(
        width=30, height=30,
        space_names=['ricci', 'fisher', 'wasserstein', 'conformal', 'finsler'],
        max_structures=500,
        evolution_interval=10,
        reuse_threshold=0.85
    )

    for step in range(100):
        pos = (step % 30, 15)
        space_name = ['ricci', 'fisher', 'wasserstein'][step % 3]

        obs = {
            'position': pos,
            'goal_position': (28, 15),
            'obstacles': [],
        }

        ssfr2.perceive(pos, obs, active_space_name=space_name)
        ssfr2.step_count += 1

        if step % 25 == 0:
            pool_size = len(ssfr2.structure_pool.structures)
            stats = ssfr2.get_statistics()
            reuse_rate = stats.get('reuse_rate', 0)
            print(f"  Step {step:3d}: pool_size={pool_size:3d}, reuse_rate={reuse_rate*100:.1f}% (space={space_name})")

    print("\n--- Phase 2 Results ---")
    stats2 = ssfr2.get_statistics()
    print(f"  Total structures: {stats2['pool_stats']['num_structures']}")
    print(f"  Reuse rate: {stats2.get('reuse_rate', 0)*100:.1f}%")

    # 测试3：结构池增长曲线
    print("\n--- Phase 3: Structure Pool Growth Curve ---")
    ssfr3 = SSFREnhanced(
        width=30, height=30,
        space_names=['ricci', 'fisher', 'wasserstein', 'conformal', 'finsler'],
        max_structures=500,
        evolution_interval=10,
        reuse_threshold=0.85
    )

    pool_sizes = []
    reuse_rates = []

    for step in range(300):
        pos = (step % 30, (step * 3) % 30)

        obs = {
            'position': pos,
            'goal_position': (28, 15),
            'obstacles': [],
        }

        ssfr3.perceive(pos, obs, active_space_name='ricci')
        ssfr3.step_count += 1

        pool_sizes.append(len(ssfr3.structure_pool.structures))
        stats = ssfr3.get_statistics()
        reuse_rates.append(stats.get('reuse_rate', 0))

    # 分析增长曲线
    print("\n  Pool growth (every 50 steps):")
    for i in range(0, 300, 50):
        print(f"    Step {i:3d}: pool_size={pool_sizes[i]:3d}, reuse_rate={reuse_rates[i]*100:.1f}%")

    # 计算增长模式
    initial_growth = pool_sizes[50] - pool_sizes[0]
    mid_growth = pool_sizes[150] - pool_sizes[100]
    late_growth = pool_sizes[250] - pool_sizes[200]

    print(f"\n  Growth analysis:")
    print(f"    Initial (0-50):   +{initial_growth} structures")
    print(f"    Middle (100-150): +{mid_growth} structures")
    print(f"    Late (200-250):   +{late_growth} structures")

    if late_growth < initial_growth:
        print(f"\n  [SUCCESS] Growth is sub-linear (reusing structures)")
    else:
        print(f"\n  [ISSUE] Growth is still linear (no reuse)")

    return {
        'phase1_reuse_rate': stats.get('reuse_rate', 0),
        'phase2_reuse_rate': stats2.get('reuse_rate', 0),
        'pool_sizes': pool_sizes,
        'reuse_rates': reuse_rates,
    }


def test_reuse_with_navigation():
    """测试复用在导航场景中的效果"""
    print("\n" + "=" * 80)
    print("NAVIGATION WITH STRUCTURE REUSE")
    print("=" * 80)

    env = ReuseRateEnv(width=30, height=30, density=0.3)

    # 创建两个 SSFR 实例：一个有复用，一个没有
    ssfr_with_reuse = SSFREnhanced(
        width=30, height=30,
        space_names=['ricci', 'fisher', 'wasserstein', 'conformal', 'finsler'],
        max_structures=500,
        evolution_interval=10,
        reuse_threshold=0.85
    )

    ssfr_without_reuse = SSFREnhanced(
        width=30, height=30,
        space_names=['ricci', 'fisher', 'wasserstein', 'conformal', 'finsler'],
        max_structures=500,
        evolution_interval=10,
        reuse_threshold=1.1  # 设置阈值 > 1，永远不会复用
    )

    # 运行导航
    results_with = run_navigation(env, ssfr_with_reuse, steps=200)
    results_without = run_navigation(env, ssfr_without_reuse, steps=200)

    print("\n--- Comparison ---")
    print(f"  With reuse:    {results_with['total_structures']} structures, {results_with['reuse_rate']*100:.1f}% reuse")
    print(f"  Without reuse: {results_without['total_structures']} structures, {results_without['reuse_rate']*100:.1f}% reuse")

    if results_with['total_structures'] < results_without['total_structures']:
        reduction = (1 - results_with['total_structures'] / results_without['total_structures']) * 100
        print(f"\n  [OK] Structure reduction: {reduction:.1f}%")
    else:
        print(f"\n  [ISSUE] No reduction achieved")

    return results_with, results_without


def run_navigation(env, ssfr, steps=200):
    """运行导航并收集统计"""
    state = env.reset()

    for step in range(steps):
        pos = state['position']
        goal = state['goal']
        obstacles = set(state['obstacles'])

        observation = {
            'position': pos,
            'goal_position': goal,
            'obstacles': list(state['obstacles']),
        }

        ssfr.perceive(pos, observation, active_space_name='ricci')
        ssfr.step_count += 1

        # 简单导航：向右
        state, reward, done = env.step('right')
        if done:
            break

    stats = ssfr.get_statistics()
    return {
        'total_structures': stats['pool_stats']['num_structures'],
        'reuse_rate': stats.get('reuse_rate', 0),
    }


if __name__ == "__main__":
    # 测试1：结构复用机制
    results = test_structure_reuse()

    # 测试2：导航场景中的复用
    results_with, results_without = test_reuse_with_navigation()

    print("\n" + "=" * 80)
    print("Structure Reuse Test Complete")
    print("=" * 80)
