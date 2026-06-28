"""
真正的"搜索→知识"转化分析

核心问题：当前架构中，A* 是路径规划工具，不是结构发现工具。
真正的转化应该体现在：
1. 空间选择效率：从探索到利用的转变
2. 结构竞争效率：新假设生成率 vs 复用率
3. 预测准确性：结构预测误差随时间下降
4. 认知负担：每步需要处理的信息量

改进的度量：
- 探索率 = 生成新结构的步数 / 总步数（应该下降）
- 利用深度 = 平均每个结构被使用的次数（应该上升）
- 预测准确性 = 结构预测与实际观测的匹配度（应该上升）
- 空间稳定性 = 连续多少步使用同一空间（应该增加）
"""

import numpy as np
import time
import sys
import os
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from atlas.core.registry import create_space
from atlas.core.ssfr_enhanced import SSFREnhanced
from atlas.core.path_planning import astar_path, action_from_positions


class ReuseRateEnv:
    """简单环境 - 增加随机移动让路径更长"""

    def __init__(self, width: int = 30, height: int = 30, density: float = 0.3, seed: int = 42):
        self.width = width
        self.height = height
        random.seed(seed)

        self.start = (1, height // 2)
        self.goal = (width - 2, height // 2)
        self.position = self.start

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
        return self.get_state(), 0, False  # 永不结束，用于长期观察


def analyze_true_knowledge_conversion():
    """分析真正的"搜索→知识"转化"""
    print("=" * 80)
    print("TRUE KNOWLEDGE CONVERSION ANALYSIS")
    print("=" * 80)

    env = ReuseRateEnv(width=30, height=30, density=0.3, seed=42)
    ssfr = SSFREnhanced(
        width=30, height=30,
        space_names=['ricci', 'fisher', 'wasserstein', 'conformal', 'finsler'],
        max_structures=500,
        evolution_interval=10
    )

    # 追踪指标
    exploration_rate = []  # 生成新结构的步数比例
    structure_usage = {}  # 结构ID -> 使用次数
    space_stability = []  # 连续使用同一空间的步数
    prediction_errors = []  # 预测误差

    current_space = 'euclidean'
    space_history = []
    prev_pool_size = 0

    state = env.reset()
    for step in range(300):
        pos = state['position']
        goal = state['goal']
        obstacles = set(state['obstacles'])

        observation = {
            'position': pos,
            'goal_position': goal,
            'obstacles': list(state['obstacles']),
        }

        # SSFR perceive
        ssfr.perceive(pos, observation, active_space_name=current_space)

        # 获取最佳结构
        best = ssfr.get_best_structures(n=1)
        if best:
            # 记录结构使用
            for struct in best:
                if struct.id not in structure_usage:
                    structure_usage[struct.id] = 0
                structure_usage[struct.id] += 1

            # 空间选择
            if best[0].representations:
                new_space = list(best[0].representations.keys())[0]
                space_history.append(new_space)
                if new_space != current_space:
                    current_space = new_space

        # 路径规划（简化版）
        space = create_space(current_space, 30, 30)
        space.update_from_observation(pos, observation)
        path = astar_path(space, pos, goal, obstacles, 30, 30, max_steps=5000)

        if path and len(path) > 1:
            action = action_from_positions(pos, path[1])
        else:
            action = 'right'

        state, reward, done = env.step(action)
        if done:
            break

        # 记录指标
        pool_size = len(ssfr.structure_pool.structures)
        is_exploration = pool_size > prev_pool_size
        exploration_rate.append(1.0 if is_exploration else 0.0)
        prev_pool_size = pool_size

    # 分析1: 探索率随时间变化
    print("\n--- 1. Exploration Rate Over Time ---")
    window = 50
    for i in range(0, len(exploration_rate), window):
        w = exploration_rate[i:i+window]
        if w:
            print(f"  Steps {i:4d}-{i+len(w)-1:4d}: exploration={np.mean(w)*100:5.1f}%")

    # 分析2: 结构使用分布
    print("\n--- 2. Structure Usage Distribution ---")
    usage_counts = list(structure_usage.values())
    if usage_counts:
        print(f"  Total structures: {len(usage_counts)}")
        print(f"  Mean usage: {np.mean(usage_counts):.1f}")
        print(f"  Max usage: {np.max(usage_counts)}")
        print(f"  Structures used once: {sum(1 for u in usage_counts if u == 1)}/{len(usage_counts)}")
        print(f"  Structures used >5x: {sum(1 for u in usage_counts if u > 5)}/{len(usage_counts)}")

    # 分析3: 空间稳定性
    print("\n--- 3. Space Stability ---")
    if space_history:
        # 计算连续使用同一空间的平均长度
        streaks = []
        current_streak = 1
        for i in range(1, len(space_history)):
            if space_history[i] == space_history[i-1]:
                current_streak += 1
            else:
                streaks.append(current_streak)
                current_streak = 1
        streaks.append(current_streak)

        print(f"  Total space switches: {len(streaks) - 1}")
        print(f"  Mean streak length: {np.mean(streaks):.1f}")
        print(f"  Max streak length: {np.max(streaks)}")
        print(f"  Space distribution:")
        from collections import Counter
        for space, count in Counter(space_history).most_common():
            print(f"    {space}: {count} steps ({count/len(space_history)*100:.1f}%)")

    # 分析4: 结构池增长曲线
    print("\n--- 4. Structure Pool Growth ---")
    # 重新计算每步的池大小
    env2 = ReuseRateEnv(width=30, height=30, density=0.3, seed=42)
    ssfr2 = SSFREnhanced(
        width=30, height=30,
        space_names=['ricci', 'fisher', 'wasserstein', 'conformal', 'finsler'],
        max_structures=500,
        evolution_interval=10
    )

    pool_sizes = []
    state = env2.reset()
    for step in range(300):
        pos = state['position']
        observation = {
            'position': pos,
            'goal_position': (28, 15),
            'obstacles': list(state['obstacles']),
        }
        ssfr2.perceive(pos, observation, active_space_name='ricci')
        pool_sizes.append(len(ssfr2.structure_pool.structures))
        state, reward, done = env2.step('right')
        if done:
            break

    for i in range(0, len(pool_sizes), 50):
        w = pool_sizes[i:i+50]
        if w:
            print(f"  Steps {i:4d}-{i+len(w)-1:4d}: pool_size={w[-1]}")

    # 分析5: 知识积累效率
    print("\n--- 5. Knowledge Accumulation Efficiency ---")
    total_steps = len(exploration_rate)
    total_structures = len(structure_usage)
    if total_structures > 0:
        print(f"  Structures per step: {total_structures/total_steps:.2f}")
        print(f"  Exploration rate: {np.mean(exploration_rate)*100:.1f}%")
        print(f"  Knowledge density: {total_structures/max(1, pool_sizes[-1]):.2f}")


if __name__ == "__main__":
    analyze_true_knowledge_conversion()
    print("\n" + "=" * 80)
    print("Knowledge Conversion Analysis Complete")
    print("=" * 80)
