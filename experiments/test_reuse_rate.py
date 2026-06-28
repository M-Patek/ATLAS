"""
结构复用率分析：验证"搜索→知识"转化假设

核心假设：
随着结构池增长，系统应该越来越依赖已发现的结构知识，
而不是每次都进行在线搜索（A*）。

度量指标：
1. 结构复用率 = 复用结构的步数 / 总步数
2. A* 调用率 = 调用 A* 的步数 / 总步数（应该随时间下降）
3. 缓存命中率 = 路径缓存命中 / 总步数
4. 空间切换频率 = 切换空间的次数 / 总步数（应该稳定或下降）
5. 新结构生成率 = 新结构数 / 步数（应该随时间下降）
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

    def __init__(self, width: int = 30, height: int = 30, density: float = 0.3):
        self.width = width
        self.height = height
        random.seed(42)

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


class ReuseRateNavigator:
    """带详细复用率追踪的导航器"""

    def __init__(self, env: ReuseRateEnv):
        self.env = env

        # SSFR
        self.ssfr = SSFREnhanced(
            width=env.width, height=env.height,
            space_names=['ricci', 'fisher', 'wasserstein', 'conformal', 'finsler'],
            max_structures=500,
            evolution_interval=10
        )

        self.current_space = 'euclidean'
        self.spaces = {}

        # 路径缓存
        self._cached_path = []
        self._cached_goal = None
        self._path_valid = False

        # 复用率追踪
        self.records = []
        self.step_count = 0

    def _get_space(self, name):
        if name not in self.spaces:
            try:
                self.spaces[name] = create_space(name, self.env.width, self.env.height)
            except:
                self.spaces[name] = create_space('euclidean', self.env.width, self.env.height)
        return self.spaces.get(name)

    def solve(self, state):
        record = {
            'step': self.step_count,
            'used_astar': False,
            'cache_hit': False,
            'space_switched': False,
            'new_structure_generated': False,
            'structure_reused': False,
            'pool_size_before': len(self.ssfr.structure_pool.structures),
        }

        pos = state['position']
        goal = state['goal']
        obstacles = set(state['obstacles'])

        observation = {
            'position': pos,
            'goal_position': goal,
            'obstacles': list(state['obstacles']),
        }

        # 1. SSFR perceive
        self.ssfr.perceive(pos, observation, active_space_name=self.current_space)

        # 2. 获取最佳结构
        best = self.ssfr.get_best_structures(n=1)
        if best and best[0].representations:
            new_space = list(best[0].representations.keys())[0]
            if new_space != self.current_space:
                record['space_switched'] = True
                self.current_space = new_space
                self._path_valid = False
            else:
                # 空间没变，但结构被复用了
                record['structure_reused'] = True

        # 3. 空间更新
        space = self._get_space(self.current_space)
        if not self._path_valid:
            space.update_from_observation(pos, observation)

        # 4. 路径缓存检查
        if self._path_valid and self._cached_goal == goal:
            valid = all(p not in obstacles for p in self._cached_path)
            if valid and pos in self._cached_path:
                idx = self._cached_path.index(pos)
                if idx + 1 < len(self._cached_path):
                    next_pos = self._cached_path[idx + 1]
                    record['cache_hit'] = True
                    action = action_from_positions(pos, next_pos)
                    # 记录结构池大小（即使缓存命中也要记录）
                    record['pool_size_after'] = len(self.ssfr.structure_pool.structures)
                    record['new_structure_generated'] = record['pool_size_after'] > record['pool_size_before']
                    self.records.append(record)
                    self.step_count += 1
                    return action
            self._path_valid = False

        # 5. A* 规划
        record['used_astar'] = True
        path = astar_path(space, pos, goal, obstacles,
                         self.env.width, self.env.height, max_steps=5000)

        if path and len(path) > 1:
            self._cached_path = path
            self._cached_goal = goal
            self._path_valid = True
            next_pos = path[1]
            action = action_from_positions(pos, next_pos)
        else:
            action = 'right'

        record['pool_size_after'] = len(self.ssfr.structure_pool.structures)
        record['new_structure_generated'] = record['pool_size_after'] > record['pool_size_before']
        self.records.append(record)
        self.step_count += 1
        return action


def analyze_reuse_rate():
    """分析复用率"""
    print("=" * 80)
    print("STRUCTURE REUSE RATE ANALYSIS")
    print("=" * 80)

    env = ReuseRateEnv(width=30, height=30, density=0.3)
    nav = ReuseRateNavigator(env)

    # 运行多回合
    num_episodes = 5
    for ep in range(num_episodes):
        state = env.reset()
        nav._path_valid = False
        nav._cached_path = []

        for step in range(200):
            action = nav.solve(state)
            state, reward, done = env.step(action)
            if done:
                break

    # 分析复用率
    records = nav.records
    total_steps = len(records)

    print(f"\n--- Overall Statistics ---")
    print(f"  Total steps: {total_steps}")
    print(f"  Total structures: {len(nav.ssfr.structure_pool.structures)}")

    # 计算各指标
    astar_count = sum(1 for r in records if r['used_astar'])
    cache_hit_count = sum(1 for r in records if r['cache_hit'])
    switch_count = sum(1 for r in records if r['space_switched'])
    reuse_count = sum(1 for r in records if r['structure_reused'])
    new_struct_count = sum(1 for r in records if r.get('new_structure_generated', False))

    print(f"\n--- Aggregate Metrics ---")
    print(f"  A* call rate: {astar_count}/{total_steps} = {astar_count/total_steps*100:.1f}%")
    print(f"  Cache hit rate: {cache_hit_count}/{total_steps} = {cache_hit_count/total_steps*100:.1f}%")
    print(f"  Space switch rate: {switch_count}/{total_steps} = {switch_count/total_steps*100:.1f}%")
    print(f"  Structure reuse rate: {reuse_count}/{total_steps} = {reuse_count/total_steps*100:.1f}%")
    print(f"  New structure rate: {new_struct_count}/{total_steps} = {new_struct_count/total_steps*100:.1f}%")

    # 分析随时间的变化趋势
    print(f"\n--- Temporal Analysis (per 50 steps) ---")
    window_size = 50
    for i in range(0, total_steps, window_size):
        window = records[i:i+window_size]
        if not window:
            break

        astar_rate = sum(1 for r in window if r['used_astar']) / len(window)
        cache_rate = sum(1 for r in window if r['cache_hit']) / len(window)
        reuse_rate = sum(1 for r in window if r['structure_reused']) / len(window)
        switch_rate = sum(1 for r in window if r['space_switched']) / len(window)

        print(f"  Steps {i:4d}-{i+len(window)-1:4d}: "
              f"A*={astar_rate*100:5.1f}%, cache={cache_rate*100:5.1f}%, "
              f"reuse={reuse_rate*100:5.1f}%, switch={switch_rate*100:5.1f}%")

    # 分析结构池增长与 A* 调用率的关系
    print(f"\n--- Structure Pool vs A* Rate (per 20 steps) ---")
    window_size = 20
    for i in range(0, min(total_steps, 500), window_size):
        window = records[i:i+window_size]
        if not window:
            break

        # 获取该窗口的结构池大小
        pool_size = window[-1].get('pool_size_after', 0) if 'pool_size_after' in window[-1] else 0
        astar_rate = sum(1 for r in window if r['used_astar']) / len(window)

        print(f"  Steps {i:4d}-{i+len(window)-1:4d}: pool_size={pool_size:3d}, A* rate={astar_rate*100:5.1f}%")


if __name__ == "__main__":
    analyze_reuse_rate()
    print("\n" + "=" * 80)
    print("Reuse Rate Analysis Complete")
    print("=" * 80)
