"""
真实场景压力测试：模拟实际导航中的性能退化

关键设计：
1. 使用完整的导航器（含 SSFR + A* + 路径缓存）
2. 长回合（500+ 步）
3. 每步都调用完整的 solve() 流程
4. 观察随着结构池增长，步时间是否退化
5. 对比：有/无路径缓存的性能差异
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Set
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from atlas.core.registry import create_space
from atlas.core.ssfr_enhanced import SSFREnhanced
from atlas.core.path_planning import astar_path, greedy_step, action_from_positions


# ============================================================================
# 1. 真实迷宫环境
# ============================================================================

class RealisticMazeEnv:
    """真实迷宫环境"""

    def __init__(self, width: int = 50, height: int = 50, density: float = 0.3):
        self.width = width
        self.height = height
        self.density = density

        self.start = (1, height // 2)
        self.goal = (width - 2, height // 2)
        self.position = self.start

        # 生成迷宫
        import random
        random.seed(42)
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


# ============================================================================
# 2. 完整导航器（真实场景）
# ============================================================================

class RealisticNavigator:
    """真实导航器（完整流程）"""

    def __init__(self, env: RealisticMazeEnv,
                 use_ssfr: bool = True,
                 use_cache: bool = True,
                 active_only: bool = True):
        self.env = env
        self.use_ssfr = use_ssfr
        self.use_cache = use_cache
        self.active_only = active_only

        # SSFR
        self.ssfr = SSFREnhanced(
            width=env.width, height=env.height,
            space_names=['ricci', 'fisher', 'wasserstein', 'conformal', 'finsler'],
            max_structures=500,
            evolution_interval=10
        )

        self.current_space_name = 'euclidean'
        self.spaces = {}

        # 路径缓存
        self._cached_path = []
        self._cached_obstacles = set()
        self._cached_goal = None
        self._path_valid = False

        # 性能计时
        self.step_times = []
        self.perceive_times = []
        self.astar_times = []
        self.structure_counts = []

    def _get_space(self, name: str):
        if name not in self.spaces:
            try:
                self.spaces[name] = create_space(name, self.env.width, self.env.height)
            except:
                self.spaces[name] = create_space('euclidean', self.env.width, self.env.height)
        return self.spaces.get(name)

    def solve(self, state: Dict) -> str:
        step_start = time.time()
        pos = state['position']
        goal = state['goal']
        obstacles = set(state['obstacles'])

        observation = {
            'position': pos,
            'goal_position': goal,
            'obstacles': list(state['obstacles']),
        }

        # SSFR perceive
        perceive_start = time.time()
        if self.use_ssfr:
            active = self.current_space_name if self.active_only else None
            try:
                self.ssfr.perceive(pos, observation, active_space_name=active)
            except:
                pass

            best = self.ssfr.get_best_structures(n=1)
            if best and best[0].representations:
                self.current_space_name = list(best[0].representations.keys())[0]
        self.perceive_times.append((time.time() - perceive_start) * 1000)

        # 获取空间
        space = self._get_space(self.current_space_name)

        # 路径缓存检查
        if self.use_cache and self._path_valid:
            if self._cached_goal == goal:
                valid = all(p not in obstacles for p in self._cached_path)
                if valid and pos in self._cached_path:
                    idx = self._cached_path.index(pos)
                    if idx + 1 < len(self._cached_path):
                        next_pos = self._cached_path[idx + 1]
                        self.step_times.append((time.time() - step_start) * 1000)
                        self.structure_counts.append(len(self.ssfr.structure_pool.structures))
                        return action_from_positions(pos, next_pos)
                else:
                    self._path_valid = False
            else:
                self._path_valid = False

        # A* 规划
        astar_start = time.time()
        path = astar_path(space, pos, goal, obstacles,
                         self.env.width, self.env.height, max_steps=5000)
        self.astar_times.append((time.time() - astar_start) * 1000)

        if path and len(path) > 1:
            self._cached_path = path
            self._cached_goal = goal
            self._path_valid = True
            next_pos = path[1]
            action = action_from_positions(pos, next_pos)
        else:
            next_pos = greedy_step(space, pos, goal, obstacles,
                                  self.env.width, self.env.height)
            if next_pos:
                action = action_from_positions(pos, next_pos)
            else:
                actions = ['up', 'down', 'left', 'right']
                action = actions[0]

        self.step_times.append((time.time() - step_start) * 1000)
        self.structure_counts.append(len(self.ssfr.structure_pool.structures))
        return action


# ============================================================================
# 3. 真实场景压力测试
# ============================================================================

def run_realistic_stress_test():
    """真实场景压力测试"""
    print("=" * 80)
    print("REALISTIC STRESS TEST")
    print("=" * 80)

    configs = [
        # (width, height, density, max_steps, description)
        (30, 30, 0.3, 200, "Small maze"),
        (50, 50, 0.3, 500, "Medium maze"),
        (50, 50, 0.5, 500, "Dense maze"),
        (80, 80, 0.3, 1000, "Large maze"),
    ]

    for width, height, density, max_steps, desc in configs:
        print(f"\n--- {desc} ({width}x{height}, density={density}) ---")

        env = RealisticMazeEnv(width=width, height=height, density=density)
        nav = RealisticNavigator(env, use_ssfr=True, use_cache=True, active_only=True)

        state = env.reset()
        total_time = 0
        steps = 0

        for step in range(max_steps):
            action = nav.solve(state)
            state, reward, done = env.step(action)
            steps += 1
            if done:
                break

        # 分析结果
        success = env.position == env.goal
        print(f"  Success: {success}, Steps: {steps}")
        print(f"  Avg step time: {np.mean(nav.step_times):.2f}ms")
        print(f"  Max step time: {np.max(nav.step_times):.2f}ms")
        print(f"  Avg perceive time: {np.mean(nav.perceive_times):.2f}ms")
        print(f"  Avg A* time: {np.mean(nav.astar_times):.2f}ms")
        print(f"  Final pool size: {nav.structure_counts[-1] if nav.structure_counts else 0}")

        # 检查性能退化
        if len(nav.step_times) > 100:
            early = np.mean(nav.step_times[:50])
            late = np.mean(nav.step_times[-50:])
            print(f"  Early avg (first 50): {early:.2f}ms")
            print(f"  Late avg (last 50): {late:.2f}ms")
            if early > 0:
                print(f"  Degradation: {late/early:.2f}x")


if __name__ == "__main__":
    run_realistic_stress_test()
    print("\n" + "=" * 80)
    print("Realistic Stress Test Complete")
    print("=" * 80)
