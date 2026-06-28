"""
Spike 来源分析：找到性能 spikes 的根因

设计：
1. 记录每步的详细时间分解
2. 找出哪些步骤产生了 spikes
3. 分析 spike 步骤的特征
"""

import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from atlas.core.registry import create_space
from atlas.core.ssfr_enhanced import SSFREnhanced
from atlas.core.path_planning import astar_path, action_from_positions


class DetailedProfiler:
    """详细性能分析器"""

    def __init__(self, env):
        self.env = env
        self.ssfr = SSFREnhanced(
            width=env.width, height=env.height,
            space_names=['ricci', 'fisher', 'wasserstein', 'conformal', 'finsler'],
            max_structures=500,
            evolution_interval=10
        )
        self.spaces = {}
        self.current_space = 'euclidean'

        # 路径缓存
        self._cached_path = []
        self._cached_goal = None
        self._path_valid = False

        # 详细记录
        self.records = []

    def _get_space(self, name):
        if name not in self.spaces:
            try:
                self.spaces[name] = create_space(name, self.env.width, self.env.height)
            except:
                self.spaces[name] = create_space('euclidean', self.env.width, self.env.height)
        return self.spaces.get(name)

    def solve(self, state):
        record = {'step': len(self.records)}
        t0 = time.time()

        pos = state['position']
        goal = state['goal']
        obstacles = set(state['obstacles'])

        observation = {
            'position': pos,
            'goal_position': goal,
            'obstacles': list(state['obstacles']),
        }

        # 1. SSFR perceive
        t1 = time.time()
        try:
            self.ssfr.perceive(pos, observation, active_space_name=self.current_space)
        except:
            pass
        record['perceive_ms'] = (time.time() - t1) * 1000

        # 2. 获取最佳结构
        t2 = time.time()
        best = self.ssfr.get_best_structures(n=1)
        record['get_best_ms'] = (time.time() - t2) * 1000

        if best and best[0].representations:
            new_space = list(best[0].representations.keys())[0]
            if new_space != self.current_space:
                self.current_space = new_space
                self._path_valid = False

        # 3. 获取空间实例
        t3 = time.time()
        space = self._get_space(self.current_space)
        record['get_space_ms'] = (time.time() - t3) * 1000

        # 4. 空间更新
        t4 = time.time()
        if not self._path_valid:
            space.update_from_observation(pos, observation)
        record['space_update_ms'] = (time.time() - t4) * 1000

        # 5. 路径缓存检查
        t5 = time.time()
        cache_hit = False
        if self._path_valid and self._cached_goal == goal:
            valid = all(p not in obstacles for p in self._cached_path)
            if valid and pos in self._cached_path:
                idx = self._cached_path.index(pos)
                if idx + 1 < len(self._cached_path):
                    next_pos = self._cached_path[idx + 1]
                    cache_hit = True
                    action = action_from_positions(pos, next_pos)
        record['cache_check_ms'] = (time.time() - t5) * 1000
        record['cache_hit'] = cache_hit

        if not cache_hit:
            self._path_valid = False

            # 6. A* 规划
            t6 = time.time()
            path = astar_path(space, pos, goal, obstacles,
                            self.env.width, self.env.height, max_steps=5000)
            record['astar_ms'] = (time.time() - t6) * 1000

            if path and len(path) > 1:
                self._cached_path = path
                self._cached_goal = goal
                self._path_valid = True
                next_pos = path[1]
                action = action_from_positions(pos, next_pos)
            else:
                action = 'right'

        record['total_ms'] = (time.time() - t0) * 1000
        record['pool_size'] = len(self.ssfr.structure_pool.structures)
        record['space'] = self.current_space

        self.records.append(record)
        return action


def analyze_spikes():
    """分析 spikes"""
    print("=" * 80)
    print("SPIKE ANALYSIS")
    print("=" * 80)

    # 创建环境
    import random
    random.seed(42)

    class SimpleEnv:
        def __init__(self, w, h, density):
            self.width = w
            self.height = h
            self.start = (1, h // 2)
            self.goal = (w - 2, h // 2)
            self.position = self.start
            self.obstacles = set()
            for x in range(w):
                self.obstacles.add((x, 0))
                self.obstacles.add((x, h - 1))
            for y in range(h):
                self.obstacles.add((0, y))
                self.obstacles.add((w - 1, y))
            for x in range(2, w - 2):
                for y in range(2, h - 2):
                    if random.random() < density:
                        self.obstacles.add((x, y))
            for x in range(1, w - 1):
                for dy in [-1, 0, 1]:
                    self.obstacles.discard((x, h // 2 + dy))

        def reset(self):
            self.position = self.start
            return {'position': self.position, 'goal': self.goal, 'obstacles': list(self.obstacles)}

        def step(self, action):
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
            return {'position': self.position, 'goal': self.goal, 'obstacles': list(self.obstacles)}, 0, self.position == self.goal

    # 运行测试
    env = SimpleEnv(80, 80, 0.3)
    nav = DetailedProfiler(env)

    state = env.reset()
    for i in range(500):
        action = nav.solve(state)
        state, reward, done = env.step(action)
        if done:
            break

    # 分析 spikes
    records = nav.records
    total_times = [r['total_ms'] for r in records]
    mean_time = np.mean(total_times)
    std_time = np.std(total_times)
    threshold = mean_time + 3 * std_time

    spikes = [r for r in records if r['total_ms'] > threshold]

    print(f"\n--- Statistics ---")
    print(f"  Total steps: {len(records)}")
    print(f"  Mean step time: {mean_time:.2f}ms")
    print(f"  Std step time: {std_time:.2f}ms")
    print(f"  Max step time: {np.max(total_times):.2f}ms")
    print(f"  Spike threshold (mean + 3*std): {threshold:.2f}ms")
    print(f"  Number of spikes: {len(spikes)}")

    if spikes:
        print(f"\n--- Spike Details ---")
        for r in spikes[:10]:
            print(f"\n  Step {r['step']}:")
            print(f"    total={r['total_ms']:.1f}ms")
            print(f"    perceive={r['perceive_ms']:.1f}ms")
            print(f"    get_best={r['get_best_ms']:.1f}ms")
            print(f"    get_space={r['get_space_ms']:.1f}ms")
            print(f"    space_update={r['space_update_ms']:.1f}ms")
            print(f"    cache_check={r['cache_check_ms']:.1f}ms")
            print(f"    astar={r.get('astar_ms', 0):.1f}ms")
            print(f"    cache_hit={r['cache_hit']}")
            print(f"    pool_size={r['pool_size']}")
            print(f"    space={r['space']}")

    # 分析各组件的平均时间
    print(f"\n--- Component Averages ---")
    components = ['perceive_ms', 'get_best_ms', 'get_space_ms', 'space_update_ms', 'cache_check_ms']
    for comp in components:
        times = [r[comp] for r in records]
        print(f"  {comp:20s}: avg={np.mean(times):.2f}ms, max={np.max(times):.2f}ms")

    astar_times = [r.get('astar_ms', 0) for r in records if 'astar_ms' in r]
    if astar_times:
        print(f"  {'astar_ms':20s}: avg={np.mean(astar_times):.2f}ms, max={np.max(astar_times):.2f}ms")

    # 分析 cache hit 率
    cache_hits = sum(1 for r in records if r['cache_hit'])
    print(f"\n--- Cache Statistics ---")
    print(f"  Cache hits: {cache_hits}/{len(records)} ({cache_hits/len(records)*100:.1f}%)")


if __name__ == "__main__":
    analyze_spikes()
    print("\n" + "=" * 80)
    print("Spike Analysis Complete")
    print("=" * 80)
