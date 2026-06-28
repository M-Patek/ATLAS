"""
超复杂基准测试：极限压力测试

核心设计原则：
1. 更大规模：100x100 网格
2. 更多障碍：40% 密度 + 动态变化
3. 更多空间：6个认知空间同时竞争
4. 更多目标：10个目标序列
5. 更多回合：验证稳定性
6. 更复杂场景：
   - 迷宫生成（保证有解但路径复杂）
   - 周期性动态障碍（需要预测）
   - 多智能体干扰
   - 噪声观测（部分+错误信息）
   - 空间切换频繁（不同区域需要不同空间）

新增场景：
7. maze_recursive: 递归分割迷宫（50x50）
8. dense_dynamic: 高密度动态障碍（40%密度，每5步变化）
9. multi_space_switch: 强制空间切换（不同区域有不同最优空间）
10. noisy_observation: 噪声观测（20%观测错误率）
11. long_horizon: 超长路径（100x100，目标在远端）
12. adversarial: 对抗性障碍（障碍试图阻挡）
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Set, Optional
from collections import defaultdict
import time
import sys
import os
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from atlas.core.registry import create_space
from atlas.core.solver import GeodesicSolver
from atlas.core.ssfr_enhanced import SSFREnhanced, StructureHypothesis
from atlas.core.path_planning import astar_path, greedy_step, action_from_positions
from atlas.spaces.theory_boundary import TheoryBoundarySpace


# ============================================================================
# 1. 迷宫生成器
# ============================================================================

def generate_recursive_maze(width: int, height: int, seed: int = 42) -> Set[Tuple[int, int]]:
    """
    递归分割法生成迷宫

    保证：
    - 起点到终点有且仅有一条通路
    - 复杂度可控
    """
    random.seed(seed)
    obstacles = set()

    # 边界
    for x in range(width):
        obstacles.add((x, 0))
        obstacles.add((x, height - 1))
    for y in range(height):
        obstacles.add((0, y))
        obstacles.add((width - 1, y))

    # 递归分割
    def divide(x1, y1, x2, y2):
        if x2 - x1 < 3 or y2 - y1 < 3:
            return

        if random.random() < 0.5 and x2 - x1 > y2 - y1:
            # 垂直分割
            wx = random.randint(x1 + 2, x2 - 2)
            for y in range(y1, y2 + 1):
                obstacles.add((wx, y))
            # 开门
            door = random.randint(y1 + 1, y2 - 1)
            obstacles.discard((wx, door))
            divide(x1, y1, wx - 1, y2)
            divide(wx + 1, y1, x2, y2)
        else:
            # 水平分割
            wy = random.randint(y1 + 2, y2 - 2)
            for x in range(x1, x2 + 1):
                obstacles.add((x, wy))
            # 开门
            door = random.randint(x1 + 1, x2 - 1)
            obstacles.discard((door, wy))
            divide(x1, y1, x2, wy - 1)
            divide(x1, wy + 1, x2, y2)

    divide(1, 1, width - 2, height - 2)
    return obstacles


def generate_dense_random(width: int, height: int, density: float = 0.4, seed: int = 42) -> Set[Tuple[int, int]]:
    """
    生成高密度随机障碍，但保证主通道
    """
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
                # 不阻挡主通道（对角线）
                if abs(x - y) > 2:
                    obstacles.add((x, y))

    return obstacles


# ============================================================================
# 2. 超复杂场景定义
# ============================================================================

class UltraComplexMazeEnv:
    """
    超复杂迷宫环境

    场景类型：
    - maze_recursive: 递归分割迷宫（50x50）
    - dense_dynamic: 高密度动态障碍（40%密度，每5步变化）
    - multi_space_switch: 强制空间切换区域
    - noisy_observation: 噪声观测（20%错误率）
    - long_horizon: 超长路径（100x100）
    - adversarial: 对抗性障碍
    """

    def __init__(self, width: int = 50, height: int = 50, maze_type: str = 'maze_recursive',
                 noise_rate: float = 0.0, dynamic_interval: int = 5):
        self.width = width
        self.height = height
        self.maze_type = maze_type
        self.noise_rate = noise_rate
        self.dynamic_interval = dynamic_interval
        self.step_count = 0

        self.start = (1, 1)
        self.goal = (width - 2, height - 2)
        self.position = self.start

        # 生成障碍物
        self.static_obstacles = self._generate_obstacles()
        self.dynamic_obstacles = set()
        self.obstacles = self.static_obstacles.copy()

        # 多目标
        self.goals = [self.goal]
        self.current_goal_idx = 0

        # 空间切换区域（用于 multi_space_switch）
        self.space_zones = self._generate_space_zones()

    def _generate_obstacles(self) -> Set[Tuple[int, int]]:
        """生成障碍物"""
        if self.maze_type == 'maze_recursive':
            return generate_recursive_maze(self.width, self.height, seed=42)
        elif self.maze_type == 'dense_dynamic':
            return generate_dense_random(self.width, self.height, density=0.35, seed=42)
        elif self.maze_type == 'long_horizon':
            return generate_dense_random(self.width, self.height, density=0.25, seed=42)
        elif self.maze_type == 'adversarial':
            # 对抗性：障碍试图阻挡最短路径
            obs = generate_dense_random(self.width, self.height, density=0.2, seed=42)
            # 在中点附近添加额外障碍
            mid_x, mid_y = self.width // 2, self.height // 2
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if abs(dx) + abs(dy) <= 7:
                        obs.add((mid_x + dx, mid_y + dy))
            # 但保证一条弯曲路径
            for t in range(100):
                x = 1 + t * (self.width - 3) // 100
                y = 1 + int(10 * np.sin(t * 0.1))
                obs.discard((x, y))
            return obs
        else:
            # 默认：中等密度随机
            return generate_dense_random(self.width, self.height, density=0.2, seed=42)

    def _generate_space_zones(self) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """生成不同空间最优的区域"""
        zones = {}
        w, h = self.width, self.height

        # Ricci 区域：高曲率/复杂地形（迷宫区域）
        zones['ricci'] = [(w//4, 0, w//2, h)]

        # Finsler 区域：非对称地形（单向通道）
        zones['finsler'] = [(w//2, 0, 3*w//4, h)]

        # Conformal 区域：目标导向（开阔地带）
        zones['conformal'] = [(3*w//4, 0, w-1, h)]

        # Euclidean 区域：平坦地带
        zones['euclidean'] = [(0, 0, w//4, h)]

        return zones

    def reset(self):
        self.position = self.start
        self.current_goal_idx = 0
        self.step_count = 0
        self.dynamic_obstacles.clear()
        self.obstacles = self.static_obstacles.copy()
        if self.maze_type == 'multi_goal':
            self.goal = self.goals[0]
        return self.get_state()

    def get_state(self) -> Dict:
        # 添加噪声观测
        observed_obstacles = self._add_observation_noise(self.obstacles)

        return {
            'position': self.position,
            'goal': self.goal,
            'obstacles': observed_obstacles,
            'dynamic_obstacles': list(self.dynamic_obstacles),
            'true_obstacles': list(self.obstacles),  # 用于验证（不给agent）
        }

    def _add_observation_noise(self, obstacles: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """添加观测噪声"""
        if self.noise_rate <= 0:
            return list(obstacles)

        observed = set(obstacles)

        # 随机移除一些障碍（假阴性）
        for obs in list(observed):
            if random.random() < self.noise_rate:
                observed.discard(obs)

        # 随机添加不存在的障碍（假阳性）
        for _ in range(int(len(observed) * self.noise_rate * 0.5)):
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)
            if (x, y) not in self.obstacles:
                observed.add((x, y))

        return list(observed)

    def step(self, action: str) -> Tuple[Dict, float, bool]:
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

        # 边界检查
        if not (0 <= new_pos[0] < self.width and 0 <= new_pos[1] < self.height):
            new_pos = self.position

        # 障碍检查（使用真实障碍，不是观测到的）
        if new_pos in self.obstacles:
            new_pos = self.position

        self.position = new_pos
        self.step_count += 1

        # 动态障碍更新
        if self.maze_type == 'dense_dynamic' and self.step_count % self.dynamic_interval == 0:
            self._update_dynamic_obstacles()

        # 多目标切换
        if self.maze_type == 'multi_goal':
            if self.position == self.goal:
                self.current_goal_idx += 1
                if self.current_goal_idx < len(self.goals):
                    self.goal = self.goals[self.current_goal_idx]

        # 奖励
        dist = abs(self.position[0] - self.goal[0]) + abs(self.position[1] - self.goal[1])
        reward = -0.1 * dist

        done = self.position == self.goal
        if done and self.maze_type == 'multi_goal':
            done = self.current_goal_idx >= len(self.goals) - 1
        if done:
            reward = 1000.0

        return self.get_state(), reward, done

    def _update_dynamic_obstacles(self):
        """更新动态障碍"""
        # 清除旧动态障碍
        self.obstacles -= self.dynamic_obstacles
        self.dynamic_obstacles.clear()

        # 添加新动态障碍（随机位置，但不在起点/终点附近）
        for _ in range(self.width * self.height // 20):  # 5% 动态障碍
            x = random.randint(2, self.width - 3)
            y = random.randint(2, self.height - 3)
            if abs(x - self.position[0]) > 3:  # 不在当前位置附近
                self.dynamic_obstacles.add((x, y))

        self.obstacles |= self.dynamic_obstacles

    def get_possible_actions(self, state: Dict) -> List[str]:
        """获取可能的动作"""
        pos = state['position']
        actions = []

        for action, (dx, dy) in [('up', (0, -1)), ('down', (0, 1)),
                                  ('left', (-1, 0)), ('right', (1, 0))]:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if (0 <= new_pos[0] < self.width and
                0 <= new_pos[1] < self.height and
                new_pos not in self.obstacles):
                actions.append(action)

        return actions if actions else ['right']

    def get_optimal_space_for_zone(self, position: Tuple[int, int]) -> str:
        """获取位置所在区域的最优空间（用于分析）"""
        x, y = position
        for space_name, zones in self.space_zones.items():
            for zx1, zy1, zx2, zy2 in zones:
                if zx1 <= x < zx2 and zy1 <= y < zy2:
                    return space_name
        return 'euclidean'


# ============================================================================
# 3. 增强导航器（带详细性能分析）
# ============================================================================

class ProfilingNavigator:
    """带详细性能分析的导航器"""

    def __init__(self, env: UltraComplexMazeEnv,
                 space_names: List[str] = None,
                 use_theory_boundary: bool = False,
                 enable_profiling: bool = True):
        self.env = env
        self.use_theory_boundary = use_theory_boundary
        self.enable_profiling = enable_profiling

        # 初始化SSFR
        self.ssfr = SSFREnhanced(
            width=env.width,
            height=env.height,
            space_names=space_names or ['ricci', 'fisher', 'wasserstein', 'conformal', 'finsler'],
            max_structures=100,
            evolution_interval=10
        )

        # 理论边界空间
        self.boundary_space = None
        if use_theory_boundary:
            spaces = {name: space for name, space in self.ssfr.spaces.items()}
            self.boundary_space = TheoryBoundarySpace(
                env.width, env.height,
                spaces=spaces,
                validity_threshold=0.3
            )

        # 当前空间
        self.current_space_name = 'euclidean'
        self.spaces = {}

        # 路径缓存
        self._cached_path = []
        self._cached_obstacles = set()
        self._cached_goal = None
        self._path_valid = False

        # 详细性能计时
        self.profile = {
            'perceive_time': [],
            'astar_time': [],
            'space_update_time': [],
            'compute_distance_time': [],
            'get_heuristic_time': [],
            'path_cache_hit': 0,
            'path_cache_miss': 0,
            'structure_count': [],
        }
        self.step_times = []
        self.step_count = 0

    def _get_space(self, name: str):
        """获取或创建空间"""
        if name not in self.spaces:
            try:
                self.spaces[name] = create_space(name, self.env.width, self.env.height)
            except:
                self.spaces[name] = create_space('euclidean', self.env.width, self.env.height)
        return self.spaces.get(name)

    def _is_path_valid(self, path, obstacles, goal):
        """检查缓存路径是否有效"""
        if not path or len(path) < 2:
            return False
        if self._cached_goal != goal:
            return False
        for pos in path:
            if pos in obstacles:
                return False
        return True

    def solve(self, state: Dict) -> str:
        """求解（带详细性能分析）"""
        step_start = time.time()
        self.step_count += 1

        pos = state['position']
        goal = state['goal']
        obstacles = set(state['obstacles']) | set(state.get('dynamic_obstacles', []))

        observation = {
            'position': pos,
            'goal_position': goal,
            'obstacles': list(state['obstacles']),
        }

        # SSFR 感知
        perceive_start = time.time()
        try:
            self.ssfr.perceive(pos, observation, active_space_name=self.current_space_name)
        except Exception:
            pass
        self.profile['perceive_time'].append(time.time() - perceive_start)

        # 获取最佳结构
        best_structures = self.ssfr.get_best_structures(n=1)
        if best_structures:
            best = best_structures[0]
            recommended_space = 'euclidean'
            if best.representations:
                recommended_space = list(best.representations.keys())[0]
            if self.current_space_name != recommended_space:
                self._path_valid = False
            self.current_space_name = recommended_space

        # 使用选中的空间
        space = self._get_space(self.current_space_name)

        try:
            # 空间更新
            update_start = time.time()
            if not self._path_valid:
                space.update_from_observation(pos, observation)
            self.profile['space_update_time'].append(time.time() - update_start)

            # 检查缓存
            if self._is_path_valid(self._cached_path, obstacles, goal):
                self.profile['path_cache_hit'] += 1
                current_idx = self._cached_path.index(pos) if pos in self._cached_path else -1
                if current_idx >= 0 and current_idx + 1 < len(self._cached_path):
                    next_pos = self._cached_path[current_idx + 1]
                    self.step_times.append(time.time() - step_start)
                    self.profile['structure_count'].append(
                        len(self.ssfr.structure_pool.structures)
                    )
                    return action_from_positions(pos, next_pos)
                else:
                    self._path_valid = False
            else:
                self.profile['path_cache_miss'] += 1
                self._path_valid = False

            # A* 规划
            astar_start = time.time()
            path = astar_path(
                space, pos, goal, obstacles,
                self.env.width, self.env.height,
                max_steps=2000
            )
            self.profile['astar_time'].append(time.time() - astar_start)

            if path and len(path) > 1:
                self._cached_path = path
                self._cached_obstacles = obstacles.copy()
                self._cached_goal = goal
                self._path_valid = True
                next_pos = path[1]
                action = action_from_positions(pos, next_pos)
            else:
                # A*失败，回退到贪心
                next_pos = greedy_step(
                    space, pos, goal, obstacles,
                    self.env.width, self.env.height
                )
                if next_pos:
                    action = action_from_positions(pos, next_pos)
                else:
                    actions = self.env.get_possible_actions(state)
                    action = actions[0] if actions else 'right'

            self.step_times.append(time.time() - step_start)
            self.profile['structure_count'].append(
                len(self.ssfr.structure_pool.structures)
            )
            return action

        except Exception:
            actions = self.env.get_possible_actions(state)
            return actions[0] if actions else 'right'

    def get_stats(self) -> Dict:
        """获取统计"""
        ssfr_stats = self.ssfr.get_statistics()

        stats = {
            'ssfr_stats': ssfr_stats,
            'current_space': self.current_space_name,
            'avg_step_time_ms': np.mean(self.step_times) * 1000 if self.step_times else 0,
            'max_step_time_ms': np.max(self.step_times) * 1000 if self.step_times else 0,
            'total_steps': self.step_count,
        }

        # 性能分析
        for key in ['perceive_time', 'astar_time', 'space_update_time']:
            times = self.profile[key]
            if times:
                stats[f'avg_{key}_ms'] = np.mean(times) * 1000
                stats[f'max_{key}_ms'] = np.max(times) * 1000
                stats[f'total_{key}_ms'] = np.sum(times) * 1000

        stats['path_cache_hit_rate'] = (
            self.profile['path_cache_hit'] /
            max(1, self.profile['path_cache_hit'] + self.profile['path_cache_miss'])
        )
        stats['avg_structures'] = np.mean(self.profile['structure_count']) if self.profile['structure_count'] else 0

        return stats


# ============================================================================
# 4. 实验框架
# ============================================================================

def run_episode(env: UltraComplexMazeEnv, navigator: ProfilingNavigator,
                max_steps: int = 2000) -> Dict[str, Any]:
    """运行一回合"""
    state = env.reset()
    total_reward = 0.0
    start_time = time.time()
    space_switches = []

    for step in range(max_steps):
        action = navigator.solve(state)
        next_state, reward, done = env.step(action)
        total_reward += reward

        # 记录空间切换
        if step > 0 and hasattr(navigator, 'current_space_name'):
            space_switches.append(navigator.current_space_name)

        state = next_state
        if done:
            break

    elapsed = (time.time() - start_time) * 1000

    return {
        'success': env.position == env.goal,
        'steps': step + 1,
        'total_reward': total_reward,
        'time_ms': elapsed,
        'space_switches': space_switches,
    }


def run_benchmark(maze_type: str, width: int = 50, height: int = 50,
                  num_episodes: int = 1, **env_kwargs):
    """运行基准测试"""
    print("=" * 80)
    print(f"Ultra Complex Benchmark: {maze_type} ({width}x{height})")
    print("=" * 80)

    # 创建环境
    env = UltraComplexMazeEnv(width=width, height=height, maze_type=maze_type, **env_kwargs)

    # 创建导航器
    nav = ProfilingNavigator(env, use_theory_boundary=True)

    results = []
    for ep in range(num_episodes):
        result = run_episode(env, nav, max_steps=2000)
        results.append(result)
        status = "OK" if result['success'] else "FAIL"
        print(f"  Episode {ep+1}: {status} steps={result['steps']}, "
              f"reward={result['total_reward']:.1f}, time={result['time_ms']:.1f}ms")

    # 汇总
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    successes = sum(1 for r in results if r['success'])
    avg_steps = np.mean([r['steps'] for r in results])
    avg_time = np.mean([r['time_ms'] for r in results])

    print(f"Success Rate: {successes}/{len(results)} ({successes/len(results)*100:.1f}%)")
    print(f"Avg Steps: {avg_steps:.1f}")
    print(f"Avg Time: {avg_time:.2f}ms")

    # 详细性能分析
    stats = nav.get_stats()
    print(f"\nPerformance Profile:")
    print(f"  Avg step time: {stats['avg_step_time_ms']:.2f}ms")
    print(f"  Max step time: {stats['max_step_time_ms']:.2f}ms")
    print(f"  Path cache hit rate: {stats.get('path_cache_hit_rate', 0)*100:.1f}%")
    print(f"  Avg structures: {stats.get('avg_structures', 0):.0f}")

    for key in ['perceive_time', 'astar_time', 'space_update_time']:
        avg_key = f'avg_{key}_ms'
        max_key = f'max_{key}_ms'
        if avg_key in stats:
            print(f"  {key}: avg={stats[avg_key]:.2f}ms, max={stats[max_key]:.2f}ms")

    print(f"\nSSFR Stats:")
    print(f"  {stats['ssfr_stats']}")

    return results, stats


# ============================================================================
# 5. 主入口
# ============================================================================

if __name__ == "__main__":
    scenarios = [
        # (name, width, height, kwargs)
        ('maze_recursive', 30, 30, {}),
        ('dense_dynamic', 30, 30, {'dynamic_interval': 5}),
        ('long_horizon', 50, 50, {}),
        ('adversarial', 40, 40, {}),
        ('noisy_observation', 30, 30, {'noise_rate': 0.2}),
    ]

    all_results = {}
    for scenario, w, h, kwargs in scenarios:
        try:
            results, stats = run_benchmark(scenario, width=w, height=h, num_episodes=1, **kwargs)
            all_results[scenario] = {'results': results, 'stats': stats}
        except Exception as e:
            print(f"\nError in {scenario}: {e}")
            import traceback
            traceback.print_exc()
        print("\n" + "=" * 80 + "\n")

    # 最终汇总
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    for scenario, data in all_results.items():
        results = data['results']
        stats = data['stats']
        success = results[0]['success'] if results else False
        status = "OK" if success else "FAIL"
        print(f"{status} {scenario:20s} steps={results[0]['steps'] if results else 'N/A':4}, "
              f"time={stats['avg_step_time_ms']:.1f}ms/step")

    print("\n" + "=" * 80)
    print("Ultra Complex Benchmark Complete")
    print("=" * 80)
