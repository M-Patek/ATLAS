"""
复杂基准测试：验证理论边界切换与结构发现

核心场景：
1. 多走廊交叉（结构切换）
2. 动态障碍（时序预测）
3. 多目标切换（目标变化时结构复用）
4. 大规模迷宫（50x50，结构层次）
5. 非对称地形（Finsler 优势）
6. 信息缺失（部分观测下的结构推断）

核心指标：
- 结构发现率 = 发现的独特结构数 / 场景中的真实结构数
- 结构复用率 = 复用次数 / 总步数
- 首次发现步数 = 发现第一个稳定结构需要的步数
- 适应度收敛速度 = fitness 达到 0.8 需要的竞争次数
- 空间切换效率 = SSFR推荐 vs UCB 的性能差距
- 计算开销比 = SSFR时间 / 基线时间
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Set, Optional
from collections import defaultdict
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from atlas.core.registry import create_space
from atlas.core.solver import GeodesicSolver
from atlas.core.ssfr_enhanced import SSFREnhanced, StructureHypothesis
from atlas.core.path_planning import astar_path, greedy_step, action_from_positions
from atlas.spaces.theory_boundary import TheoryBoundarySpace


# ============================================================================
# 1. 复杂场景定义
# ============================================================================

class ComplexMazeEnv:
    """
    复杂迷宫环境

    场景类型：
    - multi_corridor: 多走廊交叉
    - dynamic_obstacles: 动态障碍
    - multi_goal: 多目标切换
    - large_scale: 大规模迷宫
    - asymmetric: 非对称地形
    - partial_obs: 信息缺失
    """

    def __init__(self, width: int = 30, height: int = 30, maze_type: str = 'multi_corridor'):
        self.width = width
        self.height = height
        self.maze_type = maze_type

        self.start = (1, height // 2)
        self.goal = (width - 2, height // 2)
        self.position = self.start

        # 生成障碍物
        self.obstacles = self._generate_obstacles()
        self.dynamic_obstacles = set()  # 动态障碍
        self.goals = [self.goal]  # 多目标
        self.current_goal_idx = 0

    def _generate_obstacles(self) -> Set[Tuple[int, int]]:
        """生成障碍物"""
        obstacles = set()

        if self.maze_type == 'multi_corridor':
            # 多走廊交叉
            # 主走廊
            for x in range(self.width):
                obstacles.add((x, 0))
                obstacles.add((x, self.height - 1))
            # 垂直走廊
            for y in range(self.height):
                obstacles.add((0, y))
                obstacles.add((self.width - 1, y))
            # 内部走廊
            for x in range(5, self.width - 5):
                obstacles.add((x, self.height // 3))
                obstacles.add((x, 2 * self.height // 3))
            for y in range(5, self.height - 5):
                obstacles.add((self.width // 3, y))
                obstacles.add((2 * self.width // 3, y))

        elif self.maze_type == 'dynamic_obstacles':
            # 动态障碍场景
            for x in range(self.width):
                obstacles.add((x, 0))
                obstacles.add((x, self.height - 1))
            # 静态障碍
            for y in range(5, self.height - 5):
                obstacles.add((self.width // 2, y))

        elif self.maze_type == 'multi_goal':
            # 多目标场景
            for x in range(self.width):
                obstacles.add((x, 0))
                obstacles.add((x, self.height - 1))
            self.goals = [
                (self.width - 2, self.height // 4),
                (self.width - 2, self.height // 2),
                (self.width - 2, 3 * self.height // 4),
            ]
            self.goal = self.goals[0]

        elif self.maze_type == 'large_scale':
            # 大规模迷宫 - 确保有通路
            # 使用随机障碍物 + 保证主通道
            import random
            random.seed(42)
            # 边界
            for x in range(self.width):
                obstacles.add((x, 0))
                obstacles.add((x, self.height - 1))
            for y in range(self.height):
                obstacles.add((0, y))
                obstacles.add((self.width - 1, y))
            # 随机内部障碍（10%密度）
            for x in range(2, self.width - 2):
                for y in range(2, self.height - 2):
                    if random.random() < 0.1:
                        # 不阻挡主通道
                        if not (x == self.width // 2 or y == self.height // 2):
                            obstacles.add((x, y))

        elif self.maze_type == 'asymmetric':
            # 非对称地形
            for x in range(self.width):
                obstacles.add((x, 0))
                obstacles.add((x, self.height - 1))
            # 左侧密集障碍
            for x in range(2, self.width // 3):
                for y in range(2, self.height - 2):
                    if (x + y) % 3 == 0:
                        obstacles.add((x, y))

        elif self.maze_type == 'partial_obs':
            # 信息缺失场景 - 使用随机障碍但保证有通路
            for x in range(self.width):
                obstacles.add((x, 0))
                obstacles.add((x, self.height - 1))
            # 随机障碍（3%密度，确保有通路）
            import random
            random.seed(42)
            for _ in range(self.width * self.height // 30):
                ox = random.randint(1, self.width - 2)
                oy = random.randint(1, self.height - 2)
                # 不阻挡主通道
                if not (ox == self.width // 2 or oy == self.height // 2):
                    obstacles.add((ox, oy))

        return obstacles

    def reset(self):
        self.position = self.start
        self.current_goal_idx = 0
        if self.maze_type == 'multi_goal':
            self.goal = self.goals[0]
        return self.get_state()

    def get_state(self) -> Dict:
        return {
            'position': self.position,
            'goal': self.goal,
            'obstacles': self.obstacles,
            'dynamic_obstacles': self.dynamic_obstacles,
        }

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

        # 障碍检查
        if new_pos in self.obstacles or new_pos in self.dynamic_obstacles:
            new_pos = self.position

        self.position = new_pos

        # 动态障碍更新
        if self.maze_type == 'dynamic_obstacles':
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
            reward = 100.0

        return self.get_state(), reward, done

    def _update_dynamic_obstacles(self):
        """更新动态障碍"""
        # 简单模拟：每隔几步移动障碍
        if self.position[0] % 3 == 0:
            # 清除旧动态障碍
            self.dynamic_obstacles.clear()
            # 添加新动态障碍
            for y in range(2, self.height - 2):
                if y % 4 == self.position[0] % 4:
                    self.dynamic_obstacles.add((self.width // 2 + 1, y))

    def get_possible_actions(self, state: Dict) -> List[str]:
        """获取可能的动作"""
        pos = state['position']
        actions = []

        for action, (dx, dy) in [('up', (0, -1)), ('down', (0, 1)),
                                  ('left', (-1, 0)), ('right', (1, 0))]:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if (0 <= new_pos[0] < self.width and
                0 <= new_pos[1] < self.height and
                new_pos not in self.obstacles and
                new_pos not in self.dynamic_obstacles):
                actions.append(action)

        return actions if actions else ['right']


# ============================================================================
# 2. 导航器
# ============================================================================

class SSFRNavigator:
    """SSFR导航器（深度优化版 + 结构复用）"""

    def __init__(self, env: ComplexMazeEnv,
                 space_names: List[str] = None,
                 use_theory_boundary: bool = False,
                 reuse_threshold: float = 0.85):
        self.env = env
        self.use_theory_boundary = use_theory_boundary

        # 初始化SSFR（带复用机制）
        self.ssfr = SSFREnhanced(
            width=env.width,
            height=env.height,
            space_names=space_names or ['ricci', 'fisher', 'wasserstein', 'conformal'],
            max_structures=50,
            evolution_interval=5,
            reuse_threshold=reuse_threshold
        )

        # 理论边界空间
        self.boundary_space = None
        if use_theory_boundary:
            spaces = {
                name: space for name, space in self.ssfr.spaces.items()
            }
            self.boundary_space = TheoryBoundarySpace(
                env.width, env.height,
                spaces=spaces,
                validity_threshold=0.3
            )

        # 当前空间
        self.current_space_name = 'euclidean'
        self.spaces = {}

        # 路径缓存（关键优化）
        self._cached_path: List[Tuple[int, int]] = []
        self._cached_obstacles: Set[Tuple[int, int]] = set()
        self._cached_goal: Optional[Tuple[int, int]] = None
        self._path_valid = False

        # 统计
        self.structure_reuse_count = 0
        self.structure_discovery_count = 0
        self.space_switches = 0
        self.step_times = []
        self.astar_calls = 0
        self.cache_hits = 0

    def _get_space(self, name: str):
        """获取或创建空间"""
        if name not in self.spaces:
            try:
                self.spaces[name] = create_space(name, self.env.width, self.env.height)
            except:
                self.spaces[name] = create_space('euclidean', self.env.width, self.env.height)
        return self.spaces.get(name)

    def _is_path_valid(self, path: List[Tuple[int, int]],
                       obstacles: Set[Tuple[int, int]],
                       goal: Tuple[int, int]) -> bool:
        """检查缓存路径是否仍然有效"""
        if not path or len(path) < 2:
            return False
        if self._cached_goal != goal:
            return False
        # 检查路径上是否有新障碍
        for pos in path:
            if pos in obstacles:
                return False
        return True

    def solve(self, state: Dict) -> str:
        """基于SSFR推荐求解（深度优化版 + 复用追踪）"""
        start = time.time()

        pos = state['position']
        goal = state['goal']
        obstacles = set(state['obstacles']) | set(state.get('dynamic_obstacles', []))

        observation = {
            'position': pos,
            'goal_position': goal,
            'obstacles': list(state['obstacles']),
        }

        # SSFR 感知（优化：只感知当前活跃空间）
        try:
            self.ssfr.perceive(pos, observation, active_space_name=self.current_space_name)
        except Exception:
            pass

        # 获取最佳结构，决定使用哪个空间
        best_structures = self.ssfr.get_best_structures(n=1)

        if best_structures:
            best = best_structures[0]
            self.structure_discovery_count += 1

            recommended_space = 'euclidean'
            if best.representations:
                recommended_space = list(best.representations.keys())[0]

            # 空间切换时清除路径缓存
            if self.current_space_name != recommended_space:
                self._path_valid = False
                self.space_switches += 1

            self.current_space_name = recommended_space

        # 使用选中的空间求解
        space = self._get_space(self.current_space_name)

        try:
            # 只在空间切换时更新（避免每步重复更新）
            if not self._path_valid:
                space.update_from_observation(pos, observation)

            # 检查缓存路径是否有效
            if self._is_path_valid(self._cached_path, obstacles, goal):
                # 缓存命中！直接沿路径走
                self.cache_hits += 1
                current_idx = self._cached_path.index(pos) if pos in self._cached_path else -1
                if current_idx >= 0 and current_idx + 1 < len(self._cached_path):
                    next_pos = self._cached_path[current_idx + 1]
                    action = action_from_positions(pos, next_pos)
                    elapsed = time.time() - start
                    self.step_times.append(elapsed)
                    return action
                else:
                    self._path_valid = False
            else:
                self._path_valid = False

            # 缓存未命中，重新规划（A*调用）
            self.astar_calls += 1
            if not self._path_valid:
                path = astar_path(
                    space, pos, goal, obstacles,
                    self.env.width, self.env.height,
                    max_steps=500
                )

                if path and len(path) > 1:
                    # 缓存路径
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

            elapsed = time.time() - start
            self.step_times.append(elapsed)

            return action

        except Exception:
            actions = self.env.get_possible_actions(state)
            return actions[0] if actions else 'right'

    def get_stats(self) -> Dict:
        """获取统计"""
        ssfr_stats = self.ssfr.get_statistics()
        total_steps = len(self.step_times)
        return {
            'ssfr_stats': ssfr_stats,
            'structure_discovery_count': self.structure_discovery_count,
            'current_space': self.current_space_name,
            'avg_step_time_ms': np.mean(self.step_times) * 1000 if self.step_times else 0,
            'astar_call_rate': self.astar_calls / total_steps if total_steps > 0 else 0,
            'cache_hit_rate': self.cache_hits / total_steps if total_steps > 0 else 0,
            'space_switches': self.space_switches,
            'reuse_rate': ssfr_stats.get('reuse_rate', 0),
        }


# ============================================================================
# 3. 实验框架
# ============================================================================

def run_episode(env: ComplexMazeEnv, navigator,
                max_steps: int = 500) -> Dict[str, Any]:
    """运行一回合"""
    state = env.reset()
    total_reward = 0.0
    start_time = time.time()

    for step in range(max_steps):
        action = navigator.solve(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state

        if done:
            break

    elapsed = (time.time() - start_time) * 1000

    return {
        'success': env.position == env.goal,
        'steps': step + 1,
        'total_reward': total_reward,
        'time_ms': elapsed,
    }


def run_benchmark(maze_type: str = 'multi_corridor',
                  num_episodes: int = 3,
                  reuse_threshold: float = 0.85):
    """运行基准测试"""
    print("=" * 70)
    print(f"Complex Benchmark: {maze_type} (reuse_threshold={reuse_threshold})")
    print("=" * 70)

    # 环境参数
    if maze_type == 'large_scale':
        width, height = 50, 50
    else:
        width, height = 30, 30

    # 创建环境
    env = ComplexMazeEnv(width=width, height=height, maze_type=maze_type)

    # 创建导航器
    nav = SSFRNavigator(env, use_theory_boundary=True, reuse_threshold=reuse_threshold)

    results = []

    for ep in range(num_episodes):
        result = run_episode(env, nav)
        results.append(result)
        print(f"  Episode {ep+1}: success={result['success']}, "
              f"steps={result['steps']}, reward={result['total_reward']:.1f}, "
              f"time={result['time_ms']:.1f}ms")

    # 汇总
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    successes = sum(1 for r in results if r['success'])
    avg_steps = np.mean([r['steps'] for r in results])
    avg_reward = np.mean([r['total_reward'] for r in results])
    avg_time = np.mean([r['time_ms'] for r in results])

    print(f"Success Rate: {successes}/{len(results)} ({successes/len(results)*100:.1f}%)")
    print(f"Avg Steps: {avg_steps:.1f}")
    print(f"Avg Reward: {avg_reward:.2f}")
    print(f"Avg Time: {avg_time:.2f}ms")

    # SSFR 统计
    stats = nav.get_stats()
    print(f"\nSSFR Statistics:")
    print(f"  Structures discovered: {stats['structure_discovery_count']}")
    print(f"  Pool size: {stats['ssfr_stats']['pool_stats']['num_structures']}")
    print(f"  Reuse rate: {stats['reuse_rate']*100:.1f}%")
    print(f"  A* call rate: {stats['astar_call_rate']*100:.1f}%")
    print(f"  Cache hit rate: {stats['cache_hit_rate']*100:.1f}%")
    print(f"  Space switches: {stats['space_switches']}")
    print(f"  Avg step time: {stats['avg_step_time_ms']:.2f}ms")

    return results, stats


def compare_reuse_scenarios():
    """对比有复用和无复用在复杂场景中的表现"""
    print("\n" + "=" * 70)
    print("STRUCTURE REUSE COMPARISON IN COMPLEX SCENARIOS")
    print("=" * 70)

    scenarios = [
        'multi_corridor',
        'dynamic_obstacles',
        'multi_goal',
        'large_scale',
        'asymmetric',
        'partial_obs',
    ]

    all_results = {}

    for scenario in scenarios:
        print(f"\n--- Scenario: {scenario} ---")

        # 有复用
        results_with, stats_with = run_benchmark(
            maze_type=scenario, num_episodes=2, reuse_threshold=0.85
        )

        # 无复用（阈值 > 1，永不复用）
        results_without, stats_without = run_benchmark(
            maze_type=scenario, num_episodes=2, reuse_threshold=1.1
        )

        all_results[scenario] = {
            'with': (results_with, stats_with),
            'without': (results_without, stats_without),
        }

        # 对比
        print(f"\n  Comparison:")
        print(f"    With reuse:    {stats_with['reuse_rate']*100:.1f}% reuse, "
              f"{stats_with['astar_call_rate']*100:.1f}% A*, "
              f"{stats_with['ssfr_stats']['pool_stats']['num_structures']} structures")
        print(f"    Without reuse: {stats_without['reuse_rate']*100:.1f}% reuse, "
              f"{stats_without['astar_call_rate']*100:.1f}% A*, "
              f"{stats_without['ssfr_stats']['pool_stats']['num_structures']} structures")

        # 计算改善
        struct_reduction = (1 - stats_with['ssfr_stats']['pool_stats']['num_structures'] /
                           max(1, stats_without['ssfr_stats']['pool_stats']['num_structures'])) * 100
        print(f"    Structure reduction: {struct_reduction:.1f}%")

    # 总体汇总
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    for scenario, data in all_results.items():
        stats_with = data['with'][1]
        stats_without = data['without'][1]

        print(f"\n{scenario}:")
        print(f"  Reuse rate: {stats_with['reuse_rate']*100:.1f}%")
        print(f"  A* call rate: {stats_with['astar_call_rate']*100:.1f}%")
        print(f"  Cache hit rate: {stats_with['cache_hit_rate']*100:.1f}%")
        print(f"  Pool size: {stats_with['ssfr_stats']['pool_stats']['num_structures']} "
              f"(vs {stats_without['ssfr_stats']['pool_stats']['num_structures']} without reuse)")

    return all_results


# ============================================================================
# 4. 主入口
# ============================================================================

if __name__ == "__main__":
    # 运行对比实验
    compare_reuse_scenarios()

    print("\n" + "=" * 70)
    print("Complex Benchmark Complete")
    print("=" * 70)
