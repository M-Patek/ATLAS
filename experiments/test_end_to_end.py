"""
端到端场景验证：走廊与迷宫

验证 SSFR 的核心价值：
1. 结构发现：在环境中发现稳定结构（走廊、拐角、死胡同）
2. 结构复用：遇到相似场景时直接激活旧结构
3. 性能提升：第二圈比第一圈快

对比基线：
- 无SSFR：传统A*，每一步重新规划
- 有SSFR：基于发现的结构加速规划
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


# ============================================================================
# 1. 场景定义
# ============================================================================

class CorridorMazeEnv:
    """
    走廊与迷宫环境

    场景布局：
    S = 起点
    G = 目标
    # = 墙
    . = 空地

    走廊场景 (20x5):
    ####################
    S..................G
    ####################

    迷宫场景 (15x15):
    ###############
    S.#.#.#.#.#.G
    #.#.#.#.#.#.#
    .#.#.#.#.#.#.
    #.#.#.#.#.#.#
    .#.#.#.#.#.#.
    #.#.#.#.#.#.#
    .#.#.#.#.#.#.
    #.#.#.#.#.#.#
    .#.#.#.#.#.#.
    #.#.#.#.#.#.#
    .#.#.#.#.#.#.
    #.#.#.#.#.#.#
    .#.#.#.#.#.#.
    #.#.#.#.#.#.#
    ###############
    """

    def __init__(self, width: int = 20, height: int = 5, maze_type: str = 'corridor'):
        self.width = width
        self.height = height
        self.maze_type = maze_type

        self.start = (1, height // 2)
        self.goal = (width - 2, height // 2)
        self.position = self.start

        # 生成障碍物
        self.obstacles = self._generate_obstacles()

    def _generate_obstacles(self) -> Set[Tuple[int, int]]:
        """生成障碍物"""
        obstacles = set()

        if self.maze_type == 'corridor':
            # 走廊：上下墙壁
            for x in range(self.width):
                obstacles.add((x, 0))
                obstacles.add((x, self.height - 1))

        elif self.maze_type == 'maze_simple':
            # 简单迷宫：垂直障碍
            for y in range(self.height):
                for x in range(2, self.width - 2, 4):
                    if y % 2 == 0:
                        obstacles.add((x, y))

        elif self.maze_type == 'maze_complex':
            # 复杂迷宫
            for x in range(self.width):
                obstacles.add((x, 0))
                obstacles.add((x, self.height - 1))
            for y in range(self.height):
                obstacles.add((0, y))
                obstacles.add((self.width - 1, y))

            # 内部障碍
            for x in range(2, self.width - 2, 3):
                for y in range(1, self.height - 1):
                    if (x + y) % 2 == 0:
                        obstacles.add((x, y))

        return obstacles

    def reset(self):
        self.position = self.start
        return self.get_state()

    def get_state(self) -> Dict:
        return {
            'position': self.position,
            'goal': self.goal,
            'obstacles': self.obstacles,
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
        if new_pos in self.obstacles:
            new_pos = self.position

        self.position = new_pos

        # 奖励
        dist = abs(self.position[0] - self.goal[0]) + abs(self.position[1] - self.goal[1])
        reward = -0.1 * dist

        done = self.position == self.goal
        if done:
            reward = 100.0

        return self.get_state(), reward, done

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


# ============================================================================
# 2. 基线：无SSFR的传统A*
# ============================================================================

class BaselineNavigator:
    """
    基线导航器：传统A*

    不使用SSFR，每一步重新规划
    """

    def __init__(self, env: CorridorMazeEnv):
        self.env = env
        self.space = create_space('euclidean', env.width, env.height)
        self.solver = GeodesicSolver(self.space)
        self.path = []
        self.path_index = 0

    def solve(self, state: Dict) -> str:
        """求解下一步动作"""
        pos = state['position']
        goal = state['goal']

        # 重新规划路径
        try:
            result = self.solver.solve(
                start=pos,
                goal=goal,
                obstacles=state['obstacles']
            )

            if result.success and result.path and len(result.path) > 1:
                # 取下一步
                next_pos = result.path[1]
                dx = next_pos[0] - pos[0]
                dy = next_pos[1] - pos[1]

                if dx > 0:
                    return 'right'
                elif dx < 0:
                    return 'left'
                elif dy > 0:
                    return 'down'
                elif dy < 0:
                    return 'up'
        except Exception as e:
            pass

        # 回退：随机选择
        actions = self.env.get_possible_actions(state)
        return actions[0] if actions else 'right'


# ============================================================================
# 3. SSFR导航器
# ============================================================================

class SSFRNavigator:
    """
    SSFR导航器：基于发现的结构加速规划

    核心逻辑：
    1. 使用SSFR感知环境，发现结构
    2. 基于结构推荐选择空间
    3. 使用推荐的空间求解
    """

    def __init__(self, env: CorridorMazeEnv,
                 space_names: List[str] = None):
        self.env = env

        # 初始化SSFR
        self.ssfr = SSFREnhanced(
            width=env.width,
            height=env.height,
            space_names=space_names or ['ricci', 'fisher', 'wasserstein', 'conformal'],
            max_structures=50,
            evolution_interval=5
        )

        # 当前空间
        self.current_space_name = 'euclidean'
        self.spaces = {}

        # 统计
        self.structure_reuse_count = 0
        self.structure_discovery_count = 0

    def _get_space(self, name: str):
        """获取或创建空间"""
        if name not in self.spaces:
            try:
                self.spaces[name] = create_space(name, self.env.width, self.env.height)
            except:
                # 回退到欧氏
                self.spaces[name] = create_space('euclidean', self.env.width, self.env.height)
        return self.spaces.get(name)

    def solve(self, state: Dict) -> str:
        """
        基于SSFR推荐求解

        策略：
        1. SSFR感知 + 竞争 + 演化
        2. 获取最佳结构
        3. 基于结构推荐选择空间
        4. 使用选中的空间求解
        """
        pos = state['position']
        observation = {
            'position': pos,
            'goal_position': state['goal'],
            'obstacles': list(state['obstacles']),
        }

        # SSFR 完整步骤：感知 + 竞争 + 演化
        try:
            # 构建 actual（用于验证）
            actual = {
                'position': pos,
                'obstacles': list(state['obstacles']),
            }
            self.ssfr.step(pos, observation, actual)
        except Exception as e:
            pass

        # 获取最佳结构
        best_structures = self.ssfr.get_best_structures(n=1)

        if best_structures:
            best = best_structures[0]
            self.structure_discovery_count += 1

            # 从结构推荐中提取空间
            recommended_space = 'euclidean'
            if best.representations:
                recommended_space = list(best.representations.keys())[0]

            self.current_space_name = recommended_space

        # 使用选中的空间求解
        space = self._get_space(self.current_space_name)

        try:
            # 更新空间
            space.update_from_observation(pos, observation)

            # 计算到目标的距离
            goal = state['goal']
            best_action = 'right'
            best_dist = float('inf')

            for action in self.env.get_possible_actions(state):
                if action == 'up':
                    next_pos = (pos[0], pos[1] - 1)
                elif action == 'down':
                    next_pos = (pos[0], pos[1] + 1)
                elif action == 'left':
                    next_pos = (pos[0] - 1, pos[1])
                elif action == 'right':
                    next_pos = (pos[0] + 1, pos[1])
                else:
                    continue

                dist = space.compute_distance(next_pos, goal)
                if dist < best_dist:
                    best_dist = dist
                    best_action = action

            return best_action

        except Exception as e:
            # 回退
            actions = self.env.get_possible_actions(state)
            return actions[0] if actions else 'right'

    def get_stats(self) -> Dict:
        """获取统计"""
        ssfr_stats = self.ssfr.get_statistics()
        return {
            'ssfr_stats': ssfr_stats,
            'structure_discovery_count': self.structure_discovery_count,
            'current_space': self.current_space_name,
        }


# ============================================================================
# 4. 实验框架
# ============================================================================

def run_episode(env: CorridorMazeEnv, navigator,
                max_steps: int = 200) -> Dict[str, Any]:
    """
    运行一回合

    Returns:
        {
            'success': bool,
            'steps': int,
            'total_reward': float,
            'time_ms': float,
        }
    """
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

    elapsed = (time.time() - start_time) * 1000  # ms

    return {
        'success': env.position == env.goal,
        'steps': step + 1,
        'total_reward': total_reward,
        'time_ms': elapsed,
    }


def run_comparison_experiment(maze_type: str = 'corridor',
                              num_rounds: int = 3,
                              episodes_per_round: int = 5):
    """
    运行对比实验

    比较：
    - 基线（无SSFR）
    - SSFR导航器

    指标：
    - 成功率
    - 平均步数
    - 平均时间
    - 结构发现数量（仅SSFR）
    """
    print("=" * 70)
    print(f"End-to-End Validation: {maze_type}")
    print("=" * 70)

    # 环境参数
    if maze_type == 'corridor':
        width, height = 20, 5
    else:
        width, height = 15, 15

    # 创建环境
    env = CorridorMazeEnv(width=width, height=height, maze_type=maze_type)

    # 创建导航器
    baseline = BaselineNavigator(env)
    ssfr_nav = SSFRNavigator(env)

    results = {
        'baseline': [],
        'ssfr': [],
    }

    # 运行多轮
    for round_idx in range(num_rounds):
        print(f"\n--- Round {round_idx + 1} ---")

        # 基线
        print("  Baseline:")
        for ep in range(episodes_per_round):
            result = run_episode(env, baseline)
            results['baseline'].append(result)
            print(f"    Episode {ep+1}: success={result['success']}, "
                  f"steps={result['steps']}, reward={result['total_reward']:.1f}")

        # SSFR
        print("  SSFR:")
        for ep in range(episodes_per_round):
            result = run_episode(env, ssfr_nav)
            results['ssfr'].append(result)
            print(f"    Episode {ep+1}: success={result['success']}, "
                  f"steps={result['steps']}, reward={result['total_reward']:.1f}")

    # 汇总
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    for name, data in results.items():
        successes = sum(1 for r in data if r['success'])
        avg_steps = np.mean([r['steps'] for r in data])
        avg_reward = np.mean([r['total_reward'] for r in data])
        avg_time = np.mean([r['time_ms'] for r in data])

        print(f"\n{name.upper()}:")
        print(f"  Success Rate: {successes}/{len(data)} ({successes/len(data)*100:.1f}%)")
        print(f"  Avg Steps: {avg_steps:.1f}")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Avg Time: {avg_time:.2f}ms")

    # SSFR 统计
    if hasattr(ssfr_nav, 'get_stats'):
        ssfr_stats = ssfr_nav.get_stats()
        print(f"\nSSFR Statistics:")
        print(f"  Structures discovered: {ssfr_stats['structure_discovery_count']}")
        print(f"  Pool stats: {ssfr_stats['ssfr_stats']}")

    return results


# ============================================================================
# 5. 主入口
# ============================================================================

if __name__ == "__main__":
    # 走廊场景
    run_comparison_experiment(maze_type='corridor', num_rounds=2, episodes_per_round=3)

    print("\n" + "=" * 70)

    # 简单迷宫
    run_comparison_experiment(maze_type='maze_simple', num_rounds=2, episodes_per_round=3)

    print("\n" + "=" * 70)
    print("End-to-End Validation Complete")
    print("=" * 70)
