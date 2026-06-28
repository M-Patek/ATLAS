"""
Meta-SSFR: 完整的可插拔架构设计

核心设计：
1. ATLAS 空间作为可插拔组件
2. Meta-Learning 选择最优空间
3. SSFR 在空间内部发现结构
4. 完整的实验框架验证

架构层次：
Level 3: Meta-Cognitive (选择空间)
Level 2: Cognitive Space (Ricci/Fisher/Wasserstein/Finsler)
Level 1: Structure Discovery (空间内部结构)
Level 0: Action Execution (执行)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Set, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import sys
import os

# 添加 ATLAS 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from atlas.core.registry import registry, list_available_spaces, create_space
from atlas.core.space import CognitiveSpace, SpaceMetrics
from atlas.core.experiment import Experiment, ConditionResult, TrialResult


# ============================================================================
# 1. 空间适配器：将 ATLAS 空间包装为求解器
# ============================================================================

class SpaceAdapter:
    """
    ATLAS 空间适配器

    将 CognitiveSpace 包装为可用于网格世界的求解器
    """

    def __init__(self, space: CognitiveSpace, name: str):
        self.space = space
        self.name = name
        self.path_history = []

    def solve(self, state: Dict) -> str:
        """
        使用空间求解

        策略：
        1. 更新空间状态
        2. 计算到目标的距离
        3. 选择最小距离的方向
        """
        pos = state.get('position', (0, 0))
        goal = state.get('goal', (9, 9))
        obstacles = state.get('obstacles', set())

        # 更新空间（如果有新观测）
        self._update_space(pos, obstacles, goal)

        # 计算各方向的距离
        best_action = 'right'
        best_distance = float('inf')

        for action in ['up', 'down', 'left', 'right']:
            new_pos = self._get_new_pos(pos, action)
            if new_pos:
                # 使用 ATLAS 空间的 compute_distance
                try:
                    dist = self.space.compute_distance(new_pos, goal)
                except:
                    # 如果空间不支持，回退到曼哈顿
                    dist = abs(new_pos[0] - goal[0]) + abs(new_pos[1] - goal[1])

                if dist < best_distance:
                    best_distance = dist
                    best_action = action

        return best_action

    def _update_space(self, pos: Tuple, obstacles: Set, goal: Tuple = None):
        """更新空间状态"""
        try:
            # 构建观测数据
            observation = {
                'obstacles': list(obstacles),
                'position': pos,
            }

            # 添加目标信息（如果有）
            if goal:
                observation['goal_position'] = goal

            self.space.update_from_observation(pos, observation)
        except Exception as e:
            # 某些空间可能不支持动态更新
            pass

    def _get_new_pos(self, pos, action):
        x, y = pos
        if action == 'up':
            return (x, y + 1)
        elif action == 'down':
            return (x, y - 1)
        elif action == 'left':
            return (x - 1, y)
        elif action == 'right':
            return (x + 1, y)
        return None

    def update(self, state: Dict, action: str, reward: float, next_state: Dict):
        """更新（用于学习）"""
        self.path_history.append((state, action, reward, next_state))

    def get_space_metrics(self) -> Dict:
        """获取空间指标"""
        try:
            return self.space.get_statistics()
        except:
            return {'name': self.name}


# ============================================================================
# 2. Meta-SSFR：空间选择器
# ============================================================================

@dataclass
class SpacePerformance:
    """空间性能记录"""
    space_name: str
    total_reward: float = 0.0
    visit_count: int = 0
    success_count: int = 0
    last_used: int = 0

    @property
    def average_reward(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count


class MetaSSFR:
    """
    Meta-SSFR：完整的可插拔架构

    核心功能：
    1. 管理多个 ATLAS 空间
    2. Meta-Learning 选择最优空间
    3. 在线学习空间性能
    4. 动态切换
    """

    def __init__(self, width: int = 10, height: int = 10):
        self.name = "MetaSSFR"
        self.width = width
        self.height = height

        # 获取所有可用空间
        self.available_spaces = list_available_spaces()
        print(f"Available spaces: {list(self.available_spaces.keys())}")

        # 创建空间实例
        self.spaces: Dict[str, SpaceAdapter] = {}
        self._create_spaces()

        # Meta-Learning 状态
        self.performance: Dict[str, SpacePerformance] = {}
        self._init_performance()

        # 当前选择
        self.current_space = 'euclidean'  # 默认从欧氏开始
        self.steps = 0
        self.last_switch = 0
        self.switch_cooldown = 3

        # 探索参数
        self.epsilon = 0.3
        self.min_epsilon = 0.05

        # 统计
        self.switch_count = 0
        self.exploration_count = 0
        self.space_usage = defaultdict(int)

    def _create_spaces(self):
        """创建所有可用空间"""
        for space_name in self.available_spaces.keys():
            try:
                space = create_space(space_name, self.width, self.height)
                self.spaces[space_name] = SpaceAdapter(space, space_name)
            except Exception as e:
                print(f"Warning: Could not create space {space_name}: {e}")

        # 如果没有创建成功，创建基本空间
        if not self.spaces:
            print("Warning: No spaces created, using fallback")
            # 创建简单的欧氏空间
            from atlas.spaces.euclidean import EuclideanSpace
            space = EuclideanSpace(self.width, self.height)
            self.spaces['euclidean'] = SpaceAdapter(space, 'euclidean')

    def _init_performance(self):
        """初始化性能记录"""
        for name in self.spaces.keys():
            self.performance[name] = SpacePerformance(name)

    def _get_state_signature(self, state: Dict) -> str:
        """提取状态签名"""
        pos = state.get('position', (0, 0))
        goal = state.get('goal', (0, 0))
        obstacles = state.get('obstacles', set())

        # 距离桶
        dist = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        dist_bucket = min(dist // 2, 5)

        # 障碍密度
        nearby = sum(1 for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]
                     if (pos[0]+dx, pos[1]+dy) in obstacles)

        return f"d{dist_bucket}_o{nearby}"

    def _select_space(self, state: Dict) -> str:
        """
        Meta-Learning 选择最优空间

        策略：
        1. ε-贪婪探索
        2. UCB 选择（利用+探索）
        3. 基于历史性能
        """
        signature = self._get_state_signature(state)

        # ε-贪婪
        if np.random.random() < self.epsilon:
            self.exploration_count += 1
            return np.random.choice(list(self.spaces.keys()))

        # UCB 选择
        best_space = self.current_space
        best_score = float('-inf')

        for space_name in self.spaces.keys():
            perf = self.performance[space_name]

            # 平均奖励
            avg_reward = perf.average_reward

            # UCB 探索奖励
            total_visits = sum(p.visit_count for p in self.performance.values())
            if perf.visit_count > 0:
                ucb = np.sqrt(2 * np.log(total_visits + 1) / perf.visit_count)
            else:
                ucb = float('inf')  # 未探索的有无限奖励

            score = avg_reward + ucb

            if score > best_score:
                best_score = score
                best_space = space_name

        return best_space

    def solve(self, state: Dict) -> str:
        """求解：选择空间并执行"""
        self.steps += 1

        # 定期选择空间
        if self.steps - self.last_switch >= self.switch_cooldown:
            new_space = self._select_space(state)
            if new_space != self.current_space:
                self.current_space = new_space
                self.last_switch = self.steps
                self.switch_count += 1

        # 执行
        self.space_usage[self.current_space] += 1
        adapter = self.spaces[self.current_space]
        return adapter.solve(state)

    def update(self, state: Dict, action: str, reward: float, next_state: Dict):
        """更新性能"""
        # 更新当前空间的性能
        perf = self.performance[self.current_space]
        perf.total_reward += reward
        perf.visit_count += 1
        perf.last_used = self.steps

        # 衰减探索率
        self.epsilon = max(self.min_epsilon, self.epsilon * 0.9999)

        # 更新底层空间
        for adapter in self.spaces.values():
            adapter.update(state, action, reward, next_state)

    def get_stats(self) -> Dict:
        """获取统计"""
        return {
            'switches': self.switch_count,
            'explorations': self.exploration_count,
            'epsilon': self.epsilon,
            'usage': dict(self.space_usage),
            'performance': {
                name: {
                    'avg_reward': perf.average_reward,
                    'visits': perf.visit_count,
                    'success_rate': perf.success_count / max(perf.visit_count, 1)
                }
                for name, perf in self.performance.items()
            }
        }


# ============================================================================
# 3. 测试环境
# ============================================================================

class GridWorldEnv:
    """网格世界环境"""

    def __init__(self, size: int = 10):
        self.size = size
        self.reset()

    def reset(self):
        self.position = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        self.steps = 0

        # 障碍
        self.obstacles = set()
        self.obstacles.add((3, 3))
        self.obstacles.add((5, 5))
        self.obstacles.add((7, 7))

        return self.get_state()

    def get_state(self) -> Dict:
        return {
            'position': self.position,
            'goal': self.goal,
            'obstacles': self.obstacles,
            'steps': self.steps
        }

    def step(self, action: str) -> Tuple[Dict, float, bool]:
        x, y = self.position

        if action == 'up':
            new_pos = (x, y + 1)
        elif action == 'down':
            new_pos = (x, y - 1)
        elif action == 'left':
            new_pos = (x - 1, y)
        elif action == 'right':
            new_pos = (x + 1, y)
        else:
            new_pos = self.position

        # 边界
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            new_pos = self.position

        # 障碍
        if new_pos in self.obstacles:
            reward = -5.0
        else:
            self.position = new_pos
            dist = abs(self.position[0] - self.goal[0]) + abs(self.position[1] - self.goal[1])
            reward = -0.1 * dist

        done = self.position == self.goal
        if done:
            reward = 100.0

        self.steps += 1
        return self.get_state(), reward, done


# ============================================================================
# 4. 测试
# ============================================================================

def test_meta_ssfr():
    """测试 Meta-SSFR"""
    print("=" * 70)
    print("Meta-SSFR with ATLAS Spaces Test")
    print("=" * 70)

    # 创建 Meta-SSFR
    meta = MetaSSFR(width=10, height=10)

    # 创建环境
    env = GridWorldEnv(size=10)

    # 运行
    total_reward = 0
    successes = 0
    total_steps = 0

    for episode in range(20):
        state = env.reset()
        episode_reward = 0

        for step in range(100):
            action = meta.solve(state)
            next_state, reward, done = env.step(action)
            meta.update(state, action, reward, next_state)

            episode_reward += reward
            state = next_state

            if done:
                successes += 1
                break

        total_reward += episode_reward
        total_steps += step + 1

    avg_reward = total_reward / 20
    success_rate = successes / 20
    avg_steps = total_steps / 20

    print(f"\nResults:")
    print(f"  Avg Reward: {avg_reward:.2f}")
    print(f"  Success Rate: {success_rate:.1%}")
    print(f"  Avg Steps: {avg_steps:.1f}")

    # 统计
    stats = meta.get_stats()
    print(f"\nMeta-Stats:")
    print(f"  Switches: {stats['switches']}")
    print(f"  Explorations: {stats['explorations']}")
    print(f"  Epsilon: {stats['epsilon']:.3f}")
    print(f"  Space Usage: {stats['usage']}")

    print(f"\nPerformance by Space:")
    for name, perf in stats['performance'].items():
        print(f"  {name}: avg_reward={perf['avg_reward']:.2f}, visits={perf['visits']}")

    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)


# ============================================================================
# 5. 对比实验
# ============================================================================

def compare_spaces():
    """对比所有 ATLAS 空间"""
    print("=" * 70)
    print("ATLAS Spaces Comparison")
    print("=" * 70)

    # 获取所有空间
    spaces = list_available_spaces()
    print(f"Available spaces: {list(spaces.keys())}")

    # 创建实验
    experiment = Experiment("space_comparison")

    # 注册空间
    for space_name in spaces.keys():
        try:
            space = create_space(space_name, 10, 10)
            experiment.register_space(space_name, space)
        except Exception as e:
            print(f"Warning: Could not register {space_name}: {e}")

    print(f"\nRegistered spaces: {list(experiment.spaces.keys())}")

    # 打印空间信息
    print("\nSpace Details:")
    for name, space in experiment.spaces.items():
        stats = space.get_statistics()
        print(f"  {name}: {stats}")

    print("\n" + "=" * 70)
    print("Comparison Complete")
    print("=" * 70)


# ============================================================================
# 6. 主入口
# ============================================================================

if __name__ == "__main__":
    # 测试 Meta-SSFR
    test_meta_ssfr()

    # 对比空间
    print("\n")
    compare_spaces()
