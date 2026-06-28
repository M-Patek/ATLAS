"""
Cognitive Core - 认知核心

实现快层和慢层，只依赖接口
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Deque, Callable, Any
from collections import deque, defaultdict
from dataclasses import dataclass
import random

import sys
sys.path.append('..')
from interfaces import (
    ICognitiveLayer, IFastLayer, ISlowLayer,
    Percept, Action, IEnvironment
)


# ============================================================================
# 快层实现
# ============================================================================

@dataclass
class FastPathEntry:
    """快层缓存条目"""
    action: Action
    success_count: int = 0
    last_used: int = 0
    avg_prediction_error: float = 0.0


class FastLayer(IFastLayer):
    """
    快层实现 - 程序性记忆

    自动化习惯缓存：(状态签名, 意图) → 动作
    """

    def __init__(self, automation_threshold: int = 3, max_cache: int = 1000):
        self._automation_threshold = automation_threshold
        self._max_cache = max_cache

        # (state_key, intention_id) -> entry
        self._cache: Dict[Tuple[str, int], FastPathEntry] = {}
        self._time = 0

    def _make_state_key(self, percept: Percept) -> str:
        """生成状态签名"""
        # 简化：位置哈希 + 部分特征
        pos_hash = f"{int(percept.position[0])//2},{int(percept.position[1])//2}"
        feature_hash = hash(percept.features[:4].tobytes()) % 1000
        return f"{pos_hash}_{feature_hash}"

    def process(self, percept: Percept,
                context: Dict[str, Any]) -> Tuple[Action, Dict]:
        """快层处理：查找缓存"""
        intention_id = context.get('intention_id', 0)
        state_key = self._make_state_key(percept)
        cache_key = (state_key, intention_id)

        entry = self._cache.get(cache_key)

        if entry and entry.success_count >= self._automation_threshold:
            entry.last_used = self._time
            self._time += 1
            return entry.action, {
                'layer': 'fast',
                'automation_level': entry.success_count / 10.0,
                'source': 'habit_cache'
            }

        return Action.STAY, {  # 返回STAY表示需要慢层处理
            'layer': 'fast',
            'cache_miss': True,
            'source': 'no_habit'
        }

    def learn(self, experience: Dict[str, Any]) -> None:
        """从经验中学习：固化或失效"""
        state_key = experience.get('state_key')
        intention_id = experience.get('intention_id', 0)
        action = experience.get('action')
        success = experience.get('success', False)
        prediction_error = experience.get('prediction_error', 1.0)

        if not state_key or not action:
            return

        cache_key = (state_key, intention_id)

        if success and prediction_error < 0.3:
            # 固化习惯
            if cache_key not in self._cache:
                self._cache[cache_key] = FastPathEntry(action)

            entry = self._cache[cache_key]
            entry.success_count = min(10, entry.success_count + 1)
            entry.avg_prediction_error = (
                0.9 * entry.avg_prediction_error + 0.1 * prediction_error
            )
        else:
            # 失效习惯
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                entry.success_count = max(0, entry.success_count - 2)
                if entry.success_count == 0:
                    del self._cache[cache_key]

        # 清理旧缓存
        if len(self._cache) > self._max_cache:
            oldest = min(self._cache.items(), key=lambda x: x[1].last_used)
            del self._cache[oldest[0]]

    def get_state(self) -> Dict[str, Any]:
        return {
            'cached_paths': len(self._cache),
            'automation_rate': sum(
                1 for e in self._cache.values()
                if e.success_count >= self._automation_threshold
            ) / max(1, len(self._cache))
        }

    def invalidate(self, percept: Percept, intention_id: int) -> None:
        """显式失效某条路径"""
        state_key = self._make_state_key(percept)
        cache_key = (state_key, intention_id)
        if cache_key in self._cache:
            del self._cache[cache_key]


# ============================================================================
# 慢层实现
# ============================================================================

@dataclass
class Node:
    """MCTS节点"""
    state_key: str
    parent: Optional['Node']
    action: Optional[Action]

    children: Dict[Action, 'Node'] = None
    visits: int = 0
    value: float = 0.0
    prior: float = 0.0

    def __post_init__(self):
        if self.children is None:
            self.children = {}

    def uct_score(self, parent_visits: int, c_puct: float = 1.0) -> float:
        """UCT分数"""
        if self.visits == 0:
            return float('inf')
        q_value = self.value / self.visits
        u_value = c_puct * self.prior * np.sqrt(parent_visits) / (1 + self.visits)
        return q_value + u_value


class SlowLayer(ISlowLayer):
    """
    慢层实现 - MCTS搜索

    基于预测场的蒙特卡洛树搜索
    """

    def __init__(self, env: IEnvironment,
                 num_simulations: int = 20,
                 planning_depth: int = 10):
        self._env = env
        self._num_simulations = num_simulations
        self._planning_depth = planning_depth

        # 预测场: (x, y, action) -> (expected_reward, uncertainty)
        self._predictive_field: Dict[Tuple[int, int, Action], Tuple[float, float]] = {}
        self._visit_count = np.zeros((100, 100), dtype=np.int32)

        self._root: Optional[Node] = None

    def process(self, percept: Percept,
                context: Dict[str, Any]) -> Tuple[Action, Dict]:
        """慢层处理：MCTS搜索"""
        state_key = self._make_state_key(percept)

        # 创建根节点
        self._root = Node(state_key, None, None)

        # MCTS模拟
        for _ in range(self._num_simulations):
            self._simulate(percept, planning=True)

        # 选择最佳动作
        if not self._root.children:
            return self._default_action(percept), {
                'layer': 'slow',
                'source': 'default',
                'simulations': 0
            }

        best_action = max(
            self._root.children.items(),
            key=lambda x: x[1].visits
        )[0]

        return best_action, {
            'layer': 'slow',
            'source': 'mcts',
            'simulations': self._num_simulations,
            'root_visits': self._root.visits
        }

    def _simulate(self, percept: Percept, planning: bool = True) -> float:
        """单次模拟"""
        # 简化版：直接使用预测场评估
        current_pos = percept.position
        x, y = int(current_pos[0]), int(current_pos[1])

        total_reward = 0.0
        valid_actions = []

        for action in list(Action)[1:]:  # 排除STAY
            # 预测结果
            pred = self._predictive_field.get((x, y, action), (0.0, 1.0))
            expected_reward, uncertainty = pred

            if uncertainty < 0.5:  # 可信预测
                total_reward += expected_reward
                valid_actions.append((action, expected_reward))

        # 选择奖励最高的动作
        if valid_actions:
            valid_actions.sort(key=lambda x: x[1], reverse=True)
            return valid_actions[0][1]

        return 0.0

    def _default_action(self, percept: Percept) -> Action:
        """默认动作：随机探索"""
        return random.choice([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT])

    def _make_state_key(self, percept: Percept) -> str:
        """生成状态键"""
        pos = percept.position
        return f"{int(pos[0])},{int(pos[1])}"

    def learn(self, experience: Dict[str, Any]) -> None:
        """更新预测场"""
        position = experience.get('position', (0, 0))
        action = experience.get('action')
        reward = experience.get('reward', 0.0)

        if not action:
            return

        x, y = int(position[0]), int(position[1])
        old = self._predictive_field.get((x, y, action), (0.0, 1.0))

        # 指数移动平均
        new_reward = 0.8 * old[0] + 0.2 * reward
        new_uncertainty = max(0.0, min(1.0, old[1] * 0.95))

        self._predictive_field[(x, y, action)] = (new_reward, new_uncertainty)
        self._visit_count[y, x] += 1

    def get_state(self) -> Dict[str, Any]:
        return {
            'field_size': len(self._predictive_field),
            'avg_visits': np.mean(self._visit_count),
            'exploration_coverage': np.count_nonzero(self._visit_count) / 10000
        }


# ============================================================================
# 认知协调器
# ============================================================================

class CognitiveOrchestrator(ICognitiveLayer):
    """
    认知协调器

    整合快层和慢层，根据情况切换
    """

    def __init__(self, fast_layer: IFastLayer, slow_layer: ISlowLayer,
                 awareness_threshold: float = 0.3):
        self._fast = fast_layer
        self._slow = slow_layer
        self._awareness_threshold = awareness_threshold

        self._last_slow_result: Optional[Tuple[Action, Dict]] = None
        self._average_prediction_error = 1.0

    def process(self, percept: Percept,
                context: Dict[str, Any]) -> Tuple[Action, Dict]:
        """
        协调处理流程：
        1. 先问快层
        2. 如果不自动化，唤起慢层
        """
        # 首先尝试快层
        fast_action, fast_meta = self._fast.process(percept, context)

        # 如果快层能处理（不是STAY）
        if fast_action != Action.STAY:
            return fast_action, {
                **fast_meta,
                'awareness_level': 0.1,  # 低意识水平
                'layer_switched': False
            }

        # 唤起慢层
        slow_action, slow_meta = self._slow.process(percept, context)
        self._last_slow_result = (slow_action, slow_meta)

        return slow_action, {
            **slow_meta,
            'awareness_level': 0.7,  # 高意识水平
            'layer_switched': True,
            'fast_cache_miss': fast_meta.get('cache_miss', False)
        }

    def learn(self, experience: Dict[str, Any]) -> None:
        """两层都学习"""
        # 快层学习
        self._fast.learn(experience)

        # 慢层学习
        self._slow.learn(experience)

        # 更新平均预测误差
        pe = experience.get('prediction_error', 1.0)
        self._average_prediction_error = 0.9 * self._average_prediction_error + 0.1 * pe

    def get_state(self) -> Dict[str, Any]:
        return {
            'fast': self._fast.get_state(),
            'slow': self._slow.get_state(),
            'avg_prediction_error': self._average_prediction_error,
            'current_awareness': self._estimate_awareness()
        }

    def _estimate_awareness(self) -> float:
        """估计当前意识水平"""
        # 基于预测误差和意图冲突
        return min(1.0, self._average_prediction_error * 2)

    def force_slow(self, percept: Percept, context: Dict) -> Tuple[Action, Dict]:
        """强制使用慢层"""
        return self._slow.process(percept, context)
