"""
Spatial Dynamics Core - 空间动力学核心

重构后的认知核心，空间动力学成为决策基础
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Deque, Any
from dataclasses import dataclass
from collections import deque, defaultdict
import random

import sys
sys.path.append('..')
from interfaces import (
    ICognitiveLayer, IFastLayer, ISlowLayer,
    Percept, Action, IEnvironment, ISpatialEnvironment
)


# ============================================================================
# 空间动力学数据类型
# ============================================================================

@dataclass
class Consequence:
    """
    后果：动作的预期感知变化

    核心概念：我们不预测世界状态，而是预测感知将如何变化
    """
    delta_perception: np.ndarray  # 感知变化向量（不是状态！）
    uncertainty: float            # 预测不确定性（0-1）
    valence: float                # 情感效价（-1到+1）

    def __post_init__(self):
        # 确保delta_perception是1D
        if len(self.delta_perception.shape) > 1:
            self.delta_perception = self.delta_perception.flatten()


@dataclass
class FieldEntry:
    """预测场条目"""
    consequence: Consequence
    visit_count: int = 0
    last_updated: int = 0


# ============================================================================
# 预测场 - 空间动力学核心
# ============================================================================

class PredictiveField:
    """
    预测场 - 空间-动作-后果的映射

    结构：(x, y, action) → Consequence

    这是核心数据结构，所有认知都建立在此之上。
    """

    def __init__(self, width: int, height: int, perception_dim: int):
        self.width = width
        self.height = height
        self.perception_dim = perception_dim

        self.num_actions = len(Action)

        # 核心场：存储预期的感知变化
        # shape: [x, y, action, perception_dim + 2]
        self.field = np.zeros((width, height, self.num_actions, perception_dim + 2))

        # 不确定性场：单独存储以便快速访问
        self.uncertainty = np.ones((width, height, self.num_actions))

        # 访问计数：用于学习率调整
        self.visits = np.zeros((width, height, self.num_actions), dtype=np.int32)

        # 效价场：空间中的价值分布
        self.valence = np.zeros((width, height))

        self.time_step = 0

    def get_consequence(self, x: int, y: int, action: Action) -> Consequence:
        """获取指定位置-动作的后果预测"""
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))

        pred = self.field[x, y, action.value]

        return Consequence(
            delta_perception=pred[:self.perception_dim],
            uncertainty=self.uncertainty[x, y, action.value],
            valence=pred[self.perception_dim + 1]
        )

    def update(self, x: int, y: int, action: Action,
               consequence: Consequence, learning_rate: float = 0.1):
        """更新预测场"""
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))

        # 访问计数
        self.visits[x, y, action.value] += 1
        visit_n = self.visits[x, y, action.value]

        # 自适应学习率：访问越多学习率越低
        effective_lr = learning_rate / (1 + 0.1 * visit_n)

        # 旧的预测
        old_pred = self.field[x, y, action.value]
        old_uncertainty = self.uncertainty[x, y, action.value]

        # 新的目标
        target = np.concatenate([
            consequence.delta_perception,
            [consequence.uncertainty],
            [consequence.valence]
        ])

        # 指数移动平均
        self.field[x, y, action.value] = (
            (1 - effective_lr) * old_pred + effective_lr * target
        )

        # 不确定性单独更新（传播）
        self.uncertainty[x, y, action.value] = (
            (1 - effective_lr) * old_uncertainty +
            effective_lr * consequence.uncertainty
        )

        self.time_step += 1

    def propagate_valence(self, goal_position: Tuple[float, float],
                          goal_valence: float = 1.0, radius: float = 20.0):
        """
        在预测场中传播效价（目标吸引子）

        这会创建一个从当前位置到目标的效价梯度
        """
        gx, gy = int(goal_position[0]), int(goal_position[1])

        for dx in range(-int(radius), int(radius) + 1):
            for dy in range(-int(radius), int(radius) + 1):
                x, y = gx + dx, gy + dy
                if 0 <= x < self.width and 0 <= y < self.height:
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < radius:
                        # 距离目标越近，效价越高
                        self.valence[x, y] = goal_valence * (1 - dist / radius)

    def get_gradient_at(self, x: int, y: int) -> Tuple[float, float]:
        """
        获取当前位置的效价梯度

        Returns:
            (dx, dy): 指向高效价区域的方向
        """
        x = max(1, min(x, self.width - 2))
        y = max(1, min(y, self.height - 2))

        # 计算梯度
        dx = (self.valence[x + 1, y] - self.valence[x - 1, y]) / 2
        dy = (self.valence[x, y + 1] - self.valence[x, y - 1]) / 2

        return dx, dy

    def get_attention_salience(self) -> np.ndarray:
        """
        获取注意力显著性图

        显著性 = 不确定性 × |效价| （高不确定性且高价值）
        """
        avg_uncertainty = np.mean(self.uncertainty, axis=2)
        return avg_uncertainty * np.abs(self.valence)

    def query_predictive_gradient(self, x: int, y: int) -> List[Tuple[Action, float, float]]:
        """
        查询预测梯度

        对于每个可能的动作，返回：
        (动作, 预期效价变化, 不确定性)

        这是空间动力学导航的核心！
        """
        results = []

        for action in list(Action)[1:]:  # 排除STAY
            cons = self.get_consequence(x, y, action)

            # 预测新的位置
            new_x, new_y = x, y
            if action == Action.UP:
                new_y -= 1
            elif action == Action.DOWN:
                new_y += 1
            elif action == Action.LEFT:
                new_x -= 1
            elif action == Action.RIGHT:
                new_x += 1

            new_x = max(0, min(new_x, self.width - 1))
            new_y = max(0, min(new_y, self.height - 1))

            # 预测效价变化
            predicted_valence = (
                self.valence[new_x, new_y] +
                cons.valence * (1 - cons.uncertainty)
            )

            results.append((action, predicted_valence, cons.uncertainty))

        return results


# ============================================================================
# 快层 - 透明执行
# ============================================================================

class FastLayer(IFastLayer):
    """
    快层 - 基于预测场的自动化

    当预测可靠（低不确定性）时，直接沿效价梯度滑动
    """

    def __init__(self, uncertainty_threshold: float = 0.3):
        self.uncertainty_threshold = uncertainty_threshold

    def process(self, percept: Percept,
                context: Dict[str, Any]) -> Tuple[Action, Dict]:
        """
        快层处理

        如果预测场在当前位置显示低不确定性，
        直接沿效价梯度滑动（透明执行）
        """
        field = context.get('predictive_field')
        if field is None:
            return Action.STAY, {'layer': 'fast', 'transparent': False}

        x, y = int(percept.position[0]), int(percept.position[1])

        # 查询所有动作的预测
        predictions = field.query_predictive_gradient(x, y)

        # 检查是否所有预测都可靠
        avg_uncertainty = np.mean([p[2] for p in predictions])

        if avg_uncertainty < self.uncertainty_threshold:
            # 透明执行：沿最高效价方向
            best_action = max(predictions, key=lambda p: p[1])
            return best_action[0], {
                'layer': 'fast',
                'transparent': True,
                'predicted_valence': best_action[1],
                'uncertainty': avg_uncertainty
            }

        # 预测不可靠，需要慢层
        return Action.STAY, {
            'layer': 'fast',
            'transparent': False,
            'avg_uncertainty': avg_uncertainty
        }

    def learn(self, experience: Dict[str, Any]) -> None:
        """快层不需要显式学习，预测场更新即学习"""
        pass

    def get_state(self) -> Dict[str, Any]:
        return {'uncertainty_threshold': self.uncertainty_threshold}


# ============================================================================
# 慢层 - 意识升起
# ============================================================================

class SlowLayer(ISlowLayer):
    """
    慢层 - 基于预测场的MCTS搜索

    当不确定性高时，在预测场上模拟搜索
    """

    def __init__(self, num_simulations: int = 30, depth: int = 5):
        self.num_simulations = num_simulations
        self.depth = depth

    def process(self, percept: Percept,
                context: Dict[str, Any]) -> Tuple[Action, Dict]:
        """
        慢层处理 - 在预测场上进行MCTS搜索

        如果预测场没有数据或不确定性太高，
        使用基于目标的启发式导航
        """
        field = context.get('predictive_field')
        goal_pos = context.get('goal_position')
        current_pos = percept.position

        if field is None:
            return self._heuristic_action(current_pos, goal_pos), {'layer': 'slow', 'source': 'heuristic_no_field'}

        x, y = int(current_pos[0]), int(current_pos[1])

        # 检查预测场是否有数据
        has_data = np.any(field.visits[x, y, :] > 0)

        if not has_data and goal_pos:
            # 没有数据但有目标，使用启发式导航
            return self._heuristic_action(current_pos, goal_pos), {'layer': 'slow', 'source': 'heuristic_no_data'}

        # 如果提供了目标，在预测场中设置吸引子
        if goal_pos:
            field.propagate_valence(goal_pos, goal_valence=1.0)

        # MCTS搜索
        action_scores = defaultdict(float)

        for _ in range(self.num_simulations):
            action, value = self._simulate(field, x, y, self.depth, goal_pos)
            if action:
                action_scores[action] += value

        if not action_scores or max(action_scores.values()) == 0:
            # MCTS没有产生有价值的结果，使用启发式
            return self._heuristic_action(current_pos, goal_pos), {'layer': 'slow', 'source': 'heuristic_mcts_failed'}

        best_action = max(action_scores.items(), key=lambda x: x[1])[0]

        return best_action, {
            'layer': 'slow',
            'source': 'mcts',
            'simulations': self.num_simulations,
            'expected_value': action_scores[best_action] / self.num_simulations
        }

    def _heuristic_action(self, current_pos: Tuple[float, float],
                         goal_pos: Optional[Tuple[float, float]]) -> Action:
        """启发式导航：直接向目标移动"""
        if goal_pos is None:
            return random.choice([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT])

        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]

        if abs(dx) > abs(dy):
            return Action.RIGHT if dx > 0 else Action.LEFT
        else:
            return Action.DOWN if dy > 0 else Action.UP

    def _simulate(self, field: PredictiveField, start_x: int, start_y: int,
                 depth: int, goal_pos: Optional[Tuple[float, float]] = None) -> Tuple[Optional[Action], float]:
        """单次模拟：在预测场中随机游走，支持目标导向"""
        x, y = start_x, start_y
        total_value = 0.0
        first_action = None

        for step in range(depth):
            # 获取当前预测梯度
            predictions = field.query_predictive_gradient(x, y)

            if not predictions:
                break

            # 按效价加权选择（探索与利用平衡）
            actions, valences, uncertainties = zip(*predictions)

            # UCB-like选择
            weights = np.array(valences) * (1 - np.array(uncertainties))

            # 如果效价全为零但有目标，使用目标方向的启发式偏向
            if np.sum(weights) <= 0.01 and goal_pos:
                # 计算目标方向
                dx = goal_pos[0] - x
                dy = goal_pos[1] - y

                # 偏向目标方向的动作
                for i, act in enumerate(actions):
                    if abs(dx) > abs(dy):
                        if (dx > 0 and act == Action.RIGHT) or (dx < 0 and act == Action.LEFT):
                            weights[i] = 2.0
                    else:
                        if (dy > 0 and act == Action.DOWN) or (dy < 0 and act == Action.UP):
                            weights[i] = 2.0

            # 确保所有权重非负
            weights = np.maximum(weights, 0)

            if np.sum(weights) <= 0:
                weights = np.ones(len(weights))

            weights = weights / np.sum(weights)
            chosen_idx = np.random.choice(len(actions), p=weights)
            action = actions[chosen_idx]

            if first_action is None:
                first_action = action

            # 累积效价（如果效价为零，给予小的探索奖励）
            val = valences[chosen_idx] if valences[chosen_idx] != 0 else 0.1
            total_value += val * (0.9 ** step)

            # 预测下一位置
            if action == Action.UP:
                y -= 1
            elif action == Action.DOWN:
                y += 1
            elif action == Action.LEFT:
                x -= 1
            elif action == Action.RIGHT:
                x += 1

            x = max(0, min(x, field.width - 1))
            y = max(0, min(y, field.height - 1))

        return first_action, total_value

    def _random_action(self) -> Action:
        return random.choice([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT])

    def learn(self, experience: Dict[str, Any]) -> None:
        """慢层不需要显式学习"""
        pass

    def get_state(self) -> Dict[str, Any]:
        return {
            'num_simulations': self.num_simulations,
            'depth': self.depth
        }


# ============================================================================
# 认知协调器
# ============================================================================

class CognitiveOrchestrator(ICognitiveLayer):
    """
    认知协调器

    整合快层（透明）和慢层（意识）
    """

    def __init__(self, perception_dim: int, field_width: int, field_height: int,
                 fast_layer: Optional[FastLayer] = None,
                 slow_layer: Optional[SlowLayer] = None):
        # 共享的预测场
        self.predictive_field = PredictiveField(
            field_width, field_height, perception_dim
        )

        # 两层
        self.fast = fast_layer or FastLayer()
        self.slow = slow_layer or SlowLayer()

        # 统计
        self.fast_uses = 0
        self.slow_uses = 0

    def process(self, percept: Percept,
                context: Dict[str, Any]) -> Tuple[Action, Dict]:
        """
        处理流程：
        1. 快层尝试（预测可靠时透明执行）
        2. 慢层接管（高不确定性时搜索）
        """
        # 提供预测场给快层
        fast_context = {**context, 'predictive_field': self.predictive_field}

        action, fast_meta = self.fast.process(percept, fast_context)

        # 如果快层能处理
        if fast_meta.get('transparent'):
            self.fast_uses += 1
            return action, {
                **fast_meta,
                'layer': 'fast',
                'awareness': 0.1  # 低意识
            }

        # 唤起慢层（意识升起）
        slow_context = {**context, 'predictive_field': self.predictive_field}
        action, slow_meta = self.slow.process(percept, slow_context)

        self.slow_uses += 1

        return action, {
            **slow_meta,
            'layer': 'slow',
            'awareness': 0.8,  # 高意识
            'trigger': 'high_uncertainty'
        }

    def learn(self, experience: Dict[str, Any]) -> None:
        """
        从经验学习 - 更新预测场

        这是核心学习机制：将实际感知变化写入预测场
        """
        position = experience.get('position', (0, 0))
        action = experience.get('action')

        if not action or action == Action.STAY:
            return

        x, y = int(position[0]), int(position[1])

        # 构建后果
        delta = experience.get('delta_perception', np.zeros(self.predictive_field.perception_dim))
        reward = experience.get('reward', 0.0)
        prediction_error = experience.get('prediction_error', 1.0)

        consequence = Consequence(
            delta_perception=delta,
            uncertainty=prediction_error,
            valence=np.tanh(reward)  # 情感效价
        )

        # 更新预测场
        self.predictive_field.update(x, y, action, consequence)

    def get_state(self) -> Dict[str, Any]:
        return {
            'fast_uses': self.fast_uses,
            'slow_uses': self.slow_uses,
            'fast_ratio': self.fast_uses / max(1, self.fast_uses + self.slow_uses),
            'field_coverage': np.count_nonzero(self.predictive_field.visits) / (
                self.predictive_field.width * self.predictive_field.height
            ),
            'avg_uncertainty': np.mean(self.predictive_field.uncertainty)
        }
