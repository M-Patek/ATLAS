"""
Multi-Scale Predictive Attention System
多时间尺度预测性注意力系统

基于 Evolver 架构启发，实现：
1. 快层(Fast) - 透明行动，习惯缓存
2. 慢层(Slow) - MCTS式搜索，意识升起
3. 多时间尺度意图 - 快/中/慢三层
4. 自我感知 - 内感受，预测误差监控
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Deque
from enum import Enum
from collections import deque, defaultdict
import random
import math


class Action(Enum):
    """离散动作空间"""
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    INTERACT = 5


@dataclass
class Consequence:
    """后果：预期的感知变化"""
    delta_perception: np.ndarray
    expected_uncertainty: float
    affective_valence: float

    def __post_init__(self):
        if len(self.delta_perception.shape) > 1:
            self.delta_perception = self.delta_perception.flatten()


@dataclass
class Percept:
    """感知：主观感知（非客观世界状态）"""
    local_features: np.ndarray
    proprioception: np.ndarray
    prediction_error: float


# ============================================================================
# 第一层：快层（Fast Layer）- 透明行动
# ============================================================================

@dataclass
class FastPathEntry:
    """快层缓存条目"""
    action: Action
    success_count: int = 0
    last_used: int = 0
    avg_prediction_error: float = 0.0


class FastLayer:
    """
    快层：习惯缓存，实现"透明性"

    类比：熟练骑自行车时，不需要思考每个动作
    当(状态,意图)对被多次成功执行后，进入快层
    """
    def __init__(self, automation_threshold: int = 3, max_cache: int = 1000):
        self.automation_threshold = automation_threshold
        self.max_cache = max_cache
        # (position_hash, intention_id) -> entry
        self.habit_cache: Dict[Tuple[int, int], FastPathEntry] = {}
        self.time = 0

    def _hash_position(self, pos: Tuple[float, float]) -> int:
        """位置哈希（离散化到网格）"""
        return hash((int(pos[0]), int(pos[1])))

    def try_respond(self, position: Tuple[float, float], intention_id: int) -> Optional[Action]:
        """
        尝试快层响应

        Returns:
            Action: 如果缓存命中且自动化
            None: 需要慢层处理
        """
        key = (self._hash_position(position), intention_id)

        if key in self.habit_cache:
            entry = self.habit_cache[key]
            if entry.success_count >= self.automation_threshold:
                entry.last_used = self.time
                self.time += 1
                return entry.action

        return None

    def consolidate(self, position: Tuple[float, float], intention_id: int,
                    action: Action, prediction_error: float):
        """
        巩固学习：成功执行后缓存到快层

        类比：练习使动作变得"自然"
        """
        key = (self._hash_position(position), intention_id)

        if key not in self.habit_cache:
            self.habit_cache[key] = FastPathEntry(action=action)

        entry = self.habit_cache[key]

        # 如果动作改变，重置计数
        if entry.action != action:
            entry.action = action
            entry.success_count = 1
        else:
            entry.success_count += 1

        # 更新平均预测误差
        entry.avg_prediction_error = 0.9 * entry.avg_prediction_error + 0.1 * prediction_error
        entry.last_used = self.time
        self.time += 1

        # 缓存管理
        if len(self.habit_cache) > self.max_cache:
            self._evict_oldest()

    def invalidate(self, position: Tuple[float, float], intention_id: int):
        """
        失效：当快层预测失败时

        类比：习惯动作失效，需要意识重新参与
        """
        key = (self._hash_position(position), intention_id)

        if key in self.habit_cache:
            entry = self.habit_cache[key]
            entry.success_count = max(0, entry.success_count - 2)  # 快速降低信任

            if entry.success_count == 0:
                del self.habit_cache[key]

    def _evict_oldest(self):
        """LRU淘汰"""
        if self.habit_cache:
            oldest_key = min(self.habit_cache.keys(),
                           key=lambda k: self.habit_cache[k].last_used)
            del self.habit_cache[oldest_key]

    def get_stats(self) -> Dict:
        """统计信息"""
        automated = sum(1 for e in self.habit_cache.values()
                       if e.success_count >= self.automation_threshold)
        return {
            'total_cached': len(self.habit_cache),
            'automated': automated,
            'automation_rate': automated / len(self.habit_cache) if self.habit_cache else 0
        }


# ============================================================================
# 第二层：多时间尺度意图（Intention Field）
# ============================================================================

@dataclass
class MotorIntention:
    """运动意图（快层，10ms级）- 直接动作倾向"""
    preferred_direction: np.ndarray  # 2D方向向量
    intensity: float  # 0-1强度


@dataclass
class GoalIntention:
    """目标意图（中层，100ms级）- 目标位置/状态"""
    target_position: Optional[Tuple[float, float]]
    target_feature: Optional[np.ndarray]  # 目标感知特征
    urgency: float  # 紧迫性


@dataclass
class ValueIntention:
    """价值意图（慢层，1s+级）- 长期价值/策略"""
    exploration_vs_exploitation: float  # 0=探索, 1=利用
    risk_tolerance: float  # 风险容忍度
    coherence_threshold: float  # 一致性要求（影响意识敏感度）


class IntentionField:
    """
    意图场：三层意图的协调与冲突检测

    类比：人类同时有多个时间尺度的意图
    - 快：手向杯子移动
    - 中：想喝水
    - 慢：保持健康
    """
    def __init__(self):
        self.motor = MotorIntention(
            preferred_direction=np.array([0.0, 0.0]),
            intensity=0.0
        )
        self.goal = GoalIntention(
            target_position=None,
            target_feature=None,
            urgency=0.5
        )
        self.value = ValueIntention(
            exploration_vs_exploitation=0.5,
            risk_tolerance=0.5,
            coherence_threshold=0.7
        )

        # 历史记录（用于检测意图演化）
        self.motor_history: Deque[MotorIntention] = deque(maxlen=10)
        self.goal_history: Deque[GoalIntention] = deque(maxlen=5)

    def update_motor(self, direction: np.ndarray, intensity: float):
        """更新运动意图"""
        self.motor_history.append(self.motor)
        self.motor = MotorIntention(
            preferred_direction=direction / (np.linalg.norm(direction) + 1e-8),
            intensity=intensity
        )

    def update_goal(self, target_pos: Optional[Tuple[float, float]],
                    target_feat: Optional[np.ndarray], urgency: float):
        """更新目标意图"""
        self.goal_history.append(self.goal)
        self.goal = GoalIntention(
            target_position=target_pos,
            target_feature=target_feat,
            urgency=urgency
        )

    def compute_conflict(self) -> float:
        """
        计算意图间冲突 = 意识升起的信号

        Returns:
            float: 冲突水平 0-1，越高越需要意识协调
        """
        conflicts = []

        # 1. 运动意图 vs 目标意图
        if self.goal.target_position is not None and self.motor.intensity > 0.1:
            # 计算运动方向是否指向目标
            goal_direction = np.array(self.goal.target_position) - np.array([0, 0])  # 相对当前位置
            goal_direction = goal_direction / (np.linalg.norm(goal_direction) + 1e-8)

            alignment = np.dot(self.motor.preferred_direction, goal_direction)
            conflicts.append(1.0 - max(0, alignment))  # 不一致则冲突

        # 2. 目标紧迫性 vs 价值风险容忍
        if self.goal.urgency > self.value.risk_tolerance:
            conflicts.append(self.goal.urgency - self.value.risk_tolerance)

        # 3. 运动意图稳定性（频繁改变=高冲突）
        if len(self.motor_history) >= 3:
            recent_directions = [m.preferred_direction for m in list(self.motor_history)[-3:]]
            direction_variance = np.mean([np.linalg.norm(d - recent_directions[-1])
                                         for d in recent_directions[:-1]])
            conflicts.append(min(1.0, direction_variance))

        return np.mean(conflicts) if conflicts else 0.0

    def get_dominant_intention(self) -> Tuple[str, float]:
        """获取主导意图及其强度"""
        conflict = self.compute_conflict()

        if conflict > self.value.coherence_threshold:
            # 高冲突时，慢层（价值）主导
            return ('value', 1.0 - self.value.risk_tolerance)
        elif self.goal.urgency > 0.7:
            # 高紧迫性时，中层（目标）主导
            return ('goal', self.goal.urgency)
        else:
            # 平时，快层（运动）主导
            return ('motor', self.motor.intensity)


# ============================================================================
# 第三层：自我感知（Interoception）
# ============================================================================

class Interoception:
    """
    内感受：系统对自身状态的感知

    类比：人类感知自己的心跳、肌肉张力、预测感
    """
    def __init__(self, window_size: int = 50):
        self.prediction_errors: Deque[float] = deque(maxlen=window_size)
        self.uncertainty_history: Deque[float] = deque(maxlen=window_size)
        self.cycle_times: Deque[float] = deque(maxlen=20)  # 处理周期时间

        self.current_arousal: float = 0.5  # 唤醒水平
        self.current_confidence: float = 0.5

    def update_prediction_error(self, error: float):
        """更新预测误差"""
        self.prediction_errors.append(error)

    def update_uncertainty(self, uncertainty: float):
        """更新不确定性"""
        self.uncertainty_history.append(uncertainty)

    def update_cycle_time(self, cycle_time: float):
        """更新处理周期（资源使用代理）"""
        self.cycle_times.append(cycle_time)

    def compute_felt_state(self) -> Dict:
        """
        计算"感觉状态" - 影响意图生成

        Returns:
            confidence: 信心（高=预测准确）
            arousal: 唤醒（高=需要快速响应）
            valence: 效价（正=预测误差减少趋势）
        """
        # 信心：基于近期预测误差
        if self.prediction_errors:
            recent_error = np.mean(list(self.prediction_errors)[-10:])
            self.current_confidence = 1.0 / (1.0 + recent_error * 2)

        # 唤醒：基于不确定性和处理负荷
        if self.uncertainty_history and self.cycle_times:
            uncertainty_factor = np.mean(list(self.uncertainty_history)[-5:])
            load_factor = np.mean(self.cycle_times) / max(self.cycle_times) if self.cycle_times else 0.5
            self.current_arousal = 0.5 * uncertainty_factor + 0.5 * load_factor

        # 效价：预测误差趋势
        valence = 0.0
        if len(self.prediction_errors) >= 2:
            errors = list(self.prediction_errors)
            if errors[-1] < np.mean(errors[:-1]):
                valence = 0.3  # 改善
            elif errors[-1] > np.mean(errors[:-1]):
                valence = -0.3  # 恶化

        return {
            'confidence': self.current_confidence,
            'arousal': self.current_arousal,
            'valence': valence
        }

    def should_escalate_to_conscious(self) -> bool:
        """判断是否应该升级到意识处理（慢层）"""
        felt = self.compute_felt_state()

        # 低信心或高唤醒时，需要意识
        return felt['confidence'] < 0.3 or felt['arousal'] > 0.7


# ============================================================================
# 预测场和其他组件（保留原有功能）
# ============================================================================

class PredictiveField:
    """预测场 - 后果预测的空间存储"""
    def __init__(self, width: int, height: int, perception_dim: int = 8):
        self.width = width
        self.height = height
        self.perception_dim = perception_dim
        num_actions = len(Action)
        consequence_dim = perception_dim + 2

        self.field = np.zeros((height, width, num_actions, consequence_dim))
        self.uncertainty_field = np.ones((height, width, num_actions)) * 0.5
        self.visit_count = np.zeros((height, width))

    def get_prediction(self, x: int, y: int, action: Action) -> Consequence:
        x, y = int(x), int(y)
        if not (0 <= x < self.width and 0 <= y < self.height):
            return Consequence(
                delta_perception=np.zeros(self.perception_dim),
                expected_uncertainty=1.0,
                affective_valence=0.0
            )

        pred = self.field[y, x, action.value]
        return Consequence(
            delta_perception=pred[:self.perception_dim],
            expected_uncertainty=pred[self.perception_dim],
            affective_valence=pred[self.perception_dim + 1]
        )

    def update_prediction(self, x: int, y: int, action: Action,
                         consequence: Consequence, learning_rate: float = 0.1):
        x, y = int(x), int(y)
        if not (0 <= x < self.width and 0 <= y < self.height):
            return

        target = np.concatenate([
            consequence.delta_perception,
            [consequence.expected_uncertainty, consequence.affective_valence]
        ])

        old_pred = self.field[y, x, action.value]
        self.field[y, x, action.value] = (1 - learning_rate) * old_pred + learning_rate * target

        prediction_error = np.linalg.norm(old_pred - target)
        self.uncertainty_field[y, x, action.value] = (
            (1 - learning_rate) * self.uncertainty_field[y, x, action.value] +
            learning_rate * prediction_error
        )
        self.visit_count[y, x] += 1


# ============================================================================
# 主智能体：多尺度预测性智能体
# ============================================================================

class MultiScalePredictiveAgent:
    """
    多尺度预测性智能体

    整合：
    - 快层：习惯缓存，透明行动
    - 慢层：预测场搜索，意识升起
    - 意图场：三层意图协调
    - 自我感知：内感受监控
    """
    def __init__(self, env, perception_dim: int = 8, learning_rate: float = 0.1):
        self.env = env
        self.perception_dim = perception_dim
        self.lr = learning_rate

        # 多尺度组件
        self.fast_layer = FastLayer(automation_threshold=3)
        self.intention_field = IntentionField()
        self.interoception = Interoception()

        # 慢层组件（原有）
        self.predictive_field = PredictiveField(env.width, env.height, perception_dim)

        # 状态
        self.current_percept: Optional[Percept] = None
        self.position_history: List[Tuple[float, float]] = []
        self.step_count = 0

        # 意图ID生成
        self.current_intention_id = 0

    def generate_intention_id(self, position: Tuple[float, float]) -> int:
        """基于当前情境生成意图ID"""
        # 简化：基于目标位置的哈希
        if self.intention_field.goal.target_position:
            return hash(self.intention_field.goal.target_position) % 10000
        return self.current_intention_id

    def perceive(self) -> Percept:
        """感知世界"""
        start_time = time.time() if 'time' in globals() else 0

        percept = self.env.get_percept()
        self.current_percept = percept

        # 更新自我感知
        self.interoception.update_prediction_error(percept.prediction_error)

        # 更新意图（基于感知）
        self._update_intentions_from_percept(percept)

        return percept

    def _update_intentions_from_percept(self, percept: Percept):
        """基于感知更新意图场"""
        current_pos = tuple(self.env.agent_pos)

        # 运动意图：基于最近移动方向
        if len(self.position_history) >= 2:
            direction = np.array(current_pos) - np.array(self.position_history[-2])
            speed = np.linalg.norm(direction)
            if speed > 0.01:
                self.intention_field.update_motor(direction, min(1.0, speed / 2))

        # 目标意图：基于感知特征（简化：寻找高奖励区域）
        reward_signal = percept.local_features[3] if len(percept.local_features) > 3 else 0
        if reward_signal > 0.5:
            self.intention_field.update_goal(current_pos, percept.local_features,
                                            urgency=reward_signal)

        # 价值意图：基于历史表现动态调整
        if len(self.position_history) > 50:
            exploration_rate = self._compute_exploration_rate()
            self.intention_field.value.exploration_vs_exploitation = 1.0 - exploration_rate

    def _compute_exploration_rate(self) -> float:
        """计算探索率（基于访问分布的熵）"""
        # 简化的探索度量
        if len(self.position_history) < 10:
            return 1.0
        recent_positions = self.position_history[-50:]
        unique_positions = len(set((int(p[0]), int(p[1])) for p in recent_positions))
        return min(1.0, unique_positions / 20)

    def think_and_act(self) -> Tuple[Action, Dict]:
        """
        思考并行动 - 核心决策循环

        Returns:
            action: 选择的动作
            meta: 元信息（用于分析）
        """
        self.step_count += 1
        current_pos = tuple(self.env.agent_pos)
        intention_id = self.generate_intention_id(current_pos)

        # 步骤1：尝试快层（透明行动）
        fast_action = self.fast_layer.try_respond(current_pos, intention_id)

        if fast_action is not None:
            # 快层命中 - 透明执行
            # 但仍需监控预测误差
            predicted_error = self._estimate_prediction_error(current_pos, fast_action)

            if predicted_error < 0.3:  # 预测可信
                return fast_action, {
                    'layer': 'fast',
                    'transparent': True,
                    'awareness': 0.1,
                    'intention_conflict': self.intention_field.compute_conflict()
                }
            else:
                # 预测误差高，使快层失效
                self.fast_layer.invalidate(current_pos, intention_id)

        # 步骤2：慢层处理（意识升起）
        # 检查意图冲突
        conflict = self.intention_field.compute_conflict()

        # 基于冲突和唤醒水平决定搜索深度
        felt_state = self.interoception.compute_felt_state()
        search_depth = self._compute_search_depth(conflict, felt_state)

        # 慢层搜索（简化版）
        action = self._slow_search(current_pos, search_depth)

        # 执行后巩固到快层
        # 先执行，获得真实反馈后再巩固

        meta = {
            'layer': 'slow',
            'transparent': False,
            'awareness': max(conflict, felt_state['arousal']),
            'intention_conflict': conflict,
            'search_depth': search_depth,
            'felt_state': felt_state
        }

        return action, meta

    def _estimate_prediction_error(self, pos: Tuple[float, float], action: Action) -> float:
        """估计给定动作的预测误差"""
        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < self.predictive_field.width and 0 <= y < self.predictive_field.height:
            return self.predictive_field.uncertainty_field[y, x, action.value]
        return 1.0

    def _compute_search_depth(self, conflict: float, felt_state: Dict) -> int:
        """计算搜索深度（基于意识和冲突）"""
        base_depth = 3

        # 高冲突或低信心 -> 更深搜索
        if conflict > 0.5:
            base_depth += 2
        if felt_state['confidence'] < 0.3:
            base_depth += 2
        if felt_state['arousal'] > 0.7:
            base_depth += 1

        return min(base_depth, 8)  # 上限

    def _slow_search(self, pos: Tuple[float, float], depth: int) -> Action:
        """
        慢层搜索 - 基于预测场的价值评估
        （简化实现，可用MCTS替换）
        """
        best_action = Action.STAY
        best_value = float('-inf')

        for action in Action:
            # 获取预测
            pred = self.predictive_field.get_prediction(int(pos[0]), int(pos[1]), action)

            # 价值 = 效价 - 不确定性惩罚
            value = pred.affective_valence - 0.5 * pred.expected_uncertainty

            # 加入意图对齐
            if action == Action.UP and self.intention_field.motor.preferred_direction[1] < -0.3:
                value += 0.2
            elif action == Action.DOWN and self.intention_field.motor.preferred_direction[1] > 0.3:
                value += 0.2
            elif action == Action.LEFT and self.intention_field.motor.preferred_direction[0] < -0.3:
                value += 0.2
            elif action == Action.RIGHT and self.intention_field.motor.preferred_direction[0] > 0.3:
                value += 0.2

            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def execute_and_learn(self, action: Action, meta: Dict) -> Tuple[Percept, float, bool]:
        """
        执行动作并学习

        Returns:
            同 env.step()
        """
        current_pos = tuple(self.env.agent_pos)
        intention_id = self.generate_intention_id(current_pos)

        # 执行
        percept, reward, done = self.env.step(action, self)

        # 计算预测误差
        prediction_error = abs(percept.prediction_error)
        self.interoception.update_prediction_error(prediction_error)

        # 学习：更新预测场
        self._update_predictive_field(current_pos, action, percept, reward)

        # 学习：巩固到快层（如果成功）
        if meta['layer'] == 'slow' and prediction_error < 0.3 and reward >= 0:
            self.fast_layer.consolidate(current_pos, intention_id, action, prediction_error)
        elif meta['layer'] == 'fast' and prediction_error > 0.5:
            # 快层失败，降级
            self.fast_layer.invalidate(current_pos, intention_id)

        # 记录
        self.position_history.append(tuple(self.env.agent_pos))

        return percept, reward, done

    def _update_predictive_field(self, pos: Tuple[float, float], action: Action,
                                 new_percept: Percept, reward: float):
        """更新预测场"""
        x, y = int(pos[0]), int(pos[1])

        if self.current_percept is not None:
            delta = new_percept.local_features - self.current_percept.local_features
        else:
            delta = np.zeros(self.perception_dim)

        consequence = Consequence(
            delta_perception=delta,
            expected_uncertainty=min(1.0, abs(new_percept.prediction_error) * 2),
            affective_valence=np.tanh(reward)
        )

        self.predictive_field.update_prediction(x, y, action, consequence, self.lr)

    def get_cognitive_state(self) -> Dict:
        """获取完整认知状态"""
        return {
            'position': tuple(self.env.agent_pos),
            'fast_layer_stats': self.fast_layer.get_stats(),
            'intention_conflict': self.intention_field.compute_conflict(),
            'dominant_intention': self.intention_field.get_dominant_intention(),
            'felt_state': self.interoception.compute_felt_state(),
            'awareness_level': self._compute_awareness_level(),
            'field_uncertainty': np.mean(self.predictive_field.uncertainty_field)
        }

    def _compute_awareness_level(self) -> float:
        """计算意识水平"""
        conflict = self.intention_field.compute_conflict()
        felt = self.interoception.compute_felt_state()

        # 意识 = 意图冲突 + 低信心 + 高唤醒
        awareness = (
            0.4 * conflict +
            0.3 * (1 - felt['confidence']) +
            0.3 * felt['arousal']
        )
        return min(1.0, awareness)


# ============================================================================
# 环境类（兼容原有接口）
# ============================================================================

class SimpleEnvironment:
    """简化的2D环境"""
    def __init__(self, width: int = 20, height: int = 20, perception_dim: int = 8):
        self.width = width
        self.height = height
        self.perception_dim = perception_dim

        self.reward_regions = []
        self.world_features = np.zeros((height, width, perception_dim))
        self._generate_world()

        self.agent_pos = np.array([width // 2, height // 2], dtype=float)

    def _generate_world(self):
        """生成世界"""
        self.world_features[:, :, 4:] = np.random.rand(self.height, self.width, 4) * 0.2

        num_regions = 3
        for _ in range(num_regions):
            cx, cy = random.randint(3, self.width-3), random.randint(3, self.height-3)
            color = np.random.rand(3)
            self.reward_regions.append((cx, cy, 2.0, color))

            for y in range(max(0, int(cy-3)), min(self.height, int(cy+4))):
                for x in range(max(0, int(cx-3)), min(self.width, int(cx+4))):
                    dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                    if dist < 3:
                        intensity = np.exp(-dist / 1.5)
                        self.world_features[y, x, :3] = color * intensity
                        self.world_features[y, x, 3] = intensity * 2

    def get_percept(self, x: Optional[float] = None, y: Optional[float] = None) -> Percept:
        if x is None:
            x, y = self.agent_pos

        x_int, y_int = int(x), int(y)

        if not (0 <= x_int < self.width and 0 <= y_int < self.height):
            return Percept(
                local_features=np.zeros(self.perception_dim),
                proprioception=np.array([x, y]),
                prediction_error=1.0
            )

        local = self.world_features[y_int, x_int].copy()
        local += np.random.randn(self.perception_dim) * 0.05
        local = np.clip(local, 0, 1)

        return Percept(
            local_features=local,
            proprioception=np.array([x, y]),
            prediction_error=0.05
        )

    def step(self, action: Action, agent) -> Tuple[Percept, float, bool]:
        new_pos = self.agent_pos.copy()

        if action == Action.UP:
            new_pos[1] -= 1
        elif action == Action.DOWN:
            new_pos[1] += 1
        elif action == Action.LEFT:
            new_pos[0] -= 1
        elif action == Action.RIGHT:
            new_pos[0] += 1

        new_pos[0] = np.clip(new_pos[0], 0, self.width - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.height - 1)

        self.agent_pos = new_pos

        percept = self.get_percept()

        x_int, y_int = int(self.agent_pos[0]), int(self.agent_pos[1])
        reward = self.world_features[y_int, x_int, 3] * 0.5

        for cx, cy, radius, _ in self.reward_regions:
            dist = np.sqrt((x_int-cx)**2 + (y_int-cy)**2)
            if dist < radius:
                reward += 1.0

        return percept, reward, False


# ============================================================================
# 运行演示
# ============================================================================

def run_multiscale_simulation(steps: int = 300, render_every: int = 30):
    """运行多尺度模拟"""
    print("=" * 70)
    print("多尺度预测性注意力系统 - 模拟运行")
    print("=" * 70)
    print("\n系统架构:")
    print("  - 快层(Fast): 习惯缓存，熟练后透明执行")
    print("  - 慢层(Slow): 预测场搜索，意识升起时激活")
    print("  - 意图场: 快/中/慢三层意图协调")
    print("  - 自我感知: 内感受监控，预测误差感知")
    print()

    env = SimpleEnvironment(width=20, height=20)
    agent = MultiScalePredictiveAgent(env)

    total_reward = 0
    fast_layer_uses = 0
    slow_layer_uses = 0

    for step in range(steps):
        # 感知
        agent.perceive()

        # 思考
        action, meta = agent.think_and_act()

        # 统计
        if meta['layer'] == 'fast':
            fast_layer_uses += 1
        else:
            slow_layer_uses += 1

        # 执行
        percept, reward, done = agent.execute_and_learn(action, meta)
        total_reward += reward

        # 报告
        if step % render_every == 0:
            state = agent.get_cognitive_state()
            print(f"\n[Step {step}]")
            print(f"  位置: {state['position']}")
            print(f"  动作: {action.name:10s} | 使用层: {meta['layer']:4s} | "
                  f"意识水平: {state['awareness_level']:.2f}")
            print(f"  快层统计: {state['fast_layer_stats']}")
            print(f"  意图冲突: {state['intention_conflict']:.3f} | "
                  f"主导意图: {state['dominant_intention']}")
            print(f"  感觉状态: {state['felt_state']}")
            print(f"  累计奖励: {total_reward:.2f}")

    print("\n" + "=" * 70)
    print("模拟完成")
    print(f"  总计奖励: {total_reward:.2f}")
    print(f"  快层使用: {fast_layer_uses} ({fast_layer_uses/steps*100:.1f}%)")
    print(f"  慢层使用: {slow_layer_uses} ({slow_layer_uses/steps*100:.1f}%)")
    print(f"  最终意识水平: {agent.get_cognitive_state()['awareness_level']:.3f}")
    print("=" * 70)

    return agent


if __name__ == "__main__":
    agent = run_multiscale_simulation(steps=500)
