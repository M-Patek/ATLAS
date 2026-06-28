"""
Predictive Attention System
基于"预测后果而非建模世界"的2D认知架构
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum
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
    """
    后果：不是完整状态，而是预期的感知变化
    """
    delta_perception: np.ndarray  # 感知变化向量
    expected_uncertainty: float    # 预测的不确定性
    affective_valence: float       # 情感效价（粗略的好/坏信号）

    def __post_init__(self):
        # 确保delta_perception是1D向量
        if len(self.delta_perception.shape) > 1:
            self.delta_perception = self.delta_perception.flatten()


@dataclass
class Percept:
    """
    感知：智能体实际接收到的信息
    注意：这不是客观世界状态，而是主观感知
    """
    local_features: np.ndarray     # 局部特征（颜色、纹理等）
    proprioception: np.ndarray     # 本体感觉（位置、速度）
    prediction_error: float        # 预测误差（surprise）


class PredictiveField:
    """
    预测场：空间不是被建模的，而是被编织成可能性的织物
    每个位置存储对该位置的"后果预测"
    """
    def __init__(self, width: int, height: int, perception_dim: int = 8):
        self.width = width
        self.height = height
        self.perception_dim = perception_dim

        # 初始化：每个位置对每个动作的预测
        # 形状: [height, width, num_actions, consequence_dim]
        num_actions = len(Action)
        consequence_dim = perception_dim + 2  # delta + uncertainty + valence

        # 初始化预测场为零均值高斯
        self.field = np.zeros((height, width, num_actions, consequence_dim))
        self.uncertainty_field = np.ones((height, width, num_actions)) * 0.5

        # 访问统计：记录哪些位置被频繁访问/预测
        self.visit_count = np.zeros((height, width))

    def get_prediction(self, x: int, y: int, action: Action) -> Consequence:
        """获取特定位置-动作的后果预测"""
        x, y = int(x), int(y)
        if not (0 <= x < self.width and 0 <= y < self.height):
            return Consequence(
                delta_perception=np.zeros(self.perception_dim),
                expected_uncertainty=1.0,
                affective_valence=0.0
            )

        pred = self.field[y, x, action.value]
        delta = pred[:self.perception_dim]
        uncertainty = pred[self.perception_dim]
        valence = pred[self.perception_dim + 1]

        return Consequence(delta, uncertainty, valence)

    def update_prediction(self, x: int, y: int, action: Action,
                         consequence: Consequence, learning_rate: float = 0.1):
        """基于实际观察更新预测"""
        x, y = int(x), int(y)
        if not (0 <= x < self.width and 0 <= y < self.height):
            return

        # 构建更新目标
        target = np.concatenate([
            consequence.delta_perception,
            [consequence.expected_uncertainty],
            [consequence.affective_valence]
        ])

        # 简单指数移动平均更新
        old_pred = self.field[y, x, action.value]
        self.field[y, x, action.value] = (1 - learning_rate) * old_pred + learning_rate * target

        # 更新不确定性估计（预测误差）
        prediction_error = np.linalg.norm(old_pred - target)
        self.uncertainty_field[y, x, action.value] = (
            (1 - learning_rate) * self.uncertainty_field[y, x, action.value] +
            learning_rate * prediction_error
        )

        self.visit_count[y, x] += 1

    def get_attention_salience(self) -> np.ndarray:
        """
        计算空间每个位置的注意显著性
        基于：不确定性 × 访问频率的倒数（探索-利用平衡）
        """
        # 避免除零
        exploration_bonus = 1.0 / (1.0 + np.sqrt(self.visit_count))
        mean_uncertainty = np.mean(self.uncertainty_field, axis=2)

        salience = mean_uncertainty * exploration_bonus
        return salience


class ConsequencePredictor(nn.Module):
    """
    神经网络预测器：学习 (位置, 当前感知, 动作) → 后果
    这是快速近似，与预测场互补
    """
    def __init__(self, perception_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.perception_dim = perception_dim

        # 输入: 位置(2) + 感知(perception_dim) + 动作(1) = 11
        input_dim = 2 + perception_dim + 1
        output_dim = perception_dim + 2  # delta + uncertainty + valence

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, position: torch.Tensor,
                perception: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            position: [B, 2] - (x, y) 归一化到 [0, 1]
            perception: [B, perception_dim] - 当前感知向量
            action: [B, 1] - 动作索引
        Returns:
            [B, perception_dim + 2] - 预测的delta + uncertainty + valence
        """
        x = torch.cat([position, perception, action], dim=-1)
        return self.net(x)


class AttentionMechanism:
    """
    注意力机制：有限的认知资源分配
    不是全空间处理，而是选择性关注预测价值高的区域
    """
    def __init__(self, max_attention_spots: int = 5, attention_radius: float = 3.0):
        self.max_spots = max_attention_spots
        self.radius = attention_radius
        self.attention_centers: List[Tuple[int, int]] = []
        self.attention_weights: np.ndarray = np.array([])

    def compute_attention(self, agent_pos: Tuple[float, float],
                         predictive_field: PredictiveField,
                         current_percept: Percept) -> np.ndarray:
        """
        计算注意力分布（软掩码）
        基于：预测不确定性 + 潜在信息增益 + 与当前目标相关度
        """
        height, width = predictive_field.height, predictive_field.width

        # 获取显著性图
        salience = predictive_field.get_attention_salience()

        # 添加对智能体周围的关注（近处通常更重要）
        y_pos, x_pos = np.ogrid[:height, :width]
        distance_to_agent = np.sqrt((x_pos - agent_pos[0])**2 + (y_pos - agent_pos[1])**2)
        proximity_bonus = np.exp(-distance_to_agent / self.radius)
        salience = salience * (1 + proximity_bonus)

        # 基于预测误差：如果当前感知与预期不符，增加该区域注意
        if current_percept.prediction_error > 0.1:
            error_spot = (int(agent_pos[0]), int(agent_pos[1]))
            # 在误差位置添加高斯峰值
            gaussian = np.exp(-((x_pos - error_spot[0])**2 + (y_pos - error_spot[1])**2) / 4)
            salience += gaussian * current_percept.prediction_error

        # 归一化
        if salience.sum() > 0:
            salience = salience / salience.sum()

        # 提取Top-K注意中心
        flat_salience = salience.flatten()
        top_k_indices = np.argsort(flat_salience)[-self.max_spots:]

        self.attention_centers = []
        for idx in top_k_indices:
            y, x = divmod(idx, width)
            self.attention_centers.append((x, y))

        self.attention_weights = flat_salience[top_k_indices]
        if self.attention_weights.sum() > 0:
            self.attention_weights /= self.attention_weights.sum()

        return salience

    def get_attention_mask(self, height: int, width: int) -> np.ndarray:
        """生成注意力掩码，高斯blob表示注意区域"""
        mask = np.zeros((height, width))
        y_pos, x_pos = np.ogrid[:height, :width]

        for (cx, cy), weight in zip(self.attention_centers, self.attention_weights):
            gaussian = np.exp(-((x_pos - cx)**2 + (y_pos - cy)**2) / (self.radius**2))
            mask += gaussian * weight

        mask = np.clip(mask, 0, 1)
        return mask


class ValueFunction:
    """
    价值函数：评估"预测序列"的好坏
    不是V(s)，而是V(预期后果链)
    """
    def __init__(self, gamma: float = 0.95, horizon: int = 5):
        self.gamma = gamma
        self.horizon = horizon

    def evaluate_consequence_chain(self,
                                   chain: List[Consequence],
                                   discount: Optional[float] = None) -> float:
        """
        评估后果链的价值
        包含：即时效价 + 累积预期 + 信息增益奖励
        """
        if discount is None:
            discount = self.gamma

        total_value = 0.0
        current_discount = 1.0
        total_uncertainty_reduction = 0.0

        for i, cons in enumerate(chain):
            # 基础效价
            immediate = cons.affective_valence

            # 信息增益奖励：降低不确定性是有价值的
            info_gain = max(0, 0.5 - cons.expected_uncertainty)
            total_uncertainty_reduction += info_gain * current_discount

            # 累积
            total_value += current_discount * (immediate + 0.1 * info_gain)
            current_discount *= discount

        # 对探索给予额外奖励（当不确定性高时）
        exploration_bonus = math.sqrt(total_uncertainty_reduction) * 0.1

        return total_value + exploration_bonus

    def select_action(self, current_pos: Tuple[float, float],
                     predictive_field: PredictiveField,
                     current_percept: Percept) -> Tuple[Action, float]:
        """基于预测后果链选择动作"""
        best_action = Action.STAY
        best_value = float('-inf')
        action_values = {}

        for action in Action:
            # 获取即时预测
            immediate = predictive_field.get_prediction(
                int(current_pos[0]), int(current_pos[1]), action
            )

            # 构建简单后果链（单步预测）
            # 更复杂的版本可以做多步rollout
            chain = [immediate]

            # 评估
            value = self.evaluate_consequence_chain(chain)
            action_values[action] = value

            if value > best_value:
                best_value = value
                best_action = action

        return best_action, best_value


class SimpleEnvironment:
    """
    简单的2D环境，用于测试
    """
    def __init__(self, width: int = 20, height: int = 20,
                 perception_dim: int = 8):
        self.width = width
        self.height = height
        self.perception_dim = perception_dim

        # 世界内容：颜色特征（简化的）
        # 特征0-2: RGB颜色信息
        # 特征3: "能量"或"奖励"密度
        # 特征4-7: 随机纹理
        self.reward_regions = []
        self.world_features = np.zeros((height, width, perception_dim))
        self._generate_world()

        self.agent_pos = np.array([width // 2, height // 2], dtype=float)

    def _generate_world(self):
        """生成一个有结构的世界"""
        # 基础随机纹理
        self.world_features[:, :, 4:] = np.random.rand(self.height, self.width, 4) * 0.2

        # 创建几个"奖励区域"
        num_regions = 3
        for _ in range(num_regions):
            cx, cy = random.randint(3, self.width-3), random.randint(3, self.height-3)
            color = np.random.rand(3)
            self.reward_regions.append((cx, cy, 2.0, color))  # x, y, radius, color

            # 在世界特征中标记这些区域
            for y in range(max(0, int(cy-3)), min(self.height, int(cy+4))):
                for x in range(max(0, int(cx-3)), min(self.width, int(cx+4))):
                    dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                    if dist < 3:
                        intensity = np.exp(-dist / 1.5)
                        self.world_features[y, x, :3] = color * intensity
                        self.world_features[y, x, 3] = intensity * 2  # 奖励信号

    def get_percept(self, x: Optional[float] = None, y: Optional[float] = None) -> Percept:
        """获取位置(x,y)的感知"""
        if x is None:
            x, y = self.agent_pos

        x_int, y_int = int(x), int(y)

        # 边界处理
        if not (0 <= x_int < self.width and 0 <= y_int < self.height):
            return Percept(
                local_features=np.zeros(self.perception_dim),
                proprioception=np.array([x, y]),
                prediction_error=1.0  # 最大误差（未知区域）
            )

        # 获取局部特征
        local = self.world_features[y_int, x_int].copy()

        # 添加轻微噪声
        local += np.random.randn(self.perception_dim) * 0.05
        local = np.clip(local, 0, 1)

        return Percept(
            local_features=local,
            proprioception=np.array([x, y]),
            prediction_error=0.05  # 初始假设小误差
        )

    def step(self, action: Action, agent: 'PredictiveAgent') -> Tuple[Percept, float, bool]:
        """执行动作，返回(感知, 奖励, 是否终止)"""
        # 计算新位置
        new_pos = self.agent_pos.copy()

        if action == Action.UP:
            new_pos[1] -= 1
        elif action == Action.DOWN:
            new_pos[1] += 1
        elif action == Action.LEFT:
            new_pos[0] -= 1
        elif action == Action.RIGHT:
            new_pos[0] += 1
        elif action == Action.INTERACT:
            # 交互动作：如果在奖励区域附近，获取更高奖励
            pass

        # 边界约束
        new_pos[0] = np.clip(new_pos[0], 0, self.width - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.height - 1)

        self.agent_pos = new_pos

        # 获取感知
        percept = self.get_percept()

        # 计算奖励（基于world_features中的奖励维度）
        x_int, y_int = int(self.agent_pos[0]), int(self.agent_pos[1])
        reward = self.world_features[y_int, x_int, 3] * 0.5

        # 如果离奖励中心近，额外奖励
        for cx, cy, radius, _ in self.reward_regions:
            dist = np.sqrt((x_int-cx)**2 + (y_int-cy)**2)
            if dist < radius:
                reward += 1.0

        return percept, reward, False


class PredictiveAgent:
    """
    预测性认知智能体
    """
    def __init__(self, env: SimpleEnvironment,
                 perception_dim: int = 8,
                 learning_rate: float = 0.1):
        self.env = env
        self.perception_dim = perception_dim
        self.lr = learning_rate

        # 认知架构组件
        self.predictive_field = PredictiveField(
            env.width, env.height, perception_dim
        )
        self.attention = AttentionMechanism()
        self.value_fn = ValueFunction()

        # 神经网络预测器（用于快速近似）
        self.nn_predictor = ConsequencePredictor(perception_dim)
        self.optimizer = torch.optim.Adam(self.nn_predictor.parameters(), lr=0.001)

        # 当前状态
        self.current_percept: Optional[Percept] = None
        self.position_history = []
        self.prediction_errors = []

    def perceive(self) -> Percept:
        """感知世界"""
        percept = self.env.get_percept()
        self.current_percept = percept
        return percept

    def think(self) -> Action:
        """思考：更新注意力，评估选项，选择行动"""
        if self.current_percept is None:
            self.perceive()

        pos = tuple(self.env.agent_pos)
        self.position_history.append(pos)

        # 1. 更新注意力分配
        attention_map = self.attention.compute_attention(
            pos, self.predictive_field, self.current_percept
        )

        # 2. 基于价值函数选择动作
        action, expected_value = self.value_fn.select_action(
            pos, self.predictive_field, self.current_percept
        )

        # 3. 有限注意力：只更新注意区域的预测
        self._update_predictions_in_attention_window(action)

        return action

    def _update_predictions_in_attention_window(self, intended_action: Action):
        """在注意力窗口内更新预测"""
        mask = self.attention.get_attention_mask(
            self.env.height, self.env.width
        )

        # 只更新高注意力区域
        threshold = 0.1
        update_positions = np.argwhere(mask > threshold)

        for y, x in update_positions:
            # 获取当前位置的当前感知
            percept = self.env.get_percept(x, y)

            # 对每个动作更新预测
            for action in Action:
                # 这里简化了：实际应该基于模拟或经验
                # 使用当前的field值作为基线，添加小扰动
                current_pred = self.predictive_field.get_prediction(x, y, action)

                # 基于注意力权重调整学习率
                attention_weight = mask[y, x]
                effective_lr = self.lr * attention_weight

                # 生成一个简化的后果（实际应从经验学习）
                delta = percept.local_features * 0.1  # 假设小变化
                consequence = Consequence(
                    delta_perception=delta,
                    expected_uncertainty=current_pred.expected_uncertainty * 0.95,
                    affective_valence=np.mean(delta[:3])  # 基于颜色估计效价
                )

                self.predictive_field.update_prediction(
                    x, y, action, consequence, effective_lr
                )

    def learn_from_experience(self,
                             previous_pos: Tuple[float, float],
                             action: Action,
                             new_percept: Percept,
                             reward: float):
        """从实际经验学习更新预测"""
        x, y = int(previous_pos[0]), int(previous_pos[1])

        # 计算实际的感知变化
        if self.current_percept is not None:
            delta = new_percept.local_features - self.current_percept.local_features
            prediction_error = np.linalg.norm(delta)
            self.prediction_errors.append(prediction_error)
        else:
            delta = np.zeros(self.perception_dim)
            prediction_error = 0.0

        # 构建实际后果
        actual_consequence = Consequence(
            delta_perception=delta,
            expected_uncertainty=min(1.0, prediction_error * 2),  # 大误差意味着高不确定
            affective_valence=np.tanh(reward)  # 奖励作为效价信号
        )

        # 更新预测场
        self.predictive_field.update_prediction(
            x, y, action, actual_consequence, self.lr
        )

        # 更新当前感知中的预测误差
        new_percept_with_error = Percept(
            local_features=new_percept.local_features,
            proprioception=new_percept.proprioception,
            prediction_error=prediction_error
        )
        self.current_percept = new_percept_with_error

        # 定期训练神经网络预测器
        if len(self.prediction_errors) > 0 and len(self.prediction_errors) % 10 == 0:
            self._train_nn_predictor()

    def _train_nn_predictor(self):
        """训练神经网络预测器"""
        # 简化的训练：从预测场采样
        batch_size = 32
        xs, ys = np.random.randint(0, self.env.width, batch_size), \
                 np.random.randint(0, self.env.height, batch_size)
        actions = np.random.randint(0, len(Action), batch_size)

        positions = torch.FloatTensor(np.column_stack([ys, xs])) / \
                   torch.FloatTensor([[self.env.height, self.env.width]])
        actions_t = torch.FloatTensor(actions).unsqueeze(1) / len(Action)

        # 简单的感知特征（实际应从历史数据）
        percepts = torch.randn(batch_size, self.perception_dim) * 0.1

        # 从预测场获取目标
        targets = []
        for x, y, a in zip(xs, ys, actions):
            pred = self.predictive_field.get_prediction(x, y, Action(a))
            target = np.concatenate([
                pred.delta_perception,
                [pred.expected_uncertainty, pred.affective_valence]
            ])
            targets.append(target)
        targets = torch.FloatTensor(np.array(targets))

        # 训练步骤
        self.optimizer.zero_grad()
        predictions = self.nn_predictor(positions, percepts, actions_t)
        loss = F.mse_loss(predictions, targets)
        loss.backward()
        self.optimizer.step()

    def act(self, action: Action) -> Tuple[Percept, float, bool]:
        """执行动作"""
        return self.env.step(action, self)

    def get_cognitive_state(self) -> Dict:
        """获取认知状态摘要（用于可视化）"""
        return {
            'position': tuple(self.env.agent_pos),
            'position_history': self.position_history[-50:],  # 最近50步
            'attention_centers': self.attention.attention_centers,
            'attention_weights': self.attention.attention_weights.tolist(),
            'prediction_errors': self.prediction_errors[-100:],
            'field_uncertainty': np.mean(self.predictive_field.uncertainty_field),
            'visit_entropy': self._compute_visit_entropy()
        }

    def _compute_visit_entropy(self) -> float:
        """计算访问分布的熵（探索程度的度量）"""
        visit_dist = self.predictive_field.visit_count.flatten()
        if visit_dist.sum() == 0:
            return 0.0
        visit_dist = visit_dist / visit_dist.sum()
        # 忽略零概率项
        visit_dist = visit_dist[visit_dist > 0]
        return -np.sum(visit_dist * np.log(visit_dist + 1e-10))


def run_simulation(steps: int = 200, render_every: int = 20):
    """运行完整模拟"""
    print("=" * 60)
    print("预测性认知智能体 - 模拟运行")
    print("=" * 60)

    # 创建环境
    env = SimpleEnvironment(width=20, height=20)

    # 创建智能体
    agent = PredictiveAgent(env)

    # 运行
    total_reward = 0

    for step in range(steps):
        pos_before = tuple(env.agent_pos)

        # 感知
        agent.perceive()

        # 思考
        action = agent.think()

        # 行动
        percept, reward, done = agent.act(action)

        # 学习
        agent.learn_from_experience(pos_before, action, percept, reward)

        total_reward += reward

        # 报告
        if step % render_every == 0:
            state = agent.get_cognitive_state()
            print(f"\n[Step {step}]")
            print(f"  位置: {state['position']}")
            print(f"  动作: {action.name}")
            print(f"  奖励: {reward:.3f} (总计: {total_reward:.3f})")
            print(f"  注意力中心: {state['attention_centers']}")
            print(f"  平均预测不确定性: {state['field_uncertainty']:.3f}")
            print(f"  探索熵: {state['visit_entropy']:.3f}")

    print("\n" + "=" * 60)
    print(f"模拟完成。总计奖励: {total_reward:.3f}")
    print(f"最终预测不确定性: {agent.get_cognitive_state()['field_uncertainty']:.3f}")
    print(f"探索熵: {agent.get_cognitive_state()['visit_entropy']:.3f}")
    print("=" * 60)

    return agent


if __name__ == "__main__":
    agent = run_simulation(steps=300)
