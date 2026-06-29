"""
ATLAS Learning: Neural Space
神经网络空间

使用深度神经网络学习空间表示
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Callable
from dataclasses import dataclass

from ..core.space import CognitiveSpace, register_space


@dataclass
class TrainingSample:
    """训练样本"""
    observation: np.ndarray
    position: Tuple[int, int]
    target_position: Optional[Tuple[int, int]]
    true_distance: float


class SpatialEncoder:
    """
    空间编码器

    将观测编码为隐向量
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 16):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 简单的MLP权重
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b3 = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        前向传播

        Returns:
            (output, activations for backprop)
        """
        # 层1
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU

        # 层2
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0, z2)  # ReLU

        # 层3
        z3 = a2 @ self.W3 + self.b3
        a3 = np.tanh(z3)  # tanh归一化

        return a3, [x, a1, a2, a3]

    def encode(self, observation: np.ndarray) -> np.ndarray:
        """编码观测"""
        output, _ = self.forward(observation)
        return output


class MetricNetwork:
    """
    度量网络

    学习两个隐向量之间的距离
    """

    def __init__(self, embedding_dim: int = 16, hidden_dim: int = 32):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # 权重
        self.W1 = np.random.randn(embedding_dim * 2, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, 1) * 0.1
        self.b3 = np.zeros(1)

    def forward(self, z1: np.ndarray, z2: np.ndarray) -> Tuple[float, List]:
        """
        前向传播

        Args:
            z1, z2: 两个位置的编码

        Returns:
            (predicted_distance, activations)
        """
        # 拼接
        x = np.concatenate([z1, z2])

        # 层1
        h1 = np.maximum(0, x @ self.W1 + self.b1)

        # 层2
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)

        # 输出（确保正数）
        distance = np.maximum(0, h2 @ self.W3 + self.b3)[0]

        return distance, [x, h1, h2, distance]

    def compute_distance(self, z1: np.ndarray, z2: np.ndarray) -> float:
        """计算距离"""
        dist, _ = self.forward(z1, z2)
        return dist


@register_space("neural")
class NeuralSpace(CognitiveSpace):
    """
    神经网络认知空间

    使用神经网络端到端学习空间表示和距离度量

    Example:
        space = NeuralSpace(
            width=40, height=20,
            encoder=SpatialEncoder(input_dim=100),
            metric_network=MetricNetwork()
        )

        # 在线学习
        space.train_step(obs1, obs2, true_distance)
    """

    def __init__(self,
                 width: int,
                 height: int,
                 encoder: Optional[SpatialEncoder] = None,
                 metric_network: Optional[MetricNetwork] = None,
                 observation_shape: Tuple[int, ...] = (10,),
                 embedding_dim: int = 16,
                 learning_rate: float = 0.001,
                 **kwargs):
        super().__init__(width, height, name="neural")

        # 网络组件
        input_dim = np.prod(observation_shape)
        self.encoder = encoder or SpatialEncoder(input_dim, output_dim=embedding_dim)
        self.metric = metric_network or MetricNetwork(embedding_dim)

        self.observation_shape = observation_shape
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate

        # 位置到编码的缓存
        self.position_embeddings: Dict[Tuple[int, int], np.ndarray] = {}

        # 观测历史（用于训练）
        self.training_buffer: List[TrainingSample] = []
        self.max_buffer_size = 1000

    def _observation_to_vector(self, observation: Dict[str, Any]) -> np.ndarray:
        """将观测转换为向量"""
        # 简单版本：提取数值特征
        features = []

        if 'image' in observation:
            # 如果是图像，展平
            img = np.array(observation['image']).flatten()
            features.extend(img[:50])  # 限制大小

        if 'lidar' in observation:
            lidar = np.array(observation['lidar']).flatten()
            features.extend(lidar[:20])

        # 添加位置信息
        if 'position' in observation:
            pos = observation['position']
            features.extend([pos[0] / self.width, pos[1] / self.height])

        # 填充或截断
        target_size = self.encoder.input_dim
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        features = features[:target_size]

        return np.array(features)

    def compute_distance(self, pos1: Tuple[int, int],
                        pos2: Tuple[int, int]) -> float:
        """
        神经网络距离

        如果位置没有缓存的编码，使用启发式
        """
        # 获取或创建编码
        if pos1 in self.position_embeddings:
            z1 = self.position_embeddings[pos1]
        else:
            # 使用启发式
            return self._heuristic_distance(pos1, pos2)

        if pos2 in self.position_embeddings:
            z2 = self.position_embeddings[pos2]
        else:
            return self._heuristic_distance(pos1, pos2)

        # 神经网络距离
        return self.metric.compute_distance(z1, z2)

    def _heuristic_distance(self, pos1: Tuple[int, int],
                           pos2: Tuple[int, int]) -> float:
        """启发式距离（欧氏）"""
        return np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

    def get_heuristic(self, pos: Tuple[int, int],
                     goal: Tuple[int, int]) -> float:
        """启发式"""
        return self._heuristic_distance(pos, goal)

    def update_from_observation(self, position: Tuple[int, int],
                                observation: Dict[str, Any]) -> None:
        """
        根据观测更新

        编码观测并缓存
        """
        # 转换为向量
        obs_vector = self._observation_to_vector(observation)

        # 编码
        embedding = self.encoder.encode(obs_vector)

        # 缓存
        self.position_embeddings[position] = embedding

        # 添加到训练缓冲区
        target = observation.get('goal_position')
        if target and 'true_distance' in observation:
            sample = TrainingSample(
                observation=obs_vector,
                position=position,
                target_position=target,
                true_distance=observation['true_distance']
            )
            self.training_buffer.append(sample)

            if len(self.training_buffer) > self.max_buffer_size:
                self.training_buffer.pop(0)

    def train_step(self,
                  obs1: np.ndarray,
                  obs2: np.ndarray,
                  true_distance: float) -> float:
        """
        单步训练

        Args:
            obs1, obs2: 两个观测
            true_distance: 真实距离

        Returns:
            loss
        """
        # 前向
        z1, cache1 = self.encoder.forward(obs1)
        z2, cache2 = self.encoder.forward(obs2)
        pred_dist, metric_cache = self.metric.forward(z1, z2)

        # 损失
        loss = (pred_dist - true_distance) ** 2

        # 反向传播（简化版梯度下降）
        grad_pred = 2 * (pred_dist - true_distance)

        # 这里简化处理，实际应该用autograd
        # 只更新最后一层作为演示
        self.metric.W3 -= self.learning_rate * grad_pred * metric_cache[2].reshape(-1, 1)
        self.metric.b3 -= self.learning_rate * grad_pred

        return loss

    def train_epoch(self, batch_size: int = 32) -> float:
        """
        训练一个epoch

        Returns:
            平均损失
        """
        if len(self.training_buffer) < batch_size:
            return 0.0

        # 采样批次
        indices = np.random.choice(len(self.training_buffer), batch_size, replace=False)
        total_loss = 0.0

        for idx in indices:
            sample = self.training_buffer[idx]

            # 找目标位置的观测
            target_obs = None
            for s in self.training_buffer:
                if s.position == sample.target_position:
                    target_obs = s.observation
                    break

            if target_obs is not None:
                loss = self.train_step(sample.observation, target_obs, sample.true_distance)
                total_loss += loss

        return total_loss / batch_size

    def get_visualization_fields(self) -> Dict[str, np.ndarray]:
        """可视化嵌入空间"""
        # 创建嵌入场
        embedding_map = np.zeros((self.width, self.height, self.embedding_dim))

        for (x, y), emb in self.position_embeddings.items():
            if 0 <= x < self.width and 0 <= y < self.height:
                embedding_map[x, y] = emb

        # 返回前3维作为RGB
        return {
            'embedding_r': embedding_map[:, :, 0] if self.embedding_dim > 0 else np.zeros((self.width, self.height)),
            'embedding_g': embedding_map[:, :, 1] if self.embedding_dim > 1 else np.zeros((self.width, self.height)),
            'embedding_b': embedding_map[:, :, 2] if self.embedding_dim > 2 else np.zeros((self.width, self.height)),
        }


@register_space("contrastive_neural")
class ContrastiveNeuralSpace(NeuralSpace):
    """
    对比学习神经网络空间

    使用对比损失学习更好的表示
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "contrastive_neural"
        self.temperature = 0.1

    def contrastive_loss(self,
                        anchor: np.ndarray,
                        positive: np.ndarray,
                        negatives: List[np.ndarray]) -> float:
        """
        对比损失

        拉近正样本，推远负样本
        """
        # 编码
        z_anchor = self.encoder.encode(anchor)
        z_pos = self.encoder.encode(positive)

        # 正样本相似度
        pos_sim = np.dot(z_anchor, z_pos) / (np.linalg.norm(z_anchor) * np.linalg.norm(z_pos) + 1e-8)
        pos_sim = np.exp(pos_sim / self.temperature)

        # 负样本相似度
        neg_sims = []
        for neg in negatives:
            z_neg = self.encoder.encode(neg)
            sim = np.dot(z_anchor, z_neg) / (np.linalg.norm(z_anchor) * np.linalg.norm(z_neg) + 1e-8)
            neg_sims.append(np.exp(sim / self.temperature))

        # InfoNCE损失
        loss = -np.log(pos_sim / (pos_sim + sum(neg_sims)))

        return loss

    def train_contrastive(self,
                         anchor_obs: np.ndarray,
                         positive_obs: np.ndarray,
                         negative_obs_list: List[np.ndarray]) -> float:
        """对比学习训练步"""
        loss = self.contrastive_loss(anchor_obs, positive_obs, negative_obs_list)

        # 简化更新（实际应该用反向传播）
        # 这里仅作演示
        return loss
