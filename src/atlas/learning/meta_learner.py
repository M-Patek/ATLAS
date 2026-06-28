"""
ATLAS Learning: Meta-Learning
元学习

实现学习如何学习的系统，自动为任务选择最优空间类型
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from collections import defaultdict
import json


class TaskEmbedding:
    """
    任务嵌入网络

    将任务描述编码为向量表示
    """

    def __init__(self, embedding_dim: int = 16):
        self.embedding_dim = embedding_dim
        self.feature_extractors: Dict[str, Callable] = {}

    def register_feature(self, name: str, extractor: Callable[[Any], float]):
        """注册特征提取器"""
        self.feature_extractors[name] = extractor

    def embed(self, task_description: Dict[str, Any]) -> np.ndarray:
        """
        将任务描述转换为嵌入向量

        Args:
            task_description: 任务描述字典，可能包含:
                - 'obstacle_density': 障碍物密度
                - 'dynamic_ratio': 动态障碍物比例
                - 'goal_distance': 目标距离
                - 'required_precision': 精度要求
                - ...

        Returns:
            嵌入向量
        """
        features = []

        # 预定义特征
        feature_keys = [
            'obstacle_density',
            'dynamic_ratio',
            'goal_distance',
            'required_precision',
            'exploration_importance',
            'time_constraint',
            'memory_available',
        ]

        for key in feature_keys:
            value = task_description.get(key, 0.5)
            # 归一化到 [0, 1]
            if key == 'obstacle_density':
                value = np.clip(value, 0, 1)
            elif key == 'goal_distance':
                value = np.clip(value / 100, 0, 1)
            elif key == 'time_constraint':
                value = np.clip(value, 0, 1)

            features.append(value)

        # 扩展或截断到 embedding_dim
        if len(features) < self.embedding_dim:
            features.extend([0.5] * (self.embedding_dim - len(features)))
        elif len(features) > self.embedding_dim:
            features = features[:self.embedding_dim]

        return np.array(features)


class SpaceSelectionPolicy:
    """
    空间选择策略

    基于任务嵌入选择最优空间类型
    """

    def __init__(self, space_library: List[str]):
        self.space_library = space_library
        self.n_spaces = len(space_library)

        # 简单的线性权重（实际可以用神经网络）
        self.weights = np.random.randn(self.embedding_dim, self.n_spaces) * 0.1

    @property
    def embedding_dim(self) -> int:
        return 8  # 与TaskEmbedding匹配

    def select(self, task_embedding: np.ndarray) -> Tuple[str, np.ndarray]:
        """
        选择空间

        Returns:
            (selected_space_name, selection_probabilities)
        """
        # 归一化嵌入
        x = task_embedding[:self.embedding_dim]

        # 计算每个空间的分数
        scores = x @ self.weights

        # softmax
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores)

        # 选择最高概率
        selected_idx = np.argmax(probs)

        return self.space_library[selected_idx], probs

    def update_weights(self,
                      task_embedding: np.ndarray,
                      selected_space: str,
                      reward: float,
                      learning_rate: float = 0.01):
        """
        根据奖励更新策略

        Args:
            task_embedding: 任务嵌入
            selected_space: 选择的空间
            reward: 奖励值（越高越好）
            learning_rate: 学习率
        """
        space_idx = self.space_library.index(selected_space)

        # 计算当前概率
        x = task_embedding[:self.embedding_dim]
        scores = x @ self.weights
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores)

        # 策略梯度更新
        for i in range(self.n_spaces):
            if i == space_idx:
                # 增加选中空间的权重
                self.weights[:, i] += learning_rate * reward * x * (1 - probs[i])
            else:
                # 减少其他空间的权重
                self.weights[:, i] -= learning_rate * reward * x * probs[i]


class MetaLearner:
    """
    元学习器

    学习如何为新任务选择最优认知空间
    """

    def __init__(self,
                 space_library: Optional[List[str]] = None,
                 embedding_dim: int = 16):
        # 默认空间库
        self.space_library = space_library or [
            "euclidean",
            "ricci",
            "conformal",
            "fisher",
            "finsler",
            "product",
            "temporal",
        ]

        self.embedding = TaskEmbedding(embedding_dim)
        self.policy = SpaceSelectionPolicy(self.space_library)

        # 历史记录
        self.task_history: List[Dict] = []
        self.space_performance: Dict[str, List[float]] = defaultdict(list)

        # 元训练状态
        self.meta_trained = False

    def select_space_for_task(self,
                             task_description: Dict[str, Any],
                             explore_prob: float = 0.2) -> Tuple[str, Dict]:
        """
        为新任务选择空间

        Args:
            task_description: 任务描述
            explore_prob: 探索概率（随机选择而非按策略）

        Returns:
            (selected_space, metadata)
        """
        # 编码任务
        task_emb = self.embedding.embed(task_description)

        # 选择
        if np.random.random() < explore_prob:
            # 探索：随机选择
            selected = np.random.choice(self.space_library)
            _, probs = self.policy.select(task_emb)
        else:
            # 利用：按策略选择
            selected, probs = self.policy.select(task_emb)

        metadata = {
            'task_embedding': task_emb,
            'selection_probs': probs,
            'exploration': np.random.random() < explore_prob
        }

        return selected, metadata

    def evaluate_space_on_task(self,
                              space: Any,
                              task_evaluator: Callable[[Any], float]) -> float:
        """
        评估空间在任务上的表现
        """
        try:
            return task_evaluator(space)
        except Exception as e:
            import warnings
            warnings.warn(f"Task evaluation failed: {e}")
            return float('-inf')

    def meta_train(self,
                  task_distribution: List[Dict[str, Any]],
                  task_evaluator_factory: Callable[[Dict], Callable],
                  n_episodes: int = 50,
                  n_adaptation_steps: int = 5):
        """
        元训练

        Args:
            task_distribution: 任务分布样本
            task_evaluator_factory: 根据任务描述创建评估器的工厂
            n_episodes: 元训练轮数
            n_adaptation_steps: 每任务适应步数
        """
        from ..core.registry import create_space

        print(f"Starting meta-training for {n_episodes} episodes...")

        for episode in range(n_episodes):
            # 采样任务
            task = np.random.choice(task_distribution)

            # 选择空间
            selected_space, metadata = self.select_space_for_task(task, explore_prob=0.3)

            # 创建评估器
            evaluator = task_evaluator_factory(task)

            # 评估
            total_reward = 0.0

            for step in range(n_adaptation_steps):
                try:
                    # 创建空间实例
                    space = create_space(selected_space,
                                        width=task.get('width', 40),
                                        height=task.get('height', 20))
                except Exception:
                    continue

                # 评估
                score = self.evaluate_space_on_task(space, evaluator)
                total_reward += score

                # 记录
                self.task_history.append({
                    'episode': episode,
                    'task': task,
                    'space': selected_space,
                    'step': step,
                    'score': score
                })

                self.space_performance[selected_space].append(score)

            # 更新策略
            avg_reward = total_reward / n_adaptation_steps if n_adaptation_steps > 0 else 0
            self.policy.update_weights(metadata['task_embedding'], selected_space, avg_reward)

            if episode % 10 == 0:
                print(f"  Episode {episode}: Avg reward = {avg_reward:.3f}")

        self.meta_trained = True
        print("Meta-training completed")

    def get_space_recommendation(self, task_description: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取空间推荐（带置信度）
        """
        task_emb = self.embedding.embed(task_description)
        selected, probs = self.policy.select(task_emb)

        # 构建推荐列表
        recommendations = []
        for i, space_name in enumerate(self.space_library):
            recommendations.append({
                'space': space_name,
                'probability': float(probs[i]),
                'avg_performance': np.mean(self.space_performance[space_name])
                               if space_name in self.space_performance else None
            })

        # 按概率排序
        recommendations.sort(key=lambda x: x['probability'], reverse=True)

        return {
            'recommended': selected,
            'confidence': float(max(probs)),
            'all_recommendations': recommendations[:3]
        }

    def save(self, filepath: str):
        """保存元学习器状态"""
        state = {
            'space_library': self.space_library,
            'policy_weights': self.policy.weights.tolist(),
            'space_performance': dict(self.space_performance),
            'meta_trained': self.meta_trained
        }
        with open(filepath, 'w') as f:
            json.dump(state, f)

    def load(self, filepath: str):
        """加载元学习器状态"""
        with open(filepath, 'r') as f:
            state = json.load(f)

        self.space_library = state['space_library']
        self.policy.weights = np.array(state['policy_weights'])
        self.space_performance = defaultdict(list, state.get('space_performance', {}))
        self.meta_trained = state.get('meta_trained', False)


class AdaptiveMetaLearner(MetaLearner):
    """
    自适应元学习器

    在线学习，根据实际使用反馈持续改进
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.online_buffer = []
        self.adaptation_rate = 0.01

    def report_performance(self,
                          task_description: Dict[str, Any],
                          selected_space: str,
                          performance: float):
        """
        报告实际使用中的表现

        用于在线学习
        """
        task_emb = self.embedding.embed(task_description)

        self.online_buffer.append({
            'embedding': task_emb,
            'space': selected_space,
            'performance': performance
        })

        # 立即更新
        self.policy.update_weights(task_emb, selected_space, performance, self.adaptation_rate)

        # 保持缓冲区大小
        if len(self.online_buffer) > 100:
            self.online_buffer.pop(0)

    def get_performance_history(self, space_name: Optional[str] = None) -> List[float]:
        """获取历史表现"""
        if space_name:
            return self.space_performance.get(space_name, [])

        # 返回所有平均
        all_scores = []
        for scores in self.space_performance.values():
            all_scores.extend(scores)
        return all_scores
