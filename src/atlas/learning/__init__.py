"""
ATLAS Learning: Learning Integration Module
学习集成模块

支持:
- 贝叶斯参数优化
- 元学习自动空间选择
- 神经网络空间学习
"""

from .bayesian_optimizer import BayesianOptimizer, SpaceOptimizer
from .meta_learner import MetaLearner, TaskEmbedding, SpaceSelectionPolicy
from .neural_space import NeuralSpace, SpatialEncoder, MetricNetwork
from .trainer import SpaceTrainer, CurriculumScheduler, MetaTrainingEnvironment

__all__ = [
    # 贝叶斯优化
    "BayesianOptimizer",
    "SpaceOptimizer",

    # 元学习
    "MetaLearner",
    "TaskEmbedding",
    "SpaceSelectionPolicy",

    # 神经空间
    "NeuralSpace",
    "SpatialEncoder",
    "MetricNetwork",

    # 训练工具
    "SpaceTrainer",
    "CurriculumScheduler",
    "MetaTrainingEnvironment",
]
