"""
ATLAS Core: Pluggable Cognitive Architecture Framework
可插拔认知架构核心框架

核心设计原则:
1. 空间(Space)是第一-class抽象 - CognitiveSpace
2. 世界模型(WorldModel)更新空间 - WorldModel
3. 求解器(Solver)在空间中求解 - GeodesicSolver
4. 实验框架支持对比测试 - Experiment, AblationStudy

快速开始:
    from src.core import Experiment, GeodesicSolver
    from src.core.registry import create_space

    # 创建空间
    space = create_space("ricci", width=40, height=20, curvature_scale=2.0)

    # 创建实验
    experiment = Experiment("my_experiment")
    experiment.register_space("ricci", space)

    # 运行对比
    results = experiment.run(num_trials=10)
"""

from .space import CognitiveSpace, SpaceMetrics, register_space
from .world_model import WorldModel, SimpleWorldModel, UncertaintyWorldModel
from .solver import GeodesicSolver, DijkstraSolver, GreedySolver, SolverResult
from .replanning import DStarLiteSolver, AdaptiveNavigator
from .experiment import Experiment, AblationStudy, TrialResult, ConditionResult
from .ssfr_enhanced import (
    SSFREnhanced,
    StructurePool,
    StructureHypothesis,
    MultiSpaceRepresentation,
    ValidationResult,
)
from . import registry

__all__ = [
    # 核心抽象
    "CognitiveSpace",
    "SpaceMetrics",
    "WorldModel",
    "GeodesicSolver",
    "SolverResult",

    # 实验框架
    "Experiment",
    "AblationStudy",
    "TrialResult",
    "ConditionResult",

    # 动态规划
    "DStarLiteSolver",
    "AdaptiveNavigator",

    # 具体实现
    "SimpleWorldModel",
    "UncertaintyWorldModel",
    "DijkstraSolver",
    "GreedySolver",

    # 注册表
    "registry",
    "register_space",

    # 增强版 SSFR
    "SSFREnhanced",
    "StructurePool",
    "StructureHypothesis",
    "MultiSpaceRepresentation",
    "ValidationResult",
]
