"""
ATLAS: Atlas of Technologies for Learning Autonomous Systems

核心架构: 可插拔认知空间框架
"""

__version__ = "0.2.0"

# 核心框架 - 可插拔架构基础
from . import core

# 空间实现 - 各种认知空间
from . import spaces

# 功能模块
from . import exploration
from . import navigation
from . import integration

# 可视化（可选）
try:
    from . import visualization
except ImportError:
    visualization = None  # matplotlib 未安装

# 学习模块（可选）
try:
    from . import learning
except ImportError:
    learning = None  # 依赖未安装

__all__ = [
    "core",
    "spaces",
    "exploration",
    "navigation",
    "integration",
    "visualization",
    "learning",
]
