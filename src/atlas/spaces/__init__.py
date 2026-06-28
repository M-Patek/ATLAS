"""
ATLAS Spaces: Cognitive Space Implementations
可插拔认知空间实现

所有空间都通过 @register_space 装饰器自动注册到核心框架

可用空间:
- euclidean: 欧氏距离基线
- ricci: Ricci 曲率空间（信息几何）
- conformal: 共形空间（动态度量）
- fisher: Fisher 信息几何
- wasserstein: 最优传输/Wasserstein
- finsler: Finsler 非对称度量
- exploration_ricci: 纯探索导向 Ricci
- nav_conformal: 导航导向共形空间
- integrated: 完整集成空间
- euclidean3d: 3D欧氏空间
- ricci3d: 3D Ricci流空间
- conformal3d: 3D共形空间
"""

# 导入即注册
from . import (
    ricci,
    euclidean,
    conformal,
    fisher,
    wasserstein,
    finsler,
    composite,  # 复合空间
    temporal,   # 时序空间
    grid3d,     # 3D网格空间
)

# 导入功能模块中的空间
from ..exploration import ricci_attention
from ..navigation import conformal_metric
from ..integration import world_model_space

__all__ = []
