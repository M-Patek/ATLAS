"""
ATLAS Spaces: Composite Spaces
复合/组合空间

支持多种空间组合方式:
- ProductSpace: 并行组合多个空间
- HierarchicalSpace: 全局-局部层次结构
- MixedSpace: 基于上下文的动态切换
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Callable
from dataclasses import dataclass

from ..core.space import CognitiveSpace, register_space


@register_space("product")
class ProductSpace(CognitiveSpace):
    """
    乘积空间

    将多个空间并行组合，距离是各空间距离的加权组合:
    d² = Σ wᵢ × dᵢ²

    应用:
    - 同时考虑探索(Ricci)和导航(Conformal)
    - 融合几何距离和社会约束
    - 多目标平衡

    Example:
        space = ProductSpace(
            width=40, height=20,
            spaces=[("ricci", ricci_space, 0.7), ("conformal", conformal_space, 0.3)]
        )
    """

    def __init__(self, width: int, height: int,
                 spaces: List[Tuple[str, CognitiveSpace, float]],
                 composition: str = "euclidean",  # "euclidean" or "manhattan"
                 **kwargs):
        """
        Args:
            spaces: [(name, space_instance, weight), ...]
            composition: 组合方式 "euclidean" (d²=Σwᵢdᵢ²) 或 "manhattan" (d=Σwᵢdᵢ)
        """
        super().__init__(width, height, name="product")

        self.sub_spaces = spaces
        self.composition = composition

        # 归一化权重
        total_weight = sum(w for _, _, w in spaces)
        self.normalized_weights = [
            (name, space, w / total_weight)
            for name, space, w in spaces
        ]

    def compute_distance(self, pos1: Tuple[int, int],
                        pos2: Tuple[int, int]) -> float:
        """计算加权组合距离"""
        if self.composition == "euclidean":
            # d² = Σ wᵢ × dᵢ²
            total = 0.0
            for name, space, weight in self.normalized_weights:
                d = space.compute_distance(pos1, pos2)
                total += weight * (d ** 2)
            return np.sqrt(total)

        else:  # manhattan
            # d = Σ wᵢ × dᵢ
            total = 0.0
            for name, space, weight in self.normalized_weights:
                d = space.compute_distance(pos1, pos2)
                total += weight * d
            return total

    def get_heuristic(self, pos: Tuple[int, int],
                     goal: Tuple[int, int]) -> float:
        """启发式也是加权组合"""
        if self.composition == "euclidean":
            total = 0.0
            for name, space, weight in self.normalized_weights:
                h = space.get_heuristic(pos, goal)
                total += weight * (h ** 2)
            return np.sqrt(total)
        else:
            total = 0.0
            for name, space, weight in self.normalized_weights:
                h = space.get_heuristic(pos, goal)
                total += weight * h
            return total

    def update_from_observation(self, position: Tuple[int, int],
                                observation: Dict[str, Any]) -> None:
        """将更新传递给所有子空间"""
        for name, space, weight in self.normalized_weights:
            space.update_from_observation(position, observation)

    def get_visualization_fields(self) -> Dict[str, np.ndarray]:
        """合并所有子空间的字段"""
        result = {}
        for name, space, weight in self.normalized_weights:
            fields = space.get_visualization_fields()
            for key, value in fields.items():
                result[f"{name}_{key}"] = value
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """汇总所有子空间的统计"""
        stats = super().get_statistics()
        for name, space, weight in self.normalized_weights:
            sub_stats = space.get_statistics()
            for key, value in sub_stats.items():
                if isinstance(value, (int, float)):
                    stats[f"{name}_{key}"] = value
        return stats

    def adjust_weights(self, new_weights: Dict[str, float]):
        """
        动态调整权重

        Args:
            new_weights: {space_name: new_weight, ...}
        """
        # 更新权重
        updated = []
        for name, space, _ in self.normalized_weights:
            if name in new_weights:
                updated.append((name, space, new_weights[name]))
            else:
                updated.append((name, space, 0.0))  # 默认设为0

        # 重新归一化
        total = sum(w for _, _, w in updated)
        if total > 0:
            self.normalized_weights = [
                (name, space, w / total) for name, space, w in updated
            ]


@register_space("hierarchical")
class HierarchicalSpace(CognitiveSpace):
    """
    层次空间

    多尺度空间结构:
    - 全局层: 粗粒度，用于长距离规划
    - 局部层: 细粒度，用于短距离精确导航

    核心思想: 先在全局规划，然后在局部细化

    Example:
        space = HierarchicalSpace(
            width=100, height=100,
            global_space=CoarseRicciSpace(10, 10),  # 粗粒度
            local_space=DetailedConformalSpace(100, 100),  # 细粒度
            transition_threshold=15.0  # 距离<15用局部，>15用全局
        )
    """

    def __init__(self, width: int, height: int,
                 global_space: CognitiveSpace,
                 local_space: CognitiveSpace,
                 transition_threshold: float = 10.0,
                 **kwargs):
        super().__init__(width, height, name="hierarchical")

        self.global_space = global_space
        self.local_space = local_space
        self.transition_threshold = transition_threshold

        # 尺度映射
        self.global_scale = global_space.width / width
        self.local_scale = 1.0  # 局部空间应该和主空间同尺度

    def _to_global(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """映射到全局空间坐标"""
        x = int(pos[0] * self.global_scale)
        y = int(pos[1] * self.global_scale)
        return (x, y), (x, y)  # clamped

    def _from_global(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """从全局空间映射回来"""
        x = int(pos[0] / self.global_scale)
        y = int(pos[1] / self.global_scale)
        return (min(x, self.width - 1), min(y, self.height - 1))

    def compute_distance(self, pos1: Tuple[int, int],
                        pos2: Tuple[int, int]) -> float:
        """
        层次距离计算

        策略:
        - 短距离: 使用局部空间（精确）
        - 长距离: 使用全局空间（效率）
        """
        euclidean_dist = np.sqrt(
            (pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2
        )

        if euclidean_dist < self.transition_threshold:
            # 短距离：局部精确计算
            return self.local_space.compute_distance(pos1, pos2)
        else:
            # 长距离：全局粗粒度计算
            g1 = self._to_global(pos1)[0]
            g2 = self._to_global(pos2)[0]
            global_dist = self.global_space.compute_distance(g1, g2)
            # 映射回局部尺度
            return global_dist / self.global_scale

    def get_heuristic(self, pos: Tuple[int, int],
                     goal: Tuple[int, int]) -> float:
        """
        层次启发式

        始终用全局（可接受性保证 + 计算快）
        """
        g_pos = self._to_global(pos)[0]
        g_goal = self._to_global(goal)[0]
        return self.global_space.get_heuristic(g_pos, g_goal) / self.global_scale

    def update_from_observation(self, position: Tuple[int, int],
                                observation: Dict[str, Any]) -> None:
        """更新两个层次"""
        self.local_space.update_from_observation(position, observation)

        # 全局空间用粗粒度位置
        g_pos = self._to_global(position)[0]
        self.global_space.update_from_observation(g_pos, observation)

    def get_visualization_fields(self) -> Dict[str, np.ndarray]:
        """合并两个层次的字段"""
        result = {}

        local_fields = self.local_space.get_visualization_fields()
        for key, value in local_fields.items():
            result[f"local_{key}"] = value

        global_fields = self.global_space.get_visualization_fields()
        for key, value in global_fields.items():
            # 上采样到局部尺寸
            from scipy.ndimage import zoom
            zoom_factor = self.local_space.width / self.global_space.width
            upsampled = zoom(value, zoom_factor, order=1)
            result[f"global_{key}"] = upsampled[:self.width, :self.height]

        return result

    def plan_hierarchically(self, start: Tuple[int, int],
                           goal: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        层次规划

        Returns:
            (global_path, detailed_local_segments)
        """
        # 1. 全局路径
        g_start = self._to_global(start)[0]
        g_goal = self._to_global(goal)[0]

        # 使用全局空间规划（假设有求解器）
        from ..core.solver import GeodesicSolver
        global_solver = GeodesicSolver(self.global_space)
        global_result = global_solver.solve(g_start, g_goal)

        if not global_result.success:
            return [], []

        global_path = global_result.path

        # 2. 局部细化（关键段）
        local_segments = []

        for i in range(len(global_path) - 1):
            # 映射回局部坐标
            l_start = self._from_global(global_path[i])
            l_goal = self._from_global(global_path[i + 1])

            # 如果距离足够远，局部规划
            dist = np.sqrt(
                (l_start[0] - l_goal[0])**2 + (l_start[1] - l_goal[1])**2
            )

            if dist > 3:  # 局部规划阈值
                local_solver = GeodesicSolver(self.local_space)
                local_result = local_solver.solve(l_start, l_goal)
                if local_result.success:
                    local_segments.extend(local_result.path)

        return global_path, local_segments


@register_space("mixed")
class MixedSpace(CognitiveSpace):
    """
    混合空间

    根据上下文动态切换或混合多个空间

    Example:
        space = MixedSpace([
            (exploration_space, lambda ctx: ctx['uncertainty'] > 0.5),
            (navigation_space, lambda ctx: ctx['goal_visible']),
        ], default_space=baseline_space)
    """

    def __init__(self, width: int, height: int,
                 space_conditions: List[Tuple[CognitiveSpace, Callable]],
                 default_space: Optional[CognitiveSpace] = None,
                 blending: str = "hard",  # "hard" or "soft"
                 blend_window: int = 5,
                 **kwargs):
        super().__init__(width, height, name="mixed")

        self.space_conditions = space_conditions
        self.default_space = default_space or space_conditions[0][0]
        self.blending = blending
        self.blend_window = blend_window

        # 平滑切换用的历史
        self.space_history = []
        self.current_space_idx = 0

        # 缓存距离计算（用于soft blending）
        self._distance_cache = {}

    def _select_space(self, context: Optional[Dict] = None) -> CognitiveSpace:
        """根据上下文选择空间"""
        if context is None:
            context = {}

        # 按优先级测试条件
        for i, (space, condition) in enumerate(self.space_conditions):
            try:
                if condition(context):
                    self.space_history.append(i)
                    if len(self.space_history) > self.blend_window:
                        self.space_history.pop(0)
                    return space
            except Exception:
                continue

        return self.default_space

    def _get_active_space(self, pos1: Tuple[int, int],
                         pos2: Optional[Tuple[int, int]] = None) -> CognitiveSpace:
        """
        根据位置确定使用哪个空间

        可以通过子类覆盖实现基于位置的选择
        """
        # 默认使用最近的条件
        context = {
            'position': pos1,
            'target': pos2,
        }
        return self._select_space(context)

    def compute_distance(self, pos1: Tuple[int, int],
                        pos2: Tuple[int, int]) -> float:
        """
        动态距离计算

        hard: 直接使用选中空间
        soft: 混合多个符合条件的空间
        """
        if self.blending == "hard":
            space = self._get_active_space(pos1, pos2)
            return space.compute_distance(pos1, pos2)

        else:  # soft blending
            # 收集所有符合条件的距离
            distances = []
            weights = []

            context = {'position': pos1, 'target': pos2}

            for space, condition in self.space_conditions:
                try:
                    if condition(context):
                        d = space.compute_distance(pos1, pos2)
                        distances.append(d)
                        weights.append(1.0)
                except Exception:
                    continue

            if not distances:
                return self.default_space.compute_distance(pos1, pos2)

            # 加权平均
            total_weight = sum(weights)
            weighted_sum = sum(d * w for d, w in zip(distances, weights))
            return weighted_sum / total_weight

    def get_heuristic(self, pos: Tuple[int, int],
                     goal: Tuple[int, int]) -> float:
        """使用选中空间的启发式"""
        space = self._get_active_space(pos, goal)
        return space.get_heuristic(pos, goal)

    def update_from_observation(self, position: Tuple[int, int],
                                observation: Dict[str, Any]) -> None:
        """更新所有子空间"""
        for space, _ in self.space_conditions:
            space.update_from_observation(position, observation)

        if self.default_space:
            self.default_space.update_from_observation(position, observation)

        # 更新选择
        self._select_space(observation)

    def get_visualization_fields(self) -> Dict[str, np.ndarray]:
        """返回当前激活空间的字段"""
        # 返回所有空间的字段，用前缀区分
        result = {}
        for i, (space, _) in enumerate(self.space_conditions):
            fields = space.get_visualization_fields()
            for key, value in fields.items():
                result[f"space{i}_{key}"] = value
        return result

    def get_active_space_name(self) -> str:
        """获取当前激活空间的名称"""
        space = self._select_space()
        return space.name


# ============================================================================
# Utility Functions for Building Composite Spaces
# ============================================================================

def create_exploration_navigation_balance(
    width: int,
    height: int,
    exploration_weight: float = 0.5,
    ricci_params: Optional[Dict] = None,
    conformal_params: Optional[Dict] = None
) -> ProductSpace:
    """
    创建探索-导航平衡空间

    常用于：需要在探索新区域和快速到达目标之间平衡的场景
    """
    from ..core.registry import create_space

    ricci_params = ricci_params or {"curvature_scale": 1.5}
    conformal_params = conformal_params or {}

    ricci = create_space("ricci", width, height, **ricci_params)
    conformal = create_space("conformal", width, height, **conformal_params)

    return ProductSpace(width, height, [
        ("explore", ricci, exploration_weight),
        ("navigate", conformal, 1.0 - exploration_weight)
    ])


def create_adaptive_exploration_space(
    width: int,
    height: int,
    high_uncertainty_threshold: float = 0.7
) -> MixedSpace:
    """
    创建自适应探索空间

    - 高不确定性区域：使用 Ricci 空间（探索导向）
    - 低不确定性区域：使用 Euclidean 空间（效率导向）
    """
    from ..core.registry import create_space

    ricci = create_space("ricci", width, height, curvature_scale=2.0)
    euclidean = create_space("euclidean", width, height)

    return MixedSpace(width, height, [
        (ricci, lambda ctx: ctx.get('uncertainty', 0) > high_uncertainty_threshold),
        (euclidean, lambda ctx: True),  # 默认
    ])
