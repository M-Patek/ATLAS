"""
理论边界判定与空间切换机制

核心设计：不是权重混合，而是理论边界判定 + 理论切换

类比：
- 牛顿力学：低速场景 validity=0.95，高速场景 validity=0.1 → 失效
- 相对论：低速场景 validity=0.7，高速场景 validity=0.95 → 激活

使用方式：
    # 创建多个空间
    spaces = {
        'euclidean': EuclideanSpace(40, 20),
        'ricci': RicciSpace(40, 20),
        'conformal': ConformalSpace(40, 20),
    }

    # 创建理论边界空间
    boundary_space = TheoryBoundarySpace(
        width=40, height=20,
        spaces=spaces,
        validity_threshold=0.5
    )

    # 使用
    distance = boundary_space.compute_distance(pos1, pos2)
    # 自动选择当前最有效的空间
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

from ..core.space import CognitiveSpace, register_space


@register_space("theory_boundary")
class TheoryBoundarySpace(CognitiveSpace):
    """
    理论边界空间

    不是基于外部条件切换空间，而是让每个空间自己判断：
    "我的数学模型在这个场景下还适用吗？"

    核心机制：
    1. 维护多个候选空间
    2. 每个空间报告自己的 validity
    3. 选择 validity 最高的空间作为当前空间
    4. 当当前空间 validity < threshold 时，切换到其他空间

    类比：
    - 不是"当速度>0.1c时用相对论"
    - 而是"相对论自己说：我在这个速度下还准吗？"
    """

    def __init__(self, width: int, height: int,
                 spaces: Dict[str, CognitiveSpace] = None,
                 validity_threshold: float = 0.3,
                 **kwargs):
        super().__init__(width, height, name="theory_boundary")

        self.spaces = spaces or {}
        self.validity_threshold = validity_threshold

        # 当前激活的空间
        self.current_space_name: Optional[str] = None
        self.current_space: Optional[CognitiveSpace] = None

        # 历史记录
        self.validity_history: List[Dict[str, float]] = []
        self.switch_history: List[Dict[str, Any]] = []

        # 统计
        self.switch_count = 0
        self.space_usage = defaultdict(int)

    def add_space(self, name: str, space: CognitiveSpace):
        """添加候选空间"""
        self.spaces[name] = space

    def _evaluate_all_spaces(self, position: Tuple[int, int],
                             observation: Dict[str, Any]) -> Dict[str, float]:
        """
        评估所有空间的 validity

        Returns:
            {space_name: validity_score}
        """
        validities = {}

        for name, space in self.spaces.items():
            try:
                v = space.compute_validity(position, observation)
                validities[name] = v
            except Exception as e:
                # 如果空间不支持 validity，给一个默认值
                validities[name] = 0.5

        return validities

    def _select_best_space(self, position: Tuple[int, int],
                           observation: Dict[str, Any]) -> Tuple[str, CognitiveSpace, float]:
        """
        选择最佳空间

        策略：
        1. 评估所有空间的 validity
        2. 选择 validity 最高的空间
        3. 如果最高 validity < threshold，发出警告

        Returns:
            (space_name, space_instance, validity)
        """
        validities = self._evaluate_all_spaces(position, observation)

        # 记录历史
        self.validity_history.append(validities)

        # 选择最佳
        if not validities:
            return None, None, 0.0

        best_name = max(validities.keys(), key=lambda k: validities[k])
        best_validity = validities[best_name]
        best_space = self.spaces[best_name]

        # 检查是否需要切换
        if (self.current_space_name is not None and
            self.current_space_name != best_name):
            # 记录切换
            self.switch_history.append({
                'from': self.current_space_name,
                'to': best_name,
                'from_validity': validities.get(self.current_space_name, 0),
                'to_validity': best_validity,
                'position': position,
            })
            self.switch_count += 1

        # 更新当前空间
        self.current_space_name = best_name
        self.current_space = best_space
        self.space_usage[best_name] += 1

        return best_name, best_space, best_validity

    def compute_distance(self, pos1: Tuple[int, int],
                        pos2: Tuple[int, int]) -> float:
        """
        计算距离

        使用当前最佳空间计算距离
        """
        if self.current_space is None:
            # 默认使用欧氏
            from ..spaces.euclidean import euclidean_distance
            return euclidean_distance(pos1, pos2)

        return self.current_space.compute_distance(pos1, pos2)

    def get_heuristic(self, pos: Tuple[int, int],
                     goal: Tuple[int, int]) -> float:
        """使用当前最佳空间的启发式"""
        if self.current_space is None:
            from ..spaces.euclidean import euclidean_distance
            return euclidean_distance(pos, goal)

        return self.current_space.get_heuristic(pos, goal)

    def update_from_observation(self, position: Tuple[int, int],
                                observation: Dict[str, Any]) -> None:
        """
        更新所有空间，并选择最佳空间
        """
        # 更新所有子空间
        for space in self.spaces.values():
            try:
                space.update_from_observation(position, observation)
            except Exception:
                pass

        # 选择最佳空间
        self._select_best_space(position, observation)

    def get_visualization_fields(self) -> Dict[str, np.ndarray]:
        """返回当前空间的可视化字段"""
        if self.current_space:
            fields = self.current_space.get_visualization_fields()
            # 添加当前空间名称
            fields['_current_space'] = np.full(
                (self.width, self.height),
                hash(self.current_space_name) % 256,
                dtype=np.uint8
            )
            return fields

        return {'metric': np.ones((self.width, self.height))}

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = super().get_statistics()
        stats.update({
            'current_space': self.current_space_name,
            'switch_count': self.switch_count,
            'space_usage': dict(self.space_usage),
            'num_spaces': len(self.spaces),
        })

        # 如果有历史，添加平均validity
        if self.validity_history:
            latest = self.validity_history[-1]
            stats['latest_validities'] = latest

        return stats

    def get_validity_fields(self) -> Dict[str, np.ndarray]:
        """返回所有空间的 validity 场"""
        result = {}

        for name, space in self.spaces.items():
            try:
                fields = space.get_validity_fields()
                for key, value in fields.items():
                    result[f"{name}_{key}"] = value
            except Exception:
                pass

        return result

    def predict_next_state(self, position: Tuple[int, int],
                           observation: Dict[str, Any]) -> Dict[str, Any]:
        """使用当前最佳空间预测"""
        if self.current_space:
            return self.current_space.predict_next_state(position, observation)

        # 默认
        return {
            'predicted_position': position,
            'predicted_cost': 0.0,
            'predicted_uncertainty': 1.0,
            'passable': False,
        }
