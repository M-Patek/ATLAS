"""
Euclidean Space
欧氏空间基线

最简单的认知空间，作为性能基准
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional

from ..core.space import CognitiveSpace, register_space, euclidean_distance


@register_space("euclidean")
class EuclideanSpace(CognitiveSpace):
    """
    欧氏空间基线

    简单的欧氏距离度量，没有曲率或不确定性。
    用于作为其他空间的对照基准。
    """

    def __init__(self, width: int, height: int, **kwargs):
        super().__init__(width, height, name="euclidean")

    def compute_distance(self, pos1: Tuple[int, int],
                        pos2: Tuple[int, int]) -> float:
        """欧氏距离"""
        return euclidean_distance(pos1, pos2)

    def get_heuristic(self, pos: Tuple[int, int],
                     goal: Tuple[int, int]) -> float:
        """欧氏启发式（可接受）"""
        return euclidean_distance(pos, goal)

    def update_from_observation(self, position: Tuple[int, int],
                                observation: Dict[str, Any]) -> None:
        """欧氏空间不更新"""
        pass

    def get_visualization_fields(self) -> Dict[str, np.ndarray]:
        """返回单位场"""
        return {
            'metric': np.ones((self.width, self.height)),
        }

    def predict_next_state(self, position: Tuple[int, int],
                           observation: Dict[str, Any]) -> Dict[str, Any]:
        """欧氏空间的简单预测"""
        from ..core.space import neighbors_4

        goal = observation.get('goal_position')
        obstacles = set(observation.get('obstacles', []))

        best_pos = position
        best_dist = float('inf')

        for nx, ny in neighbors_4(position, self.width, self.height):
            if (nx, ny) in obstacles:
                continue
            if goal:
                dist = euclidean_distance((nx, ny), goal)
                if dist < best_dist:
                    best_dist = dist
                    best_pos = (nx, ny)

        return {
            'predicted_position': best_pos,
            'predicted_cost': best_dist,
            'predicted_uncertainty': 0.3,  # 欧氏空间不确定性较低
            'passable': best_pos != position,
        }

    def compute_validity(self, position: Tuple[int, int],
                        observation: Dict[str, Any],
                        actual: Optional[Dict[str, Any]] = None) -> float:
        """
        欧氏空间的适用性判定

        欧氏空间是最简单的基线，适用条件：
        - 平坦地形：完全有效
        - 无障碍：完全有效
        - 有障碍/曲率：有效性下降

        类比：牛顿力学在低速场景
        - 简单场景：validity = 1.0
        - 复杂场景：validity 下降（需要更复杂的理论）
        """
        # 检查是否有障碍物
        obstacles = observation.get('obstacles', [])
        if not obstacles:
            # 无障碍：欧氏完全有效
            return 0.9

        # 检查障碍物密度
        x, y = position
        nearby_obstacles = sum(1 for ox, oy in obstacles
                               if abs(ox - x) <= 3 and abs(oy - y) <= 3)

        if nearby_obstacles == 0:
            return 0.9
        elif nearby_obstacles <= 2:
            return 0.7
        elif nearby_obstacles <= 5:
            return 0.4
        else:
            return 0.2

    def get_validity_fields(self) -> Dict[str, np.ndarray]:
        """欧氏空间的有效性场"""
        # 欧氏空间在所有位置都相对有效（基线）
        return {
            'validity': np.ones((self.width, self.height)) * 0.8,
            'metric': np.ones((self.width, self.height)),
        }
