"""
Euclidean Space
欧氏空间基线

最简单的认知空间，作为性能基准
"""

import numpy as np
from typing import Dict, Any, Tuple

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
