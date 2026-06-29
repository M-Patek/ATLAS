"""
Exploration: Ricci Curvature Attention
Ricci 曲率注意力驱动的探索

核心概念: 使用信息几何的 Ricci 曲率指导注意力分配
- 高曲率区域 = 高信息密度 = 优先探索
- 曲率 R ≈ -Δ log(uncertainty)

对应原: phase1_ricci_attention
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque

from ..core.space import CognitiveSpace, register_space
from ..core.registry import registry


class RicciExplorer:
    """
    Ricci 曲率驱动的探索器

    不规划长期路径，而是根据局部曲率梯度决定下一步
    """

    def __init__(self, width: int, height: int,
                 curiosity_weight: float = 1.0,
                 novelty_weight: float = 0.5):
        self.width = width
        self.height = height
        self.curiosity_weight = curiosity_weight
        self.novelty_weight = novelty_weight

        # 历史记录
        self.visit_count = np.zeros((width, height), dtype=np.int32)
        self.uncertainty_field = np.ones((width, height))

    def compute_curvature(self, x: int, y: int) -> float:
        """计算 (x,y) 处的 Ricci 曲率"""
        x = max(1, min(x, self.width - 2))
        y = max(1, min(y, self.height - 2))

        u = self.uncertainty_field[x, y]
        log_u = np.log(u + 1e-8)

        neighbors = [
            np.log(self.uncertainty_field[x+1, y] + 1e-8),
            np.log(self.uncertainty_field[x-1, y] + 1e-8),
            np.log(self.uncertainty_field[x, y+1] + 1e-8),
            np.log(self.uncertainty_field[x, y-1] + 1e-8),
        ]

        laplacian = sum(neighbors) - 4 * log_u
        return -laplacian

    def get_action(self, position: Tuple[int, int]) -> Tuple[int, int]:
        """
        根据曲率梯度选择移动方向

        Returns: (dx, dy) 移动方向
        """
        x, y = position

        # 计算4个方向的曲率变化
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        best_dir = (0, 0)
        best_score = -float('inf')

        c_center = self.compute_curvature(x, y)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                continue

            c_neighbor = self.compute_curvature(nx, ny)

            # 曲率增加 + 新颖性
            curvature_gain = c_neighbor - c_center
            novelty = 1.0 / (1.0 + 0.1 * self.visit_count[nx, ny])

            score = (self.curiosity_weight * curvature_gain +
                    self.novelty_weight * novelty)

            if score > best_score:
                best_score = score
                best_dir = (dx, dy)

        return best_dir

    def update(self, position: Tuple[int, int],
              observation: Optional[Dict] = None):
        """更新探索状态"""
        x, y = position
        self.visit_count[x, y] += 1

        # 降低已访问位置的 uncertainty
        self.uncertainty_field[x, y] *= 0.95

        # 根据观测更新
        if observation and 'uncertainty' in observation:
            for (ux, uy), uval in observation['uncertainty'].items():
                if 0 <= ux < self.width and 0 <= uy < self.height:
                    self.uncertainty_field[ux, uy] = max(
                        self.uncertainty_field[ux, uy], uval
                    )


@register_space("exploration_ricci")
class ExplorationRicciSpace(CognitiveSpace):
    """
    纯探索导向的 Ricci 空间

    专为探索任务优化，不考虑目标导向
    """

    def __init__(self, width: int, height: int,
                 curiosity_weight: float = 1.0,
                 **kwargs):
        super().__init__(width, height, name="exploration_ricci")

        self.explorer = RicciExplorer(width, height, curiosity_weight)

    def compute_distance(self, pos1: Tuple[int, int],
                        pos2: Tuple[int, int]) -> float:
        """探索距离：高曲率区域更近（鼓励探索）"""
        x1, y1 = pos1
        x2, y2 = pos2

        euclidean = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        n_samples = max(int(euclidean), 1)

        total = 0.0
        for i in range(n_samples):
            t = i / max(1, n_samples - 1)
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))

            # 曲率越高，距离越"短"（鼓励去高曲率区域）
            curvature = abs(self.explorer.compute_curvature(x, y))
            factor = 1.0 / (1.0 + curvature)  # 高曲率 = 低成本

            total += factor * (euclidean / n_samples)

        return total

    def get_heuristic(self, pos: Tuple[int, int],
                     goal: Tuple[int, int]) -> float:
        return self.compute_distance(pos, goal)

    def update_from_observation(self, position: Tuple[int, int],
                                observation: Dict) -> None:
        self.explorer.update(position, observation)

    def get_visualization_fields(self) -> Dict[str, np.ndarray]:
        return {
            'uncertainty': self.explorer.uncertainty_field.copy(),
            'visits': self.explorer.visit_count.copy(),
        }
