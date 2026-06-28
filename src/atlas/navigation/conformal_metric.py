"""
Navigation: Dynamic Metric Navigation
动态度量导航

核心概念: 通过共形变换 g'_ij = Ω(x)² × g_ij 变形空间
- Ω < 1: 感觉更近（吸引子）
- Ω > 1: 感觉更远（排斥子）

对应原: phase2_dynamic_metric
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from ..core.space import CognitiveSpace, register_space


@dataclass
class Attractor:
    """吸引子：降低局部度量"""
    position: Tuple[int, int]
    strength: float = 0.5
    radius: float = 10.0

    def get_factor(self, x: int, y: int) -> float:
        """计算 (x,y) 处的共形因子"""
        dx = x - self.position[0]
        dy = y - self.position[1]
        dist = np.sqrt(dx**2 + dy**2)

        if dist > self.radius:
            return 1.0

        # 距离吸引子越近，Ω越小
        normalized_dist = dist / self.radius
        omega = 1.0 - self.strength * (1.0 - normalized_dist)
        return max(0.1, omega)


@dataclass
class Repeller:
    """排斥子：增加局部度量"""
    position: Tuple[int, int]
    strength: float = 0.3
    radius: float = 5.0

    def get_factor(self, x: int, y: int) -> float:
        """计算 (x,y) 处的共形因子"""
        dx = x - self.position[0]
        dy = y - self.position[1]
        dist = np.sqrt(dx**2 + dy**2)

        if dist > self.radius:
            return 1.0

        # 距离排斥子越近，Ω越大
        normalized_dist = dist / self.radius
        omega = 1.0 + self.strength * (1.0 - normalized_dist)
        return omega


@register_space("nav_conformal")
class ConformalNavSpace(CognitiveSpace):
    """
    共形导航空间

    通过 attractors 和 repellers 动态变形空间
    """

    def __init__(self, width: int, height: int,
                 base_conformal: float = 1.0,
                 **kwargs):
        super().__init__(width, height, name="nav_conformal")

        self.base_conformal = base_conformal

        # 共形因子场
        self.conformal_field = np.ones((width, height))

        # 吸引子和排斥子
        self.attractors: List[Attractor] = []
        self.repellers: List[Repeller] = []

    def add_attractor(self, position: Tuple[int, int],
                     strength: float = 0.5,
                     radius: float = 10.0):
        """添加吸引子（通常是目标）"""
        self.attractors.append(Attractor(position, strength, radius))
        self._recalculate_field()

    def add_repeller(self, position: Tuple[int, int],
                    strength: float = 0.3,
                    radius: float = 5.0):
        """添加排斥子（通常是障碍物）"""
        self.repellers.append(Repeller(position, strength, radius))
        self._recalculate_field()

    def _recalculate_field(self):
        """重新计算共形因子场"""
        self.conformal_field.fill(self.base_conformal)

        # 应用吸引子（降低因子）
        for attr in self.attractors:
            for x in range(self.width):
                for y in range(self.height):
                    factor = attr.get_factor(x, y)
                    self.conformal_field[x, y] *= factor

        # 应用排斥子（增加因子）
        for rep in self.repellers:
            for x in range(self.width):
                for y in range(self.height):
                    factor = rep.get_factor(x, y)
                    self.conformal_field[x, y] *= factor

    def compute_distance(self, pos1: Tuple[int, int],
                        pos2: Tuple[int, int]) -> float:
        """共形距离 = ∫ Ω(x) ds"""
        x1, y1 = pos1
        x2, y2 = pos2

        euclidean = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if euclidean < 0.5:
            return 0.0

        n_samples = max(int(euclidean), 1)
        total = 0.0

        for i in range(n_samples):
            t = i / max(1, n_samples - 1)
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))

            omega = self.conformal_field[x, y]
            total += omega * (euclidean / n_samples)

        return total

    def get_heuristic(self, pos: Tuple[int, int],
                     goal: Tuple[int, int]) -> float:
        return self.compute_distance(pos, goal)

    def update_from_observation(self, position: Tuple[int, int],
                                observation: Dict) -> None:
        """根据观测更新 attractors/repellers"""

        # 设置目标为吸引子
        if 'goal_position' in observation:
            self.attractors = []  # 清空旧目标
            self.add_attractor(
                observation['goal_position'],
                strength=observation.get('goal_strength', 0.6),
                radius=observation.get('goal_radius', 15.0)
            )

        # 将障碍物设为排斥子
        if 'obstacles' in observation:
            # 只添加最近的一些障碍物，避免过多
            import heapq

            obstacles = observation['obstacles']
            if len(obstacles) > 5:
                # 只保留最近的5个
                dists = [
                    (np.sqrt((ox-position[0])**2 + (oy-position[1])**2), (ox, oy))
                    for ox, oy in obstacles
                ]
                obstacles = [pos for _, pos in heapq.nsmallest(5, dists)]

            for ox, oy in obstacles:
                self.add_repeller((ox, oy), strength=0.4, radius=6.0)

    def get_visualization_fields(self) -> Dict[str, np.ndarray]:
        return {
            'conformal_factor': self.conformal_field.copy(),
            'metric': self.conformal_field.copy(),
        }
