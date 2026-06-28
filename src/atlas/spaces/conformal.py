"""
Conformal Space - 优化版

核心优化：
1. 向量化 _recalculate_conformal（从 O(n²) 到 O(n)）
2. 限制 attractors/repellers 数量
3. 使用距离场缓存
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional

from ..core.space import CognitiveSpace, register_space


@register_space("conformal")
class ConformalSpace(CognitiveSpace):
    """
    共形认知空间（优化版）

    通过共形因子 Ω(x) 变形空间:
    ds² = Ω(x)² (dx² + dy²)
    """

    def __init__(self, width: int, height: int,
                 base_scale: float = 1.0,
                 max_attractors: int = 10,
                 max_repellers: int = 10,
                 **kwargs):
        super().__init__(width, height, name="conformal")

        self.base_scale = base_scale
        self.max_attractors = max_attractors
        self.max_repellers = max_repellers

        # 共形因子场
        self.conformal_factor = np.ones((width, height))

        # Attractors/Repellers
        self.attractors: list = []
        self.repellers: list = []

        # 预计算坐标网格（避免重复创建）
        self._x_grid, self._y_grid = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')

    def add_attractor(self, position: Tuple[int, int],
                     strength: float = 0.5,
                     radius: float = 10.0):
        """添加吸引子（限制数量）"""
        self.attractors.append({
            'position': position,
            'strength': strength,
            'radius': radius
        })
        # 限制数量，移除最旧的
        if len(self.attractors) > self.max_attractors:
            self.attractors.pop(0)
        self._recalculate_conformal()

    def add_repeller(self, position: Tuple[int, int],
                    strength: float = 0.5,
                    radius: float = 10.0):
        """添加排斥子（限制数量）"""
        self.repellers.append({
            'position': position,
            'strength': strength,
            'radius': radius
        })
        if len(self.repellers) > self.max_repellers:
            self.repellers.pop(0)
        self._recalculate_conformal()

    def _recalculate_conformal(self):
        """重新计算共形因子（向量化版）"""
        self.conformal_factor.fill(1.0)

        # 应用 attractors（向量化）
        for attr in self.attractors:
            cx, cy = attr['position']
            dist = np.sqrt((self._x_grid - cx)**2 + (self._y_grid - cy)**2)
            mask = dist < attr['radius']
            if np.any(mask):
                factor = 1.0 - attr['strength'] * (1.0 - dist[mask] / attr['radius'])
                self.conformal_factor[mask] *= np.maximum(0.1, factor)

        # 应用 repellers（向量化）
        for rep in self.repellers:
            cx, cy = rep['position']
            dist = np.sqrt((self._x_grid - cx)**2 + (self._y_grid - cy)**2)
            mask = dist < rep['radius']
            if np.any(mask):
                factor = 1.0 + rep['strength'] * (1.0 - dist[mask] / rep['radius'])
                self.conformal_factor[mask] *= factor

    def compute_distance(self, pos1: Tuple[int, int],
                        pos2: Tuple[int, int]) -> float:
        """共形距离"""
        x1, y1 = pos1
        x2, y2 = pos2

        euclidean_dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if euclidean_dist < 0.5:
            return 0.0

        # 采样积分
        n_samples = max(int(euclidean_dist), 1)
        total = 0.0

        for i in range(n_samples):
            t = i / max(1, n_samples - 1)
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))

            omega = self.conformal_factor[x, y]
            total += omega * (euclidean_dist / n_samples)

        return total

    def get_heuristic(self, pos: Tuple[int, int],
                     goal: Tuple[int, int]) -> float:
        """启发式"""
        return self.compute_distance(pos, goal)

    def update_from_observation(self, position: Tuple[int, int],
                                observation: Dict[str, Any]) -> None:
        """根据观测更新 attractors（优化版）"""
        # 添加新的 attractor
        if 'goal_position' in observation:
            strength = observation.get('attractor_strength', 0.5)
            self.add_attractor(observation['goal_position'], strength)

        # 将障碍物转换为 repellers（限制数量）
        if 'obstacles' in observation:
            obstacles = list(observation['obstacles'])[:5]  # 只取前5个
            for ox, oy in obstacles:
                self.add_repeller((ox, oy), strength=0.3, radius=5.0)

    def get_visualization_fields(self) -> Dict[str, np.ndarray]:
        """可视化数据"""
        return {
            'conformal_factor': self.conformal_factor.copy(),
            'metric': self.conformal_factor.copy(),
        }

    def predict_next_state(self, position: Tuple[int, int],
                           observation: Dict[str, Any]) -> Dict[str, Any]:
        """基于 Conformal 场数据预测下一步"""
        from ..core.space import neighbors_4

        goal = observation.get('goal_position')
        obstacles = set(observation.get('obstacles', []))

        best_pos = position
        best_score = float('-inf')

        for nx, ny in neighbors_4(position, self.width, self.height):
            if (nx, ny) in obstacles:
                continue

            conformal_penalty = -self.conformal_factor[nx, ny]

            goal_bonus = 0.0
            if goal:
                dist_to_goal = np.sqrt((nx - goal[0])**2 + (ny - goal[1])**2)
                goal_bonus = -dist_to_goal * 0.3

            score = conformal_penalty + goal_bonus

            if score > best_score:
                best_score = score
                best_pos = (nx, ny)

        x, y = best_pos
        return {
            'predicted_position': best_pos,
            'predicted_cost': self.compute_distance(position, best_pos),
            'predicted_uncertainty': self.conformal_factor[x, y] - 1.0,
            'passable': best_pos != position,
            'conformal_score': float(best_score),
            'predicted_conformal_factor': float(self.conformal_factor[x, y]),
        }

    def compute_validity(self, position: Tuple[int, int],
                        observation: Dict[str, Any],
                        actual: Optional[Dict[str, Any]] = None) -> float:
        """Conformal空间的适用性判定"""
        x, y = position

        if not (0 <= x < self.width and 0 <= y < self.height):
            return 0.0

        goal = observation.get('goal_position')
        if goal is None:
            return 0.3

        num_attractors = len(self.attractors)
        num_repellers = len(self.repellers)

        if num_attractors == 0 and num_repellers == 0:
            return 0.2

        omega = self.conformal_factor[x, y]
        deviation = abs(omega - 1.0)

        if deviation > 0.5:
            validity = 0.7 + 0.3 * min(deviation / 2.0, 1.0)
        elif deviation > 0.1:
            validity = 0.4 + 0.3 * ((deviation - 0.1) / 0.4)
        else:
            validity = 0.3

        return min(1.0, validity)

    def get_validity_fields(self) -> Dict[str, np.ndarray]:
        """Conformal空间的有效性场（向量化）"""
        omega = self.conformal_factor
        deviation = np.abs(omega - 1.0)

        validity = np.zeros_like(omega)
        mask_high = deviation > 0.5
        mask_mid = (deviation > 0.1) & ~mask_high
        mask_low = ~mask_high & ~mask_mid

        validity[mask_high] = 0.7 + 0.3 * np.minimum(deviation[mask_high] / 2.0, 1.0)
        validity[mask_mid] = 0.4 + 0.3 * ((deviation[mask_mid] - 0.1) / 0.4)
        validity[mask_low] = 0.3

        return {
            'validity': validity,
            'conformal_factor': self.conformal_factor.copy(),
        }
