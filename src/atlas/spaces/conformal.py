"""
Conformal Space
共形空间

动态度量实现 - 通过共形因子变形空间
目标导向导航的核心空间类型
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional

from ..core.space import CognitiveSpace, register_space


@register_space("conformal")
class ConformalSpace(CognitiveSpace):
    """
    共形认知空间

    通过共形因子 Ω(x) 变形空间:
    ds² = Ω(x)² (dx² + dy²)

    Ω(x) 由 attractors 和 repellers 决定
    """

    def __init__(self, width: int, height: int,
                 base_scale: float = 1.0,
                 **kwargs):
        super().__init__(width, height, name="conformal")

        self.base_scale = base_scale

        # 共形因子场
        self.conformal_factor = np.ones((width, height))

        # Attractors（降低度量，感觉更近）
        self.attractors: list = []

        # Repellers（增加度量，感觉更远）
        self.repellers: list = []

    def add_attractor(self, position: Tuple[int, int],
                     strength: float = 0.5,
                     radius: float = 10.0):
        """添加吸引子"""
        self.attractors.append({
            'position': position,
            'strength': strength,
            'radius': radius
        })
        self._recalculate_conformal()

    def add_repeller(self, position: Tuple[int, int],
                    strength: float = 0.5,
                    radius: float = 10.0):
        """添加排斥子"""
        self.repellers.append({
            'position': position,
            'strength': strength,
            'radius': radius
        })
        self._recalculate_conformal()

    def _recalculate_conformal(self):
        """重新计算共形因子"""
        self.conformal_factor.fill(1.0)

        # 应用 attractors
        for attr in self.attractors:
            cx, cy = attr['position']
            for x in range(self.width):
                for y in range(self.height):
                    dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                    if dist < attr['radius']:
                        factor = 1.0 - attr['strength'] * (1.0 - dist/attr['radius'])
                        self.conformal_factor[x, y] *= max(0.1, factor)

        # 应用 repellers
        for rep in self.repellers:
            cx, cy = rep['position']
            for x in range(self.width):
                for y in range(self.height):
                    dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                    if dist < rep['radius']:
                        factor = 1.0 + rep['strength'] * (1.0 - dist/rep['radius'])
                        self.conformal_factor[x, y] *= factor

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
        """根据观测更新 attractors"""
        # 添加新的 attractor
        if 'goal_position' in observation:
            strength = observation.get('attractor_strength', 0.5)
            self.add_attractor(observation['goal_position'], strength)

        # 将障碍物转换为 repellers
        if 'obstacles' in observation:
            for ox, oy in observation['obstacles']:
                # 简化为只添加第一个障碍物的 repeller
                self.add_repeller((ox, oy), strength=0.3, radius=5.0)
                break  # 避免过多 repellers

    def get_visualization_fields(self) -> Dict[str, np.ndarray]:
        """可视化数据"""
        return {
            'conformal_factor': self.conformal_factor.copy(),
            'metric': self.conformal_factor.copy(),
        }

    def predict_next_state(self, position: Tuple[int, int],
                           observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于 Conformal 场数据预测下一步

        预测逻辑：
        - 低共形因子 = 吸引子附近 = 更"便宜"
        - 高共形因子 = 排斥子附近 = 更"贵"
        """
        from ..core.space import neighbors_4

        goal = observation.get('goal_position')
        obstacles = set(observation.get('obstacles', []))

        best_pos = position
        best_score = float('-inf')

        for nx, ny in neighbors_4(position, self.width, self.height):
            if (nx, ny) in obstacles:
                continue

            # Conformal 特有的评分：低共形因子 = 好
            conformal_penalty = -self.conformal_factor[nx, ny]

            # 目标导向（Conformal 对目标最敏感）
            goal_bonus = 0.0
            if goal:
                dist_to_goal = np.sqrt((nx - goal[0])**2 + (ny - goal[1])**2)
                # Conformal 空间强烈偏好目标方向
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
