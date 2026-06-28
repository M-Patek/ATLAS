"""
Wasserstein Space
Wasserstein (Optimal Transport) Space

基于最优传输的费用
"""

import numpy as np
from typing import Dict, Any, Tuple

from ..core.space import CognitiveSpace, register_space


@register_space("wasserstein")
class WassersteinSpace(CognitiveSpace):
    """
    Wasserstein 认知空间

    基于传输费用的度量:
    移动质量需要付出代价

    简化版: 成本场 × 距离
    """

    def __init__(self, width: int, height: int,
                 base_cost: float = 1.0,
                 **kwargs):
        super().__init__(width, height, name="wasserstein")

        self.base_cost = base_cost

        # 成本场（地形难度）
        self.cost_field = np.ones((width, height))

        # 质量分布
        self.mass = np.ones((width, height)) / (width * height)

    def compute_distance(self, pos1: Tuple[int, int],
                        pos2: Tuple[int, int]) -> float:
        """传输成本"""
        x1, y1 = pos1
        x2, y2 = pos2

        euclidean_dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        n_samples = max(int(euclidean_dist), 1)

        total_cost = 0.0
        for i in range(n_samples):
            t = i / max(1, n_samples - 1)
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))

            # 成本 × 距离
            cost = self.cost_field[x, y] * self.base_cost
            total_cost += cost * (euclidean_dist / n_samples)

        return total_cost

    def get_heuristic(self, pos: Tuple[int, int],
                     goal: Tuple[int, int]) -> float:
        """启发式"""
        return self.compute_distance(pos, goal)

    def update_from_observation(self, position: Tuple[int, int],
                                observation: Dict[str, Any]) -> None:
        """更新成本场"""
        x, y = position

        # 访问位置积累质量
        if 0 <= x < self.width and 0 <= y < self.height:
            self.mass[x, y] += 0.01

        # 障碍物增加成本
        if 'obstacles' in observation:
            for ox, oy in observation['obstacles']:
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        nx, ny = ox + dx, oy + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            dist = np.sqrt(dx**2 + dy**2)
                            self.cost_field[nx, ny] += 0.3 * np.exp(-dist / 2.0)

        # 归一化质量
        total_mass = np.sum(self.mass)
        if total_mass > 0:
            self.mass /= total_mass

    def get_visualization_fields(self) -> Dict[str, np.ndarray]:
        """可视化数据"""
        return {
            'cost_field': self.cost_field.copy(),
            'mass_distribution': self.mass.copy(),
        }

    def predict_next_state(self, position: Tuple[int, int],
                           observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于 Wasserstein 场数据预测下一步

        预测逻辑：
        - 低成本区域 = 更"便宜"的移动
        - 高质量区域 = 更"稳定"的移动
        """
        from ..core.space import neighbors_4

        goal = observation.get('goal_position')
        obstacles = set(observation.get('obstacles', []))

        best_pos = position
        best_score = float('-inf')

        for nx, ny in neighbors_4(position, self.width, self.height):
            if (nx, ny) in obstacles:
                continue

            # Wasserstein 特有的评分：低成本 + 高质量 = 好
            cost_penalty = -self.cost_field[nx, ny]
            mass_bonus = self.mass[nx, ny]

            # 目标导向
            goal_bonus = 0.0
            if goal:
                dist_to_goal = np.sqrt((nx - goal[0])**2 + (ny - goal[1])**2)
                goal_bonus = -dist_to_goal * 0.1

            score = cost_penalty + mass_bonus + goal_bonus

            if score > best_score:
                best_score = score
                best_pos = (nx, ny)

        x, y = best_pos
        return {
            'predicted_position': best_pos,
            'predicted_cost': self.compute_distance(position, best_pos),
            'predicted_uncertainty': self.cost_field[x, y] / max(1.0, np.max(self.cost_field)),
            'passable': best_pos != position,
            'wasserstein_score': float(best_score),
            'predicted_cost_field': float(self.cost_field[x, y]),
            'predicted_mass': float(self.mass[x, y]),
        }
