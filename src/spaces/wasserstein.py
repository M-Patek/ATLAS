"""
Wasserstein Space
Wasserstein (Optimal Transport) Space

基于最优传输的费用
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional

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

        # 缓存
        self._last_obstacles: set = set()
        self._cost_field_stale = True
        self._obstacle_cost = np.zeros((width, height))

    def compute_distance(self, pos1: Tuple[int, int],
                        pos2: Tuple[int, int]) -> float:
        """传输成本（向量化优化版）"""
        x1, y1 = pos1
        x2, y2 = pos2

        euclidean_dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if euclidean_dist < 0.5:
            return 0.0

        n_samples = max(int(euclidean_dist), 1)

        # 向量化路径采样
        t_values = np.linspace(0, 1, n_samples)
        x_coords = np.clip((x1 + t_values * (x2 - x1)).astype(int), 0, self.width - 1)
        y_coords = np.clip((y1 + t_values * (y2 - y1)).astype(int), 0, self.height - 1)

        # 批量获取成本
        costs = self.cost_field[x_coords, y_coords] * self.base_cost
        total_cost = np.sum(costs) * (euclidean_dist / n_samples)

        return float(total_cost)

    def get_heuristic(self, pos: Tuple[int, int],
                     goal: Tuple[int, int]) -> float:
        """启发式"""
        return self.compute_distance(pos, goal)

    def update_from_observation(self, position: Tuple[int, int],
                                observation: Dict[str, Any]) -> None:
        """更新成本场（优化版：缓存障碍物）"""
        x, y = position

        # 访问位置积累质量
        if 0 <= x < self.width and 0 <= y < self.height:
            self.mass[x, y] += 0.01

        # 障碍物增加成本（只在障碍物变化时更新）
        if 'obstacles' in observation:
            obstacles = set(observation['obstacles'])
            if obstacles != self._last_obstacles:
                self._last_obstacles = obstacles
                self._update_obstacle_cost(obstacles)
                self._cost_field_stale = True

        # 只在需要时更新成本场
        if self._cost_field_stale:
            self.cost_field = np.minimum(5.0,
                np.ones((self.width, self.height)) + self._obstacle_cost)
            self._cost_field_stale = False

        # 归一化质量
        total_mass = np.sum(self.mass)
        if total_mass > 0:
            self.mass /= total_mass

    def _update_obstacle_cost(self, obstacles: set):
        """更新障碍物成本（向量化版）"""
        self._obstacle_cost.fill(0.0)

        if not obstacles:
            return

        # 预计算 7x7 核
        dx_grid, dy_grid = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4), indexing='ij')
        dist_kernel = np.sqrt(dx_grid**2 + dy_grid**2)
        cost_kernel = 0.3 * np.exp(-dist_kernel / 2.0)

        # 向量化：批量处理障碍物
        for ox, oy in obstacles:
            x_start = max(0, ox - 3)
            x_end = min(self.width, ox + 4)
            y_start = max(0, oy - 3)
            y_end = min(self.height, oy + 4)

            kx_start = max(0, 3 - ox)
            kx_end = min(7, self.width - ox + 3)
            ky_start = max(0, 3 - oy)
            ky_end = min(7, self.height - oy + 3)

            self._obstacle_cost[x_start:x_end, y_start:y_end] += \
                cost_kernel[kx_start:kx_end, ky_start:ky_end]

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
