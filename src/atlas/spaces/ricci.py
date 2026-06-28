"""
Ricci Cognitive Space - 优化版

核心优化：
1. 向量化 _recalculate_curvature（从 O(n²) 到 O(n)）
2. 缓存障碍物影响，避免每步重复计算
3. 延迟更新：只在需要时重新计算曲率
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional

from ..core.space import CognitiveSpace, register_space


@register_space("ricci")
class RicciSpace(CognitiveSpace):
    """
    Ricci 认知空间（优化版）

    基于信息几何的 Ricci 曲率场定义认知距离。
    高曲率区域 = 高信息密度 = 更高的"认知成本"
    """

    def __init__(self, width: int, height: int,
                 curvature_scale: float = 1.0,
                 familiarity_decay: float = 0.1,
                 **kwargs):
        super().__init__(width, height, name="ricci")

        self.curvature_scale = curvature_scale
        self.familiarity_decay = familiarity_decay

        # 核心场
        self.uncertainty = np.ones((width, height)) * 0.5
        self.curvature = np.zeros((width, height))
        self.familiarity = np.zeros((width, height))
        self.visit_count = np.zeros((width, height), dtype=np.int32)

        # 目标吸引子
        self.attractor_position: Optional[Tuple[int, int]] = None
        self.attractor_strength: float = 0.0

        # 缓存
        self._last_obstacles: set = set()
        self._curvature_stale = True
        self._obstacle_uncertainty = np.zeros((width, height))

        self._recalculate_curvature()

    def _recalculate_curvature(self):
        """从 uncertainty 计算 Ricci 曲率（向量化版）"""
        # 使用卷积计算 Laplacian
        log_u = np.log(self.uncertainty + 1e-8)

        # 离散 Laplacian: 4*u - (u_left + u_right + u_up + u_down)
        laplacian = np.zeros_like(log_u)
        laplacian[1:-1, 1:-1] = (
            4 * log_u[1:-1, 1:-1]
            - log_u[2:, 1:-1]      # right
            - log_u[:-2, 1:-1]     # left
            - log_u[1:-1, 2:]      # down
            - log_u[1:-1, :-2]     # up
        )

        raw_curvature = -laplacian

        # 应用熟悉度衰减
        familiarity_factor = 1.0 - self.familiarity
        self.curvature = raw_curvature * familiarity_factor

        # 边界条件
        self.curvature[0, :] = self.curvature[1, :]
        self.curvature[-1, :] = self.curvature[-2, :]
        self.curvature[:, 0] = self.curvature[:, 1]
        self.curvature[:, -1] = self.curvature[:, -2]

        self._curvature_stale = False

    def compute_distance(self, pos1: Tuple[int, int],
                        pos2: Tuple[int, int]) -> float:
        """计算 Ricci 距离"""
        x1, y1 = pos1
        x2, y2 = pos2

        euclidean_dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if euclidean_dist < 0.5:
            return 0.0

        # 沿路径采样积分
        n_samples = max(int(euclidean_dist * 2), 5)
        total = 0.0

        for i in range(n_samples):
            t = i / max(1, n_samples - 1)
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))

            metric_factor = 1.0 + self.curvature_scale * abs(self.curvature[x, y])
            total += metric_factor * (euclidean_dist / n_samples)

        return total

    def get_heuristic(self, pos: Tuple[int, int],
                     goal: Tuple[int, int]) -> float:
        """启发式 = Ricci 距离估计"""
        return self.compute_distance(pos, goal)

    def update_from_observation(self, position: Tuple[int, int],
                                observation: Dict[str, Any]) -> None:
        """根据观测更新 uncertainty（优化版）"""
        x, y = position

        # 更新熟悉度
        if 0 <= x < self.width and 0 <= y < self.height:
            self.visit_count[x, y] += 1
            self.familiarity[x, y] = min(1.0,
                self.familiarity[x, y] + self.familiarity_decay)

        # 障碍物增加 uncertainty（缓存避免重复计算）
        if 'obstacles' in observation:
            obstacles = set(observation['obstacles'])
            if obstacles != self._last_obstacles:
                self._last_obstacles = obstacles
                self._update_obstacle_uncertainty(obstacles)
                self._curvature_stale = True

        # 目标降低 uncertainty
        if observation.get('goal_reached') and 'goal_position' in observation:
            gx, gy = observation['goal_position']
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        self.uncertainty[nx, ny] *= 0.7
            self._curvature_stale = True

        # 更新 attractor
        if 'goal_position' in observation:
            self.attractor_position = observation['goal_position']
            self.attractor_strength = observation.get('goal_strength', 0.5)

        # 只在需要时重新计算曲率
        if self._curvature_stale:
            self._recalculate_curvature()

    def _update_obstacle_uncertainty(self, obstacles: set):
        """更新障碍物影响的 uncertainty（完全向量化版）"""
        if not obstacles:
            return

        # 重置
        self._obstacle_uncertainty.fill(0.0)

        # 预计算 7x7 核
        dx_grid, dy_grid = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4), indexing='ij')
        dist_kernel = np.sqrt(dx_grid**2 + dy_grid**2)
        boost_kernel = 0.5 * np.exp(-dist_kernel / 2.0)

        # 向量化：批量处理障碍物
        for ox, oy in obstacles:
            # 计算影响区域的边界
            x_start = max(0, ox - 3)
            x_end = min(self.width, ox + 4)
            y_start = max(0, oy - 3)
            y_end = min(self.height, oy + 4)

            # 计算核的对应切片
            kx_start = max(0, 3 - ox)
            kx_end = min(7, self.width - ox + 3)
            ky_start = max(0, 3 - oy)
            ky_end = min(7, self.height - oy + 3)

            # 应用核
            self._obstacle_uncertainty[x_start:x_end, y_start:y_end] += \
                boost_kernel[kx_start:kx_end, ky_start:ky_end]

        # 应用累积的 uncertainty
        self.uncertainty = np.minimum(1.0,
            self.uncertainty + self._obstacle_uncertainty)

    def get_visualization_fields(self) -> Dict[str, np.ndarray]:
        """获取可视化数据"""
        return {
            'uncertainty': self.uncertainty.copy(),
            'curvature': self.curvature.copy(),
            'familiarity': self.familiarity.copy(),
            'metric_factor': 1.0 + self.curvature_scale * np.abs(self.curvature),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = super().get_statistics()
        stats.update({
            'mean_curvature': float(np.mean(np.abs(self.curvature))),
            'max_curvature': float(np.max(np.abs(self.curvature))),
            'mean_uncertainty': float(np.mean(self.uncertainty)),
            'total_visits': int(np.sum(self.visit_count)),
        })
        return stats

    def predict_next_state(self, position: Tuple[int, int],
                           observation: Dict[str, Any]) -> Dict[str, Any]:
        """基于 Ricci 场数据预测下一步"""
        from ..core.space import neighbors_4

        goal = observation.get('goal_position')
        obstacles = set(observation.get('obstacles', []))

        best_pos = position
        best_score = float('-inf')

        for nx, ny in neighbors_4(position, self.width, self.height):
            if (nx, ny) in obstacles:
                continue

            curvature_penalty = -abs(self.curvature[nx, ny]) * self.curvature_scale
            familiarity_bonus = self.familiarity[nx, ny]
            uncertainty_penalty = -self.uncertainty[nx, ny]

            goal_bonus = 0.0
            if goal:
                dist_to_goal = np.sqrt((nx - goal[0])**2 + (ny - goal[1])**2)
                goal_bonus = -dist_to_goal * 0.1

            score = curvature_penalty + familiarity_bonus + uncertainty_penalty + goal_bonus

            if score > best_score:
                best_score = score
                best_pos = (nx, ny)

        x, y = best_pos
        return {
            'predicted_position': best_pos,
            'predicted_cost': self.compute_distance(position, best_pos),
            'predicted_uncertainty': float(self.uncertainty[x, y]),
            'passable': best_pos != position,
            'ricci_score': float(best_score),
            'predicted_curvature': float(self.curvature[x, y]),
            'predicted_familiarity': float(self.familiarity[x, y]),
        }

    def compute_validity(self, position: Tuple[int, int],
                        observation: Dict[str, Any],
                        actual: Optional[Dict[str, Any]] = None) -> float:
        """Ricci空间的适用性判定"""
        x, y = position

        if not (0 <= x < self.width and 0 <= y < self.height):
            return 0.0

        current_curvature = abs(self.curvature[x, y])
        current_uncertainty = self.uncertainty[x, y]

        if current_curvature < 0.5:
            validity = 0.3 + 0.3 * (current_curvature / 0.5)
        elif current_curvature < 2.0:
            validity = 0.6 + 0.4 * ((current_curvature - 0.5) / 1.5)
        else:
            validity = 1.0 - 0.3 * min((current_curvature - 2.0) / 3.0, 1.0)

        uncertainty_boost = 0.2 * current_uncertainty
        return min(1.0, validity + uncertainty_boost)

    def get_validity_fields(self) -> Dict[str, np.ndarray]:
        """Ricci空间的有效性场"""
        validity = np.zeros((self.width, self.height))
        curv_abs = np.abs(self.curvature)

        # 向量化计算
        mask_low = curv_abs < 0.5
        mask_mid = (curv_abs >= 0.5) & (curv_abs < 2.0)
        mask_high = curv_abs >= 2.0

        validity[mask_low] = 0.3 + 0.3 * (curv_abs[mask_low] / 0.5)
        validity[mask_mid] = 0.6 + 0.4 * ((curv_abs[mask_mid] - 0.5) / 1.5)
        validity[mask_high] = 1.0 - 0.3 * np.minimum((curv_abs[mask_high] - 2.0) / 3.0, 1.0)

        return {
            'validity': validity,
            'curvature': self.curvature.copy(),
            'uncertainty': self.uncertainty.copy(),
        }
