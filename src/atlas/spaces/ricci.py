"""
Ricci Cognitive Space
Ricci 认知空间

基于信息几何的 Ricci 曲率:
- 度量: g_ij = (1 + |R|)^2 δ_ij
- 曲率: R = -Δ log(uncertainty)
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional

from ..core.space import CognitiveSpace, register_space
from ..core.registry import registry


@register_space("ricci")
class RicciSpace(CognitiveSpace):
    """
    Ricci 认知空间

    基于信息几何的 Ricci 曲率场定义认知距离。
    高曲率区域 = 高信息密度 = 更高的"认知成本"

    Parameters:
        curvature_scale: 曲率影响因子（默认 1.0）
        familiarity_decay: 熟悉度衰减速率（默认 0.1）
    """

    def __init__(self, width: int, height: int,
                 curvature_scale: float = 1.0,
                 familiarity_decay: float = 0.1,
                 **kwargs):
        super().__init__(width, height, name="ricci")

        self.curvature_scale = curvature_scale
        self.familiarity_decay = familiarity_decay

        # 核心场
        self.uncertainty = np.ones((width, height)) * 0.5  # 初始为0.5，给boost空间
        self.curvature = np.zeros((width, height))
        self.familiarity = np.zeros((width, height))
        self.visit_count = np.zeros((width, height), dtype=np.int32)

        # 目标吸引子
        self.attractor_position: Optional[Tuple[int, int]] = None
        self.attractor_strength: float = 0.0

        self._recalculate_curvature()

    def _recalculate_curvature(self):
        """从 uncertainty 计算 Ricci 曲率"""
        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):
                u = self.uncertainty[x, y]
                log_u = np.log(u + 1e-8)

                # 离散 Laplacian
                neighbors = [
                    np.log(self.uncertainty[x+1, y] + 1e-8),
                    np.log(self.uncertainty[x-1, y] + 1e-8),
                    np.log(self.uncertainty[x, y+1] + 1e-8),
                    np.log(self.uncertainty[x, y-1] + 1e-8),
                ]

                laplacian = sum(neighbors) - 4 * log_u
                raw_curvature = -laplacian

                # 应用熟悉度衰减
                familiarity_factor = 1.0 - self.familiarity[x, y]
                self.curvature[x, y] = raw_curvature * familiarity_factor

        # 边界条件
        self.curvature[0, :] = self.curvature[1, :]
        self.curvature[-1, :] = self.curvature[-2, :]
        self.curvature[:, 0] = self.curvature[:, 1]
        self.curvature[:, -1] = self.curvature[:, -2]

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
        """启发式 = Ricci 距离估计（天然可接受）"""
        return self.compute_distance(pos, goal)

    def update_from_observation(self, position: Tuple[int, int],
                                observation: Dict[str, Any]) -> None:
        """根据观测更新 uncertainty"""
        x, y = position

        # 更新熟悉度
        if 0 <= x < self.width and 0 <= y < self.height:
            self.visit_count[x, y] += 1
            self.familiarity[x, y] = min(1.0,
                self.familiarity[x, y] + self.familiarity_decay)

        # 障碍物增加 uncertainty
        if 'obstacles' in observation:
            for ox, oy in observation['obstacles']:
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        nx, ny = ox + dx, oy + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            dist = np.sqrt(dx**2 + dy**2)
                            boost = 0.5 * np.exp(-dist / 2.0)
                            self.uncertainty[nx, ny] = min(1.0,
                                self.uncertainty[nx, ny] + boost)

        # 目标降低 uncertainty
        if observation.get('goal_reached') and 'goal_position' in observation:
            gx, gy = observation['goal_position']
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        self.uncertainty[nx, ny] *= 0.7

        # 更新 attractor
        if 'goal_position' in observation:
            self.attractor_position = observation['goal_position']
            self.attractor_strength = observation.get('goal_strength', 0.5)

        self._recalculate_curvature()

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
        """
        基于 Ricci 场数据预测下一步

        预测逻辑：
        - 高曲率区域 = 高信息密度 = 更"贵"的移动
        - 高不确定性区域 = 需要更多探索
        - 熟悉度高的区域 = 更"便宜"的移动
        """
        from ..core.space import neighbors_4

        goal = observation.get('goal_position')
        obstacles = set(observation.get('obstacles', []))

        best_pos = position
        best_score = float('-inf')

        for nx, ny in neighbors_4(position, self.width, self.height):
            if (nx, ny) in obstacles:
                continue

            # Ricci 特有的评分：低曲率 + 高熟悉度 = 好
            curvature_penalty = -abs(self.curvature[nx, ny]) * self.curvature_scale
            familiarity_bonus = self.familiarity[nx, ny]
            uncertainty_penalty = -self.uncertainty[nx, ny]

            # 目标导向
            goal_bonus = 0.0
            if goal:
                dist_to_goal = np.sqrt((nx - goal[0])**2 + (ny - goal[1])**2)
                goal_bonus = -dist_to_goal * 0.1  # 越近越好

            score = curvature_penalty + familiarity_bonus + uncertainty_penalty + goal_bonus

            if score > best_score:
                best_score = score
                best_pos = (nx, ny)

        # 预测不确定性基于曲率
        x, y = best_pos
        predicted_uncertainty = self.uncertainty[x, y]

        return {
            'predicted_position': best_pos,
            'predicted_cost': self.compute_distance(position, best_pos),
            'predicted_uncertainty': float(predicted_uncertainty),
            'passable': best_pos != position,
            'ricci_score': float(best_score),
            'predicted_curvature': float(self.curvature[x, y]),
            'predicted_familiarity': float(self.familiarity[x, y]),
        }
