"""
Integration: World Model + Space Separation
世界模型与空间分离架构

核心概念: 严格分离三层
1. World Model: 更新空间状态
2. Space: 被动描述认知几何
3. Solver: 在空间中求解

对应原: phase5_pure_ricci_separation (最成熟的架构)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

from ..core.space import CognitiveSpace, register_space
from ..core.solver import GeodesicSolver


@register_space("integrated")
class IntegratedSpace(CognitiveSpace):
    """
    集成的认知空间

    World Model 更新逻辑内置，提供完整的三层架构体验
    """

    def __init__(self, width: int, height: int,
                 curvature_scale: float = 1.0,
                 obstacle_influence: float = 0.5,
                 familiarity_decay: float = 0.1,
                 **kwargs):
        super().__init__(width, height, name="integrated")

        self.curvature_scale = curvature_scale
        self.obstacle_influence = obstacle_influence
        self.familiarity_decay = familiarity_decay

        # 核心场
        self.uncertainty = np.ones((width, height)) * 0.5
        self.curvature = np.zeros((width, height))
        self.familiarity = np.zeros((width, height))
        self.visit_count = np.zeros((width, height), dtype=np.int32)

        # 目标
        self.goal_position: Optional[Tuple[int, int]] = None

        self._recalculate_curvature()

    def _recalculate_curvature(self):
        """从 uncertainty 计算 Ricci 曲率"""
        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):
                u = self.uncertainty[x, y]
                log_u = np.log(u + 1e-8)

                neighbors = [
                    np.log(self.uncertainty[x+1, y] + 1e-8),
                    np.log(self.uncertainty[x-1, y] + 1e-8),
                    np.log(self.uncertainty[x, y+1] + 1e-8),
                    np.log(self.uncertainty[x, y-1] + 1e-8),
                ]

                laplacian = sum(neighbors) - 4 * log_u
                raw_curvature = -laplacian

                # 熟悉度衰减
                fam_factor = 1.0 - self.familiarity[x, y]
                self.curvature[x, y] = raw_curvature * fam_factor

        # 边界
        self.curvature[0, :] = self.curvature[1, :]
        self.curvature[-1, :] = self.curvature[-2, :]
        self.curvature[:, 0] = self.curvature[:, 1]
        self.curvature[:, -1] = self.curvature[:, -2]

    def compute_distance(self, pos1: Tuple[int, int],
                        pos2: Tuple[int, int]) -> float:
        """Integrated 距离: 曲率 + 熟悉度 + 目标影响"""
        x1, y1 = pos1
        x2, y2 = pos2

        euclidean = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if euclidean < 0.5:
            return 0.0

        n_samples = max(int(euclidean * 2), 5)
        total = 0.0

        for i in range(n_samples):
            t = i / max(1, n_samples - 1)
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))

            # 曲率因子
            curvature_factor = 1.0 + self.curvature_scale * abs(self.curvature[x, y])

            # 目标吸引因子（可选）
            goal_factor = 1.0
            if self.goal_position:
                gx, gy = self.goal_position
                dist_to_goal = np.sqrt((x-gx)**2 + (y-gy)**2)
                goal_factor = 0.5 + 0.5 * (dist_to_goal / max(self.width, self.height))

            total += curvature_factor * goal_factor * (euclidean / n_samples)

        return total

    def get_heuristic(self, pos: Tuple[int, int],
                     goal: Tuple[int, int]) -> float:
        return self.compute_distance(pos, goal)

    def update_from_observation(self, position: Tuple[int, int],
                                observation: Dict) -> None:
        """World Model 更新逻辑"""
        x, y = position

        # 更新熟悉度
        if 0 <= x < self.width and 0 <= y < self.height:
            self.visit_count[x, y] += 1
            self.familiarity[x, y] = min(1.0,
                self.familiarity[x, y] + self.familiarity_decay)

        # 障碍物增加 uncertainty
        if 'obstacles' in observation:
            for ox, oy in observation['obstacles']:
                for dx in range(-4, 5):
                    for dy in range(-4, 5):
                        nx, ny = ox + dx, oy + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            dist = np.sqrt(dx**2 + dy**2)
                            boost = self.obstacle_influence * np.exp(-dist / 2.0)
                            self.uncertainty[nx, ny] = min(1.0,
                                self.uncertainty[nx, ny] + boost)

        # 目标降低 uncertainty
        if observation.get('goal_reached'):
            if self.goal_position:
                gx, gy = self.goal_position
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            self.uncertainty[nx, ny] *= 0.7

        # 更新目标引用
        if 'goal_position' in observation:
            self.goal_position = observation['goal_position']

        self._recalculate_curvature()

    def get_visualization_fields(self) -> Dict[str, np.ndarray]:
        return {
            'uncertainty': self.uncertainty.copy(),
            'curvature': self.curvature.copy(),
            'familiarity': self.familiarity.copy(),
            'metric': 1.0 + self.curvature_scale * np.abs(self.curvature),
        }


class IntegratedSystem:
    """
    集成系统: World Model + Space + Solver 三层

    使用示例:
        system = IntegratedSystem(width=40, height=20)
        system.set_goal((35, 10))

        while not reached:
            system.perceive(position, observation)
            action = system.decide_action(position, obstacles)
    """

    def __init__(self, width: int = 50, height: int = 50,
                 curvature_scale: float = 1.0):
        self.space = IntegratedSpace(width, height, curvature_scale)
        self.solver = GeodesicSolver(self.space)

        self.current_path: Optional[List[Tuple[int, int]]] = None
        self.path_index = 0

    def set_goal(self, goal: Tuple[int, int]):
        """设置目标"""
        self.space.goal_position = goal
        self.space.update_from_observation((0, 0), {'goal_position': goal})

    def perceive(self, position: Tuple[int, int], observation: Dict):
        """感知更新"""
        self.space.update_from_observation(position, observation)

        # 如果空间发生重大变化，清除路径
        if self._should_replan():
            self.current_path = None

    def decide_action(self, current_pos: Tuple[int, int],
                     obstacles: Set[Tuple[int, int]]) -> Tuple[int, int]:
        """
        决策下一步动作

        Returns: (dx, dy) 方向向量
        """
        if self.space.goal_position is None:
            return (0, 0)

        # 检查是否需要重新规划
        if (self.current_path is None or
            self.path_index >= len(self.current_path) or
            not self._is_path_valid(self.current_path[self.path_index:], obstacles)):

            self.current_path = self.solver.solve(
                current_pos,
                self.space.goal_position,
                obstacles
            ).path
            self.path_index = 0

        # 执行路径
        if self.current_path and self.path_index < len(self.current_path):
            target = self.current_path[self.path_index]

            if current_pos == target:
                self.path_index += 1
                if self.path_index < len(self.current_path):
                    target = self.current_path[self.path_index]
                else:
                    return (0, 0)

            dx = np.sign(target[0] - current_pos[0])
            dy = np.sign(target[1] - current_pos[1])
            return (int(dx), int(dy))

        return (0, 0)

    def _should_replan(self) -> bool:
        """检测是否需要重新规划"""
        # 简化：总是假设可能需要重新规划
        return True

    def _is_path_valid(self, path: List[Tuple[int, int]],
                       obstacles: Set[Tuple[int, int]]) -> bool:
        """检查路径是否仍有效"""
        for pos in path[:5]:  # 只检查前几步
            if pos in obstacles:
                return False
        return True

    def get_state(self) -> Dict:
        """获取完整状态"""
        return self.space.get_visualization_fields()
