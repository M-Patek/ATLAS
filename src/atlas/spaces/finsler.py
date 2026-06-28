"""
Finsler Space
Finsler 空间

非对称度量空间
方向依赖的距离
"""

import numpy as np
from typing import Dict, Any, Tuple

from ..core.space import CognitiveSpace, register_space


@register_space("finsler")
class FinslerSpace(CognitiveSpace):
    """
    Finsler 认知空间

    非对称度量:
    d(x→y) ≠ d(y→x) 通常情况下

    应用:
    - 离开熟悉区域难，进入容易
    - 逆流难，顺流容易
    """

    def __init__(self, width: int, height: int,
                 asymmetry: float = 0.5,
                 **kwargs):
        super().__init__(width, height, name="finsler")

        self.asymmetry = asymmetry

        # 熟悉度场
        self.familiarity = np.zeros((width, height))

        # 目标方向场
        self.target_field = np.zeros((width, height))
        self.target_position: Optional[Tuple[int, int]] = None

    def _compute_cost(self, current: Tuple[int, int],
                     direction: Tuple[int, int]) -> float:
        """
        计算向某方向移动的成本

        考虑:
        - 离开熟悉区域惩罚
        - 朝向目标奖励
        """
        x, y = current
        dx, dy = direction

        # 基础成本
        base_cost = 1.0

        # 熟悉度惩罚（离开熟悉区域成本高）
        familiarity_penalty = 1.0 + self.asymmetry * self.familiarity[x, y]

        # 目标方向奖励
        if self.target_position:
            tx, ty = self.target_position
            # 到目标的方向
            to_target = np.array([tx - x, ty - y])
            target_dist = np.linalg.norm(to_target)

            if target_dist > 0:
                to_target = to_target / target_dist
                move_dir = np.array([dx, dy])
                move_norm = np.linalg.norm(move_dir)

                if move_norm > 0:
                    move_dir = move_dir / move_norm
                    alignment = np.dot(to_target, move_dir)
                    # 朝向目标降低成本
                    target_bonus = 1.0 - self.asymmetry * max(0, alignment)
                else:
                    target_bonus = 1.0
            else:
                target_bonus = 1.0
        else:
            target_bonus = 1.0

        return base_cost * familiarity_penalty * target_bonus

    def compute_distance(self, pos1: Tuple[int, int],
                        pos2: Tuple[int, int]) -> float:
        """
        Finsler 距离

        这里简化为直线距离（实际应该沿曲线积分）
        """
        x1, y1 = pos1
        x2, y2 = pos2

        # 方向
        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx**2 + dy**2)

        if dist < 1e-6:
            return 0.0

        # 归一化方向
        dx_norm = int(np.sign(dx)) if abs(dx) > abs(dy) else 0
        dy_norm = int(np.sign(dy)) if abs(dy) >= abs(dx) else 0

        # 沿路径积分
        n_steps = max(abs(dx), abs(dy))
        if n_steps == 0:
            return 0.0

        total = 0.0
        for i in range(n_steps):
            t = i / n_steps
            x = int(x1 + t * dx)
            y = int(y1 + t * dy)

            cost = self._compute_cost((x, y), (dx_norm, dy_norm))
            total += cost

        return total

    def get_heuristic(self, pos: Tuple[int, int],
                     goal: Tuple[int, int]) -> float:
        """
        Finsler 启发式

        使用对称化版本（Manhattan 距离）作为下界
        """
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def update_from_observation(self, position: Tuple[int, int],
                                observation: Dict[str, Any]) -> None:
        """更新熟悉度"""
        x, y = position

        if 0 <= x < self.width and 0 <= y < self.height:
            # 增加熟悉度
            self.familiarity[x, y] = min(1.0, self.familiarity[x, y] + 0.2)

        # 更新目标
        if 'goal_position' in observation:
            self.target_position = observation['goal_position']

            # 更新目标场
            gx, gy = self.target_position
            self.target_field.fill(0)
            for x in range(self.width):
                for y in range(self.height):
                    dist = np.sqrt((x-gx)**2 + (y-gy)**2)
                    if dist < 10:
                        self.target_field[x, y] = np.exp(-dist / 5.0)

    def get_visualization_fields(self) -> Dict[str, np.ndarray]:
        """可视化数据"""
        return {
            'familiarity': self.familiarity.copy(),
            'target_field': self.target_field.copy(),
        }
