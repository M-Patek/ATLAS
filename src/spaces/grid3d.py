"""
3D Grid Space
3D网格空间

扩展ATLAS到3D网格，支持真实机器人导航。
坐标系: (x, y, z) 其中z为高度/层

注意: 3D空间使用三元组(x,y,z)替代二元组(x,y)
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from ..core.space import CognitiveSpace, register_space


@dataclass
class Pos3D:
    """3D位置，支持元组和类访问"""
    x: int
    y: int
    z: int

    @classmethod
    def from_tuple(cls, t: Tuple[int, int, int]) -> 'Pos3D':
        return cls(t[0], t[1], t[2])

    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.x, self.y, self.z)

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z


def euclidean_distance_3d(pos1: Tuple[int, int, int],
                          pos2: Tuple[int, int, int]) -> float:
    """3D欧氏距离"""
    return np.sqrt(
        (pos1[0] - pos2[0])**2 +
        (pos1[1] - pos2[1])**2 +
        (pos1[2] - pos2[2])**2
    )


def manhattan_distance_3d(pos1: Tuple[int, int, int],
                          pos2: Tuple[int, int, int]) -> float:
    """3D曼哈顿距离"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) + abs(pos1[2] - pos2[2])


def neighbors_6(pos: Tuple[int, int, int],
                width: int, height: int, depth: int) -> List[Tuple[int, int, int]]:
    """获取6连通邻居 (上下左右前后)"""
    x, y, z = pos
    result = []
    for dx, dy, dz in [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]:
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx < width and 0 <= ny < height and 0 <= nz < depth:
            result.append((nx, ny, nz))
    return result


def neighbors_26(pos: Tuple[int, int, int],
                 width: int, height: int, depth: int) -> List[Tuple[int, int, int]]:
    """获取26连通邻居 (包括对角线)"""
    x, y, z = pos
    result = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < width and 0 <= ny < height and 0 <= nz < depth:
                    result.append((nx, ny, nz))
    return result


class CognitiveSpace3D(CognitiveSpace):
    """
    3D认知空间基类

    继承自CognitiveSpace，但使用3D坐标(x,y,z)替代2D坐标(x,y)
    """

    def __init__(self, width: int, height: int, depth: int, name: str = "base3d"):
        # 调用父类但设置depth为额外属性
        super().__init__(width, height, name=name)
        self.depth = depth

    def compute_distance(self, pos1: Tuple[int, ...],
                        pos2: Tuple[int, ...]) -> float:
        """计算3D距离"""
        # 确保位置是3D
        p1 = self._ensure_3d(pos1)
        p2 = self._ensure_3d(pos2)
        return euclidean_distance_3d(p1, p2)

    def get_heuristic(self, pos: Tuple[int, ...],
                     goal: Tuple[int, ...]) -> float:
        """3D启发式"""
        p = self._ensure_3d(pos)
        g = self._ensure_3d(goal)
        return euclidean_distance_3d(p, g)

    def update_from_observation(self, position: Tuple[int, ...],
                                observation: Dict[str, Any]) -> None:
        """根据观测更新空间结构"""
        pass

    def get_neighbors(self, pos: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """获取邻居节点，默认6连通"""
        p = self._ensure_3d(pos)
        return neighbors_6(p, self.width, self.height, self.depth)

    def _ensure_3d(self, pos) -> Tuple[int, int, int]:
        """确保位置是3D元组"""
        if isinstance(pos, tuple):
            if len(pos) == 2:
                return (pos[0], pos[1], 0)
            return pos
        return (int(pos[0]), int(pos[1]), int(pos[2]) if len(pos) > 2 else 0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', {self.width}x{self.height}x{self.depth})"


@register_space("euclidean3d")
class EuclideanSpace3D(CognitiveSpace3D):
    """
    3D欧氏空间基线

    最简单的3D认知空间，作为性能基准。
    """

    def __init__(self, width: int = 20, height: int = 20, depth: int = 5, **kwargs):
        super().__init__(width, height, depth, name="euclidean3d")

    def compute_distance(self, pos1: Tuple[int, ...],
                        pos2: Tuple[int, ...]) -> float:
        """3D欧氏距离"""
        p1 = self._ensure_3d(pos1)
        p2 = self._ensure_3d(pos2)
        return euclidean_distance_3d(p1, p2)

    def get_heuristic(self, pos: Tuple[int, ...],
                     goal: Tuple[int, ...]) -> float:
        """3D欧氏启发式"""
        p = self._ensure_3d(pos)
        g = self._ensure_3d(goal)
        return euclidean_distance_3d(p, g)

    def update_from_observation(self, position: Tuple[int, ...],
                                observation: Dict[str, Any]) -> None:
        """欧氏空间不更新"""
        pass

    def get_visualization_fields(self) -> Dict[str, np.ndarray]:
        """返回单位场(z=0层)"""
        return {
            'metric': np.ones((self.width, self.height)),
        }


@register_space("ricci3d")
class RicciSpace3D(CognitiveSpace3D):
    """
    3D Ricci流空间

    将Ricci曲率流扩展到3D，g_ij = (1 + |R|)^2 * delta_ij
    其中R是uncertainty的拉普拉斯。

    3D梯度计算使用6邻域或26邻域。
    """

    def __init__(self, width: int = 20, height: int = 20, depth: int = 5,
                 curvature_scale: float = 1.0,
                 neighbor_mode: str = "6"):
        super().__init__(width, height, depth, name="ricci3d")
        self.curvature_scale = curvature_scale
        self.neighbor_mode = neighbor_mode  # "6" 或 "26"

        # 3D场
        self.uncertainty = np.ones((width, height, depth)) * 0.5
        self.metric = np.ones((width, height, depth))
        self.curvature = np.zeros((width, height, depth))

    def compute_distance(self, pos1: Tuple[int, ...],
                        pos2: Tuple[int, ...]) -> float:
        """基于局部度量的距离"""
        p1 = self._ensure_3d(pos1)
        p2 = self._ensure_3d(pos2)
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        # 边界检查
        if not (0 <= x1 < self.width and 0 <= y1 < self.height and 0 <= z1 < self.depth):
            return float('inf')
        if not (0 <= x2 < self.width and 0 <= y2 < self.height and 0 <= z2 < self.depth):
            return float('inf')

        # 获取两点之间的平均度量
        g1 = self.metric[x1, y1, z1]
        g2 = self.metric[x2, y2, z2]
        g_avg = (g1 + g2) / 2.0

        # 物理欧氏距离
        euclidean = euclidean_distance_3d(p1, p2)

        # 认知距离 = 物理距离 * 平均度量
        return euclidean * g_avg

    def get_heuristic(self, pos: Tuple[int, ...],
                     goal: Tuple[int, ...]) -> float:
        """3D欧氏启发式 (可接受)"""
        p = self._ensure_3d(pos)
        g = self._ensure_3d(goal)
        return euclidean_distance_3d(p, g)

    def update_from_observation(self, position: Tuple[int, ...],
                                observation: Dict[str, Any]) -> None:
        """根据观测更新3D uncertainty场"""
        p = self._ensure_3d(position)
        x, y, z = p

        if not (0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth):
            return

        if 'uncertainty' in observation:
            self.uncertainty[x, y, z] = observation['uncertainty']

        if 'obstacles' in observation:
            for obs in observation['obstacles']:
                if isinstance(obs, tuple):
                    if len(obs) == 3:
                        ox, oy, oz = obs
                    else:
                        ox, oy = obs
                        oz = 0
                    if 0 <= ox < self.width and 0 <= oy < self.height and 0 <= oz < self.depth:
                        self.uncertainty[ox, oy, oz] = 1.0

        if 'visited' in observation:
            for v_pos in observation['visited']:
                if isinstance(v_pos, tuple):
                    if len(v_pos) == 3:
                        vx, vy, vz = v_pos
                    else:
                        vx, vy = v_pos
                        vz = 0
                    if 0 <= vx < self.width and 0 <= vy < self.height and 0 <= vz < self.depth:
                        self.uncertainty[vx, vy, vz] *= 0.9

        self._update_metric()

    def _update_metric(self):
        """基于uncertainty计算曲率和度量"""
        try:
            from scipy import ndimage

            # 3D拉普拉斯核 (6邻域)
            laplacian_kernel = np.array([
                [[0, 0, 0],
                 [0, 1, 0],
                 [0, 0, 0]],
                [[0, 1, 0],
                 [1, -6, 1],
                 [0, 1, 0]],
                [[0, 0, 0],
                 [0, 1, 0],
                 [0, 0, 0]]
            ])

            # 计算曲率 (uncertainty的拉普拉斯)
            self.curvature = ndimage.convolve(
                self.uncertainty, laplacian_kernel, mode='constant'
            )
        except ImportError:
            # scipy不可用时使用简化计算
            self.curvature = np.zeros_like(self.uncertainty)

        # 度量: g = (1 + scale * |R|)^2
        R_abs = np.abs(self.curvature)
        self.metric = (1 + self.curvature_scale * R_abs) ** 2

    def get_neighbors(self, pos: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """获取邻居"""
        p = self._ensure_3d(pos)
        if self.neighbor_mode == "26":
            return neighbors_26(p, self.width, self.height, self.depth)
        return neighbors_6(p, self.width, self.height, self.depth)

    def get_visualization_fields(self) -> Dict[str, np.ndarray]:
        """返回z=0层用于可视化"""
        return {
            'metric': self.metric[:, :, 0] if self.depth > 0 else np.ones((self.width, self.height)),
            'uncertainty': self.uncertainty[:, :, 0] if self.depth > 0 else np.ones((self.width, self.height)),
        }


@register_space("conformal3d")
class ConformalSpace3D(CognitiveSpace3D):
    """
    3D共形变换空间

    通过标量场omega进行度量变换：g_tilde = exp(2*omega) * g
    """

    def __init__(self, width: int = 20, height: int = 20, depth: int = 5,
                 omega_field: Optional[np.ndarray] = None):
        super().__init__(width, height, depth, name="conformal3d")

        if omega_field is not None:
            self.omega = omega_field
        else:
            self.omega = np.zeros((width, height, depth))

        self.metric_factor = np.exp(2 * self.omega)

    def compute_distance(self, pos1: Tuple[int, ...],
                        pos2: Tuple[int, ...]) -> float:
        """共形变换后的距离"""
        p1 = self._ensure_3d(pos1)
        p2 = self._ensure_3d(pos2)
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        if not (0 <= x1 < self.width and 0 <= y1 < self.height and 0 <= z1 < self.depth):
            return float('inf')
        if not (0 <= x2 < self.width and 0 <= y2 < self.height and 0 <= z2 < self.depth):
            return float('inf')

        omega_avg = (self.omega[x1, y1, z1] + self.omega[x2, y2, z2]) / 2.0
        euclidean = euclidean_distance_3d(p1, p2)

        return np.exp(2 * omega_avg) * euclidean

    def get_heuristic(self, pos: Tuple[int, ...],
                     goal: Tuple[int, ...]) -> float:
        """启发式（保守估计）"""
        p = self._ensure_3d(pos)
        g = self._ensure_3d(goal)
        return euclidean_distance_3d(p, g)

    def update_from_observation(self, position: Tuple[int, ...],
                                observation: Dict[str, Any]) -> None:
        """更新共形因子"""
        p = self._ensure_3d(position)
        x, y, z = p

        if not (0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth):
            return

        if 'omega' in observation:
            self.omega[x, y, z] = observation['omega']
            self.metric_factor = np.exp(2 * self.omega)

        if 'attractor' in observation:
            # 在吸引器附近减小度量（更容易到达）
            attr = observation['attractor']
            if len(attr) == 2:
                ax, ay, az = attr[0], attr[1], 0
            else:
                ax, ay, az = attr

            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    for dz in range(-2, 3):
                        nx, ny, nz = ax + dx, ay + dy, az + dz
                        if 0 <= nx < self.width and 0 <= ny < self.height and 0 <= nz < self.depth:
                            dist = np.sqrt(dx**2 + dy**2 + dz**2)
                            self.omega[nx, ny, nz] -= 0.5 / (1 + dist)
            self.metric_factor = np.exp(2 * self.omega)

    def get_visualization_fields(self) -> Dict[str, np.ndarray]:
        """返回z=0层"""
        return {
            'metric': self.metric_factor[:, :, 0] if self.depth > 0 else np.ones((self.width, self.height)),
        }

