"""
ATLAS Core: Cognitive Space Abstraction
认知空间抽象基类

定义了所有认知空间必须实现的接口。
这是可插拔架构的核心契约。
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Set, Any, Type
from dataclasses import dataclass


# Forward declaration for register_space (defined in registry)
_register_space_func = None

def register_space(name: str):
    """
    装饰器：自动注册空间类到全局注册表
    实际实现在 registry.py 中，这里仅作转发
    """
    def decorator(cls: Type) -> Type:
        # 延迟导入，避免循环依赖
        from .registry import registry
        registry.register(name, cls)
        return cls
    return decorator


@dataclass
class SpaceMetrics:
    """空间性能指标"""
    # 更新性能
    update_time_ms: float = 0.0
    update_stability: float = 0.0  # 更新后的平滑度

    # 压缩性能
    compression_ratio: float = 1.0
    compression_quality: float = 0.0  # 压缩后距离保持度

    # 规划性能
    planning_time_ms: float = 0.0
    planning_success_rate: float = 0.0
    path_efficiency: float = 0.0  # 路径长度/欧氏距离

    # 组合性能
    compose_time_ms: float = 0.0
    compose_quality: float = 0.0

    # 空间质量
    average_curvature: float = 0.0
    curvature_smoothness: float = 0.0


def neighbors_4(pos: Tuple[int, int], width: int, height: int) -> List[Tuple[int, int]]:
    """获取4连通邻居"""
    x, y = pos
    result = []
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            result.append((nx, ny))
    return result


def neighbors_8(pos: Tuple[int, int], width: int, height: int) -> List[Tuple[int, int]]:
    """获取8连通邻居"""
    x, y = pos
    result = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                result.append((nx, ny))
    return result


class CognitiveSpace(ABC):
    """
    认知空间抽象基类

    所有认知空间必须实现此接口才能被框架使用。
    这是 ATLAS 可插拔架构的核心契约。

    设计原则:
    - 空间是被动的（passive）：只描述，不主动更新
    - 所有更新通过 WorldModel 调用 update_from_observation
    - 距离计算必须是确定性的
    - 启发式必须是可接受的（admissible）
    """

    def __init__(self, width: int, height: int, name: str = "base"):
        self.width = width
        self.height = height
        self.name = name
        self.metadata: Dict[str, Any] = {}
        self.metrics = SpaceMetrics()

    @abstractmethod
    def compute_distance(self, pos1: Tuple[int, int],
                        pos2: Tuple[int, int]) -> float:
        """
        计算两点间的认知距离

        这是空间的核心度量，必须满足:
        - 非负性: d(x,y) >= 0
        - 同一性: d(x,x) = 0
        - 对称性: d(x,y) = d(y,x) [对Finsler空间可放松]
        - 三角不等式: d(x,z) <= d(x,y) + d(y,z)

        Args:
            pos1: 起点坐标 (x, y)
            pos2: 终点坐标 (x, y)

        Returns:
            两点间的认知距离（非负浮点数）
        """
        pass

    @abstractmethod
    def get_heuristic(self, pos: Tuple[int, int],
                     goal: Tuple[int, int]) -> float:
        """
        获取从 pos 到 goal 的启发式估计

        必须满足可接受性（admissible）:
        h(pos, goal) <= actual_cost(pos, goal)

        启发式的质量直接影响 A* 的搜索效率:
        - h = 0: 退化为 Dijkstra，保证最优但搜索空间大
        - h = perfect: 直接找到最优路径，无需扩展

        Args:
            pos: 当前位置
            goal: 目标位置

        Returns:
            启发式估计值（必须 <= 实际成本）
        """
        pass

    @abstractmethod
    def update_from_observation(self, position: Tuple[int, int],
                                observation: Dict[str, Any]) -> None:
        """
        根据观测更新空间结构

        这是 WorldModel 与空间的唯一交互点。
        空间在此方法中更新其内部状态（如 uncertainty, familiarity等）。

        Args:
            position: 观测发生的位置
            observation: 观测数据字典，可能包含:
                - 'obstacles': 障碍物列表
                - 'goal_position': 目标位置
                - 'goal_reached': 是否到达目标
                - 'features': 其他特征
        """
        pass

    def compute_path_cost(self, path: List[Tuple[int, int]]) -> float:
        """
        计算路径的总成本

        默认实现：累加相邻点间的距离
        子类可覆盖以实现特殊逻辑
        """
        if not path or len(path) < 2:
            return 0.0

        total = 0.0
        for i in range(len(path) - 1):
            total += self.compute_distance(path[i], path[i+1])
        return total

    def batch_update(self, observations: List[Tuple[Tuple[int, int], Dict]]) -> None:
        """
        批量更新空间

        Args:
            observations: [(position, observation_data), ...]
        """
        for position, observation in observations:
            self.update_from_observation(position, observation)

    def get_visualization_fields(self) -> Dict[str, np.ndarray]:
        """
        获取用于可视化的场数据

        返回的字典应包含可可视化的2D数组，如:
        - 'metric': 度量场
        - 'curvature': 曲率场
        - 'uncertainty': 不确定度场

        Returns:
            Dict[field_name, 2D numpy array]
        """
        return {
            'metric': np.ones((self.width, self.height)),
        }

    def predict_next_state(self, position: Tuple[int, int],
                           observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于当前空间场数据预测下一步状态

        这是 SSFR 结构假设的核心预测能力。
        每个空间基于自己的场数据预测：
        - 前方是否可通行
        - 到达目标的代价
        - 观测不确定性

        Args:
            position: 当前位置
            observation: 当前观测（包含目标、障碍物等）

        Returns:
            预测结果字典，至少包含:
            - 'predicted_position': 预测的下一步位置
            - 'predicted_cost': 预测的移动代价
            - 'predicted_uncertainty': 预测位置的不确定性
            - 'passable': 是否可通行
        """
        # 默认实现：基于当前位置和目标做简单预测
        goal = observation.get('goal_position')
        obstacles = set(observation.get('obstacles', []))

        # 找最佳邻居（最小距离方向）
        best_pos = position
        best_cost = float('inf')

        for nx, ny in neighbors_4(position, self.width, self.height):
            if (nx, ny) in obstacles:
                continue
            if goal:
                cost = self.compute_distance((nx, ny), goal)
            else:
                cost = self.compute_distance(position, (nx, ny))
            if cost < best_cost:
                best_cost = cost
                best_pos = (nx, ny)

        return {
            'predicted_position': best_pos,
            'predicted_cost': best_cost,
            'predicted_uncertainty': 0.5,
            'passable': best_pos != position,
        }

    def compute_validity(self, position: Tuple[int, int],
                        observation: Dict[str, Any],
                        actual: Optional[Dict[str, Any]] = None) -> float:
        """
        计算空间在当前场景下的"有效性"

        核心思想：不是外部条件判断，而是空间自己判断
        "我的数学模型在这个场景下还适用吗？"

        返回值：0.0 ~ 1.0
        - 1.0 = 完全适用（预测完美）
        - 0.5 = 边界区域（预测有偏差）
        - 0.0 = 完全失效（预测完全错误）

        类比：
        - 牛顿力学在低速场景：validity ≈ 1.0
        - 牛顿力学在高速场景：validity ≈ 0.1（需要相对论）
        - 欧氏空间在平坦地形：validity ≈ 1.0
        - 欧氏空间在强曲率地形：validity ≈ 0.2（需要Ricci）

        Args:
            position: 当前位置
            observation: 当前观测
            actual: 实际结果（用于验证预测，可选）

        Returns:
            有效性分数 [0.0, 1.0]
        """
        # 默认实现：基于预测能力
        prediction = self.predict_next_state(position, observation)

        if actual is None:
            # 没有实际结果，基于内部一致性
            uncertainty = prediction.get('predicted_uncertainty', 0.5)
            return max(0.0, 1.0 - uncertainty)

        # 有实际结果，计算预测误差
        predicted_pos = prediction.get('predicted_position', position)
        actual_pos = actual.get('position', position)

        pos_error = np.sqrt(
            (predicted_pos[0] - actual_pos[0])**2 +
            (predicted_pos[1] - actual_pos[1])**2
        )

        # 误差越大，validity越低
        max_error = np.sqrt(self.width**2 + self.height**2)
        normalized_error = min(pos_error / max_error, 1.0)

        return max(0.0, 1.0 - normalized_error)

    def get_validity_fields(self) -> Dict[str, np.ndarray]:
        """
        获取有效性场数据

        返回空间在每个位置的有效性分布。
        用于可视化"这个空间在哪些区域有效"。

        Returns:
            {'validity': 2D array [0,1]}
        """
        # 默认实现：基于场数据的简单估计
        fields = self.get_visualization_fields()

        # 如果有uncertainty场，用它估计validity
        if 'uncertainty' in fields:
            uncertainty = fields['uncertainty']
            validity = 1.0 - np.clip(uncertainty, 0, 1)
        elif 'metric' in fields:
            # metric越接近1（欧氏），越有效
            metric = fields['metric']
            validity = 1.0 - np.clip(np.abs(metric - 1.0), 0, 1)
        else:
            validity = np.ones((self.width, self.height)) * 0.5

        return {'validity': validity}

    def get_statistics(self) -> Dict[str, float]:
        """获取空间统计信息"""
        return {
            'name': self.name,
            'width': self.width,
            'height': self.height,
        }

    def clone(self) -> 'CognitiveSpace':
        """
        创建空间的深拷贝

        用于实验中的控制变量测试
        """
        import copy
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', {self.width}x{self.height})"


# ============================================================================
# Utility Functions
# ============================================================================

def line_sample(pos1: Tuple[int, int], pos2: Tuple[int, int],
                num_samples: Optional[int] = None) -> List[Tuple[int, int]]:
    """
    在两点间采样

    使用 Bresenham 算法的简化版
    """
    x1, y1 = pos1
    x2, y2 = pos2

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    if num_samples is None:
        num_samples = max(dx, dy) + 1

    points = []
    for i in range(num_samples):
        t = i / max(1, num_samples - 1)
        x = int(x1 + t * (x2 - x1))
        y = int(y1 + t * (y2 - y1))
        points.append((x, y))

    return points


def euclidean_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """欧氏距离"""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """曼哈顿距离"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
