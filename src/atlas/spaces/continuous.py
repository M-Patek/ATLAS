"""
Continuous Cognitive Space - 连续认知空间基类

核心设计：完全移除离散网格，所有操作在连续坐标上进行。

关键变化：
1. 位置: Tuple[int, int] → Tuple[float, float]
2. 场数据: numpy 2D array → 函数近似/采样点
3. 邻居: 4/8连通 → 连续方向采样
4. 距离: 网格积分 → 连续路径积分
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Set, Any, Type, Callable
from dataclasses import dataclass
from collections import defaultdict
import math


@dataclass
class ContinuousSpaceMetrics:
    """连续空间性能指标"""
    update_time_ms: float = 0.0
    distance_compute_time_ms: float = 0.0
    field_query_time_ms: float = 0.0
    num_field_samples: int = 0


class ContinuousField:
    """
    连续场 - 用稀疏采样点 + 插值表示连续场

    替代离散的 numpy 2D array，支持任意精度的连续查询。
    """

    def __init__(self, default_value: float = 0.0,
                 interpolation: str = 'linear'):
        self.default_value = default_value
        self.interpolation = interpolation

        # 稀疏采样点: {(x, y): value, ...}
        self.samples: Dict[Tuple[float, float], float] = {}

        # 空间索引（加速最近邻查询）
        self._grid_index: Dict[Tuple[int, int], List[Tuple[float, float]]] = defaultdict(list)
        self._index_resolution = 1.0  # 索引网格分辨率

        # 缓存
        self._cache: Dict[Tuple[float, float], float] = {}
        self._cache_size = 1000

    def _get_index_key(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        """获取空间索引键"""
        x, y = pos
        return (int(x / self._index_resolution), int(y / self._index_resolution))

    def add_sample(self, position: Tuple[float, float], value: float):
        """添加采样点"""
        self.samples[position] = value
        idx = self._get_index_key(position)
        self._grid_index[idx].append(position)
        self._cache.clear()  # 清除缓存

    def query(self, position: Tuple[float, float],
              k: int = 4, radius: float = 2.0) -> float:
        """
        查询场值（使用k近邻插值）

        Args:
            position: 查询位置 (x, y)
            k: 近邻数量
            radius: 最大搜索半径

        Returns:
            插值后的场值
        """
        # 检查缓存
        if position in self._cache:
            return self._cache[position]

        if not self.samples:
            return self.default_value

        # 快速索引查找候选点
        idx = self._get_index_key(position)
        candidates = []

        # 搜索周围索引格子
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                cell = (idx[0] + dx, idx[1] + dy)
                for sample_pos in self._grid_index.get(cell, []):
                    dist = math.sqrt(
                        (sample_pos[0] - position[0])**2 +
                        (sample_pos[1] - position[1])**2
                    )
                    if dist <= radius:
                        candidates.append((dist, sample_pos))

        if not candidates:
            return self.default_value

        # 按距离排序，取前k个
        candidates.sort(key=lambda x: x[0])
        neighbors = candidates[:k]

        # 反距离加权插值
        if len(neighbors) == 1:
            result = self.samples[neighbors[0][1]]
        else:
            weights = []
            values = []
            for dist, sample_pos in neighbors:
                if dist < 1e-6:
                    # 精确命中
                    result = self.samples[sample_pos]
                    self._cache[position] = result
                    if len(self._cache) > self._cache_size:
                        self._cache.pop(next(iter(self._cache)))
                    return result
                w = 1.0 / (dist ** 2)
                weights.append(w)
                values.append(self.samples[sample_pos])

            total_weight = sum(weights)
            result = sum(v * w for v, w in zip(values, weights)) / total_weight

        # 缓存结果
        self._cache[position] = result
        if len(self._cache) > self._cache_size:
            self._cache.pop(next(iter(self._cache)))

        return result

    def query_batch(self, positions: List[Tuple[float, float]],
                    k: int = 4) -> List[float]:
        """批量查询"""
        return [self.query(pos, k) for pos in positions]

    def get_samples_in_region(self, center: Tuple[float, float],
                              radius: float) -> List[Tuple[Tuple[float, float], float]]:
        """获取区域内的所有采样点"""
        result = []
        idx = self._get_index_key(center)

        grid_radius = int(radius / self._index_resolution) + 1
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                cell = (idx[0] + dx, idx[1] + dy)
                for sample_pos in self._grid_index.get(cell, []):
                    dist = math.sqrt(
                        (sample_pos[0] - center[0])**2 +
                        (sample_pos[1] - center[1])**2
                    )
                    if dist <= radius:
                        result.append((sample_pos, self.samples[sample_pos]))

        return result

    def num_samples(self) -> int:
        """采样点数量"""
        return len(self.samples)

    def clear(self):
        """清空所有采样点"""
        self.samples.clear()
        self._grid_index.clear()
        self._cache.clear()


class ContinuousCognitiveSpace(ABC):
    """
    连续认知空间抽象基类

    完全移除离散网格依赖，所有操作在连续坐标上进行。
    """

    def __init__(self, name: str = "continuous_base"):
        self.name = name
        self.metadata: Dict[str, Any] = {}
        self.metrics = ContinuousSpaceMetrics()

    @abstractmethod
    def compute_distance(self, pos1: Tuple[float, float],
                        pos2: Tuple[float, float]) -> float:
        """
        计算两点间的认知距离（连续坐标）

        必须满足:
        - 非负性: d(x,y) >= 0
        - 同一性: d(x,x) = 0
        - 对称性: d(x,y) = d(y,x)
        - 三角不等式: d(x,z) <= d(x,y) + d(y,z)
        """
        pass

    @abstractmethod
    def get_heuristic(self, pos: Tuple[float, float],
                     goal: Tuple[float, float]) -> float:
        """
        获取启发式估计（必须可接受）
        h(pos, goal) <= actual_cost(pos, goal)
        """
        pass

    @abstractmethod
    def update_from_observation(self, position: Tuple[float, float],
                                observation: Dict[str, Any]) -> None:
        """
        根据观测更新空间结构（连续位置）
        """
        pass

    def compute_path_cost(self, path: List[Tuple[float, float]]) -> float:
        """计算路径总成本"""
        if not path or len(path) < 2:
            return 0.0

        total = 0.0
        for i in range(len(path) - 1):
            total += self.compute_distance(path[i], path[i+1])
        return total

    def predict_next_state(self, position: Tuple[float, float],
                           observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        预测下一步状态（连续版本）

        在连续空间中，我们采样多个方向，选择最优方向。
        """
        goal = observation.get('goal_position')
        obstacles = observation.get('obstacles', [])

        # 采样多个方向（16个方向）
        num_directions = 16
        best_pos = position
        best_cost = float('inf')

        step_size = observation.get('step_size', 0.5)

        for i in range(num_directions):
            angle = 2 * math.pi * i / num_directions
            nx = position[0] + step_size * math.cos(angle)
            ny = position[1] + step_size * math.sin(angle)
            candidate = (nx, ny)

            # 检查碰撞
            if self._check_collision(candidate, obstacles):
                continue

            if goal:
                cost = self.compute_distance(candidate, goal)
            else:
                cost = self.compute_distance(position, candidate)

            if cost < best_cost:
                best_cost = cost
                best_pos = candidate

        return {
            'predicted_position': best_pos,
            'predicted_cost': best_cost,
            'predicted_uncertainty': 0.5,
            'passable': best_pos != position,
        }

    def _check_collision(self, position: Tuple[float, float],
                        obstacles: List[Tuple[float, float]],
                        obstacle_radius: float = 0.3) -> bool:
        """检查位置是否与障碍物碰撞"""
        for obs in obstacles:
            dist = math.sqrt(
                (position[0] - obs[0])**2 +
                (position[1] - obs[1])**2
            )
            if dist < obstacle_radius:
                return True
        return False

    def compute_validity(self, position: Tuple[float, float],
                        observation: Dict[str, Any],
                        actual: Optional[Dict[str, Any]] = None) -> float:
        """计算空间在当前场景下的有效性"""
        prediction = self.predict_next_state(position, observation)

        if actual is None:
            uncertainty = prediction.get('predicted_uncertainty', 0.5)
            return max(0.0, 1.0 - uncertainty)

        predicted_pos = prediction.get('predicted_position', position)
        actual_pos = actual.get('position', position)

        pos_error = math.sqrt(
            (predicted_pos[0] - actual_pos[0])**2 +
            (predicted_pos[1] - actual_pos[1])**2
        )

        # 归一化误差
        normalized_error = min(pos_error / 10.0, 1.0)
        return max(0.0, 1.0 - normalized_error)

    def get_statistics(self) -> Dict[str, Any]:
        """获取空间统计信息"""
        return {
            'name': self.name,
            'type': 'continuous',
        }

    def clone(self) -> 'ContinuousCognitiveSpace':
        """深拷贝"""
        import copy
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', continuous)"


# ============================================================================
# 连续欧氏空间
# ============================================================================

class ContinuousEuclideanSpace(ContinuousCognitiveSpace):
    """连续欧氏空间基线"""

    def __init__(self, **kwargs):
        super().__init__(name="euclidean_continuous")

    def compute_distance(self, pos1: Tuple[float, float],
                        pos2: Tuple[float, float]) -> float:
        return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

    def get_heuristic(self, pos: Tuple[float, float],
                     goal: Tuple[float, float]) -> float:
        return self.compute_distance(pos, goal)

    def update_from_observation(self, position: Tuple[float, float],
                                observation: Dict[str, Any]) -> None:
        pass

    def compute_validity(self, position: Tuple[float, float],
                        observation: Dict[str, Any],
                        actual: Optional[Dict[str, Any]] = None) -> float:
        obstacles = observation.get('obstacles', [])
        if not obstacles:
            return 0.9

        # 检查附近障碍物密度
        x, y = position
        nearby = sum(1 for ox, oy in obstacles
                    if math.sqrt((ox-x)**2 + (oy-y)**2) <= 1.5)

        if nearby == 0:
            return 0.9
        elif nearby <= 2:
            return 0.7
        elif nearby <= 5:
            return 0.4
        else:
            return 0.2


# ============================================================================
# 连续Ricci空间
# ============================================================================

class ContinuousRicciSpace(ContinuousCognitiveSpace):
    """
    连续Ricci空间

    使用连续场表示曲率、不确定性和熟悉度。
    """

    def __init__(self, curvature_scale: float = 1.0,
                 familiarity_decay: float = 0.1,
                 **kwargs):
        super().__init__(name="ricci_continuous")

        self.curvature_scale = curvature_scale
        self.familiarity_decay = familiarity_decay

        # 连续场
        self.uncertainty_field = ContinuousField(default_value=0.5)
        self.curvature_field = ContinuousField(default_value=0.0)
        self.familiarity_field = ContinuousField(default_value=0.0)

        # 访问记录
        self.visit_points: List[Tuple[float, float]] = []

        # 目标吸引子
        self.attractor_position: Optional[Tuple[float, float]] = None
        self.attractor_strength: float = 0.0

    def compute_distance(self, pos1: Tuple[float, float],
                        pos2: Tuple[float, float]) -> float:
        """连续Ricci距离 - 沿路径积分"""
        euclidean_dist = math.sqrt(
            (pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2
        )
        if euclidean_dist < 0.01:
            return 0.0

        # 沿路径采样积分
        n_samples = max(int(euclidean_dist * 4), 10)
        total = 0.0

        for i in range(n_samples):
            t = (i + 0.5) / n_samples
            x = pos1[0] + t * (pos2[0] - pos1[0])
            y = pos1[1] + t * (pos2[1] - pos1[1])

            curvature = self.curvature_field.query((x, y))
            metric_factor = 1.0 + self.curvature_scale * abs(curvature)
            total += metric_factor * (euclidean_dist / n_samples)

        return total

    def get_heuristic(self, pos: Tuple[float, float],
                     goal: Tuple[float, float]) -> float:
        return self.compute_distance(pos, goal)

    def update_from_observation(self, position: Tuple[float, float],
                                observation: Dict[str, Any]) -> None:
        """更新连续场"""
        # 记录访问
        self.visit_points.append(position)

        # 更新熟悉度（在访问位置）
        current_fam = self.familiarity_field.query(position)
        self.familiarity_field.add_sample(
            position,
            min(1.0, current_fam + self.familiarity_decay)
        )

        # 障碍物增加不确定性
        if 'obstacles' in observation:
            for obs in observation['obstacles']:
                # 在障碍物周围添加高不确定性
                for _ in range(5):  # 每个障碍物周围采样5个点
                    angle = 2 * math.pi * np.random.random()
                    dist = 0.3 + 0.5 * np.random.random()
                    sample_pos = (
                        obs[0] + dist * math.cos(angle),
                        obs[1] + dist * math.sin(angle)
                    )
                    current_unc = self.uncertainty_field.query(sample_pos)
                    self.uncertainty_field.add_sample(
                        sample_pos,
                        min(1.0, current_unc + 0.3)
                    )

        # 目标降低不确定性
        if observation.get('goal_reached') and 'goal_position' in observation:
            goal = observation['goal_position']
            for _ in range(10):
                angle = 2 * math.pi * np.random.random()
                dist = 0.5 * np.random.random()
                sample_pos = (
                    goal[0] + dist * math.cos(angle),
                    goal[1] + dist * math.sin(angle)
                )
                current_unc = self.uncertainty_field.query(sample_pos)
                self.uncertainty_field.add_sample(
                    sample_pos,
                    max(0.0, current_unc * 0.7)
                )

        # 更新attractor
        if 'goal_position' in observation:
            self.attractor_position = observation['goal_position']
            self.attractor_strength = observation.get('goal_strength', 0.5)

        # 重新计算曲率场（基于不确定性场）
        self._recalculate_curvature()

    def _recalculate_curvature(self):
        """从不确定性场计算曲率（连续版本）"""
        # 在已有采样点上计算曲率
        for pos in list(self.uncertainty_field.samples.keys()):
            # 采样周围点计算Laplacian
            h = 0.1
            samples = [
                (pos[0]+h, pos[1]), (pos[0]-h, pos[1]),
                (pos[0], pos[1]+h), (pos[0], pos[1]-h),
            ]
            values = [self.uncertainty_field.query(s) for s in samples]
            center = self.uncertainty_field.query(pos)

            # 离散Laplacian
            laplacian = (sum(values) - 4*center) / (h**2)
            curvature = -laplacian

            # 应用熟悉度衰减
            familiarity = self.familiarity_field.query(pos)
            curvature *= (1.0 - familiarity)

            self.curvature_field.add_sample(pos, curvature)

    def predict_next_state(self, position: Tuple[float, float],
                           observation: Dict[str, Any]) -> Dict[str, Any]:
        """Ricci特有的预测"""
        goal = observation.get('goal_position')
        obstacles = observation.get('obstacles', [])

        num_directions = 16
        best_pos = position
        best_score = float('-inf')
        step_size = observation.get('step_size', 0.5)

        for i in range(num_directions):
            angle = 2 * math.pi * i / num_directions
            nx = position[0] + step_size * math.cos(angle)
            ny = position[1] + step_size * math.sin(angle)
            candidate = (nx, ny)

            if self._check_collision(candidate, obstacles):
                continue

            curvature_penalty = -abs(self.curvature_field.query(candidate)) * self.curvature_scale
            familiarity_bonus = self.familiarity_field.query(candidate)
            uncertainty_penalty = -self.uncertainty_field.query(candidate)

            goal_bonus = 0.0
            if goal:
                dist_to_goal = math.sqrt((nx-goal[0])**2 + (ny-goal[1])**2)
                goal_bonus = -dist_to_goal * 0.1

            score = curvature_penalty + familiarity_bonus + uncertainty_penalty + goal_bonus

            if score > best_score:
                best_score = score
                best_pos = candidate

        return {
            'predicted_position': best_pos,
            'predicted_cost': self.compute_distance(position, best_pos),
            'predicted_uncertainty': self.uncertainty_field.query(best_pos),
            'passable': best_pos != position,
            'ricci_score': best_score,
            'predicted_curvature': self.curvature_field.query(best_pos),
            'predicted_familiarity': self.familiarity_field.query(best_pos),
        }

    def compute_validity(self, position: Tuple[float, float],
                        observation: Dict[str, Any],
                        actual: Optional[Dict[str, Any]] = None) -> float:
        curvature = abs(self.curvature_field.query(position))
        uncertainty = self.uncertainty_field.query(position)

        if curvature < 0.5:
            validity = 0.3 + 0.3 * (curvature / 0.5)
        elif curvature < 2.0:
            validity = 0.6 + 0.4 * ((curvature - 0.5) / 1.5)
        else:
            validity = 1.0 - 0.3 * min((curvature - 2.0) / 3.0, 1.0)

        return min(1.0, validity + 0.2 * uncertainty)

    def get_statistics(self) -> Dict[str, Any]:
        stats = super().get_statistics()
        stats.update({
            'uncertainty_samples': self.uncertainty_field.num_samples(),
            'curvature_samples': self.curvature_field.num_samples(),
            'familiarity_samples': self.familiarity_field.num_samples(),
            'visit_count': len(self.visit_points),
        })
        return stats


# ============================================================================
# 连续Fisher空间
# ============================================================================

class ContinuousFisherSpace(ContinuousCognitiveSpace):
    """连续Fisher信息空间"""

    def __init__(self, temperature: float = 1.0, **kwargs):
        super().__init__(name="fisher_continuous")

        self.temperature = temperature

        # 连续场
        self.belief_field = ContinuousField(default_value=0.5)
        self.confidence_field = ContinuousField(default_value=0.0)

        # 观测记录
        self.observations: List[Tuple[Tuple[float, float], Dict]] = []

    def _fisher_metric(self, position: Tuple[float, float]) -> float:
        conf = max(0.01, self.confidence_field.query(position))
        return 1.0 / conf

    def compute_distance(self, pos1: Tuple[float, float],
                        pos2: Tuple[float, float]) -> float:
        euclidean_dist = math.sqrt(
            (pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2
        )
        if euclidean_dist < 0.01:
            return 0.0

        n_samples = max(int(euclidean_dist * 4), 10)
        total = 0.0

        for i in range(n_samples):
            t = (i + 0.5) / n_samples
            x = pos1[0] + t * (pos2[0] - pos1[0])
            y = pos1[1] + t * (pos2[1] - pos1[1])

            fisher_factor = self._fisher_metric((x, y))
            total += fisher_factor * (euclidean_dist / n_samples)

        return total

    def get_heuristic(self, pos: Tuple[float, float],
                     goal: Tuple[float, float]) -> float:
        return self.compute_distance(pos, goal)

    def update_from_observation(self, position: Tuple[float, float],
                                observation: Dict[str, Any]) -> None:
        self.observations.append((position, observation))

        # 更新置信度
        current_conf = self.confidence_field.query(position)
        # 观测越多越自信
        new_conf = min(1.0, 1.0 - math.exp(-0.2 * len(self.observations)))
        self.confidence_field.add_sample(position, new_conf)

        # 障碍物降低信念
        if 'obstacles' in observation:
            for obs in observation['obstacles']:
                for _ in range(3):
                    angle = 2 * math.pi * np.random.random()
                    dist = 0.2 + 0.3 * np.random.random()
                    sample_pos = (
                        obs[0] + dist * math.cos(angle),
                        obs[1] + dist * math.sin(angle)
                    )
                    current_belief = self.belief_field.query(sample_pos)
                    self.belief_field.add_sample(sample_pos, current_belief * 0.9)
                    current_conf = self.confidence_field.query(sample_pos)
                    self.confidence_field.add_sample(sample_pos, max(0.0, current_conf - 0.05))

        # 目标增加信念
        if 'goal_position' in observation:
            goal = observation['goal_position']
            for _ in range(5):
                angle = 2 * math.pi * np.random.random()
                dist = 0.5 * np.random.random()
                sample_pos = (
                    goal[0] + dist * math.cos(angle),
                    goal[1] + dist * math.sin(angle)
                )
                current_belief = self.belief_field.query(sample_pos)
                d = math.sqrt((sample_pos[0]-goal[0])**2 + (sample_pos[1]-goal[1])**2)
                boost = 0.1 * (1 - d/5) if d < 5 else 0
                self.belief_field.add_sample(sample_pos, min(1.0, current_belief + boost))

    def predict_next_state(self, position: Tuple[float, float],
                           observation: Dict[str, Any]) -> Dict[str, Any]:
        goal = observation.get('goal_position')
        obstacles = observation.get('obstacles', [])

        num_directions = 16
        best_pos = position
        best_score = float('-inf')
        step_size = observation.get('step_size', 0.5)

        for i in range(num_directions):
            angle = 2 * math.pi * i / num_directions
            nx = position[0] + step_size * math.cos(angle)
            ny = position[1] + step_size * math.sin(angle)
            candidate = (nx, ny)

            if self._check_collision(candidate, obstacles):
                continue

            confidence_bonus = self.confidence_field.query(candidate)
            belief_bonus = self.belief_field.query(candidate)
            fisher_metric = 1.0 / max(0.01, self.confidence_field.query(candidate))
            cost_penalty = -fisher_metric * 0.1

            goal_bonus = 0.0
            if goal:
                dist_to_goal = math.sqrt((nx-goal[0])**2 + (ny-goal[1])**2)
                goal_bonus = -dist_to_goal * 0.1

            score = confidence_bonus + belief_bonus + cost_penalty + goal_bonus

            if score > best_score:
                best_score = score
                best_pos = candidate

        return {
            'predicted_position': best_pos,
            'predicted_cost': self.compute_distance(position, best_pos),
            'predicted_uncertainty': 1.0 - self.confidence_field.query(best_pos),
            'passable': best_pos != position,
            'fisher_score': best_score,
            'predicted_confidence': self.confidence_field.query(best_pos),
            'predicted_belief': self.belief_field.query(best_pos),
        }

    def compute_validity(self, position: Tuple[float, float],
                        observation: Dict[str, Any],
                        actual: Optional[Dict[str, Any]] = None) -> float:
        current_conf = self.confidence_field.query(position)
        current_belief = self.belief_field.query(position)

        if current_conf > 0.7:
            validity = 0.7 + 0.3 * min((current_conf - 0.7) / 0.3, 1.0)
        elif current_conf > 0.3:
            validity = 0.4 + 0.3 * ((current_conf - 0.3) / 0.4)
        else:
            validity = 0.2 + 0.2 * (current_conf / 0.3)

        return min(1.0, validity + 0.1 * current_belief)


# ============================================================================
# 连续Wasserstein空间
# ============================================================================

class ContinuousWassersteinSpace(ContinuousCognitiveSpace):
    """连续Wasserstein空间"""

    def __init__(self, base_cost: float = 1.0, **kwargs):
        super().__init__(name="wasserstein_continuous")

        self.base_cost = base_cost

        # 连续场
        self.cost_field = ContinuousField(default_value=1.0)
        self.mass_field = ContinuousField(default_value=0.1)

    def compute_distance(self, pos1: Tuple[float, float],
                        pos2: Tuple[float, float]) -> float:
        euclidean_dist = math.sqrt(
            (pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2
        )
        if euclidean_dist < 0.01:
            return 0.0

        n_samples = max(int(euclidean_dist * 4), 10)
        total_cost = 0.0

        for i in range(n_samples):
            t = (i + 0.5) / n_samples
            x = pos1[0] + t * (pos2[0] - pos1[0])
            y = pos1[1] + t * (pos2[1] - pos1[1])

            cost = self.cost_field.query((x, y)) * self.base_cost
            total_cost += cost * (euclidean_dist / n_samples)

        return total_cost

    def get_heuristic(self, pos: Tuple[float, float],
                     goal: Tuple[float, float]) -> float:
        return self.compute_distance(pos, goal)

    def update_from_observation(self, position: Tuple[float, float],
                                observation: Dict[str, Any]) -> None:
        # 访问位置积累质量
        current_mass = self.mass_field.query(position)
        self.mass_field.add_sample(position, current_mass + 0.01)

        # 障碍物增加成本
        if 'obstacles' in observation:
            for obs in observation['obstacles']:
                for _ in range(3):
                    angle = 2 * math.pi * np.random.random()
                    dist = 0.2 + 0.3 * np.random.random()
                    sample_pos = (
                        obs[0] + dist * math.cos(angle),
                        obs[1] + dist * math.sin(angle)
                    )
                    current_cost = self.cost_field.query(sample_pos)
                    self.cost_field.add_sample(sample_pos, min(5.0, current_cost + 0.3))

        # 归一化质量
        # 注意：连续场的归一化需要积分，这里简化为采样点求和
        total_mass = sum(self.mass_field.samples.values())
        if total_mass > 0:
            for pos in list(self.mass_field.samples.keys()):
                self.mass_field.samples[pos] /= total_mass

    def predict_next_state(self, position: Tuple[float, float],
                           observation: Dict[str, Any]) -> Dict[str, Any]:
        goal = observation.get('goal_position')
        obstacles = observation.get('obstacles', [])

        num_directions = 16
        best_pos = position
        best_score = float('-inf')
        step_size = observation.get('step_size', 0.5)

        for i in range(num_directions):
            angle = 2 * math.pi * i / num_directions
            nx = position[0] + step_size * math.cos(angle)
            ny = position[1] + step_size * math.sin(angle)
            candidate = (nx, ny)

            if self._check_collision(candidate, obstacles):
                continue

            cost_penalty = -self.cost_field.query(candidate)
            mass_bonus = self.mass_field.query(candidate)

            goal_bonus = 0.0
            if goal:
                dist_to_goal = math.sqrt((nx-goal[0])**2 + (ny-goal[1])**2)
                goal_bonus = -dist_to_goal * 0.1

            score = cost_penalty + mass_bonus + goal_bonus

            if score > best_score:
                best_score = score
                best_pos = candidate

        return {
            'predicted_position': best_pos,
            'predicted_cost': self.compute_distance(position, best_pos),
            'predicted_uncertainty': self.cost_field.query(best_pos) / max(1.0, max(self.cost_field.samples.values()) if self.cost_field.samples else 1.0),
            'passable': best_pos != position,
            'wasserstein_score': best_score,
            'predicted_cost_field': self.cost_field.query(best_pos),
            'predicted_mass': self.mass_field.query(best_pos),
        }

    def compute_validity(self, position: Tuple[float, float],
                        observation: Dict[str, Any],
                        actual: Optional[Dict[str, Any]] = None) -> float:
        cost = self.cost_field.query(position)
        max_cost = max(self.cost_field.samples.values()) if self.cost_field.samples else 1.0
        return max(0.0, 1.0 - cost / max_cost)
