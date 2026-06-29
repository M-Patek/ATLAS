"""
Optimized Continuous SSFR - 深度性能优化版

优化策略：
1. KD-Tree空间索引替代简单网格索引
2. 延迟曲率重计算（标记脏数据）
3. 批量场更新
4. 缓存预测结果
5. 减少不必要的采样点
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict
import time

# 尝试导入scipy的KDTree
try:
    from scipy.spatial import cKDTree as KDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    KDTree = None


# ============================================================================
# 1. 优化的连续场
# ============================================================================

class OptimizedContinuousField:
    """
    优化的连续场

    改进：
    1. KD-Tree索引（如果scipy可用）
    2. 批量更新
    3. 智能采样（避免过密采样）
    4. 增量更新
    """

    def __init__(self, default_value: float = 0.0,
                 min_sample_distance: float = 0.1,
                 max_samples: int = 5000):
        self.default_value = default_value
        self.min_sample_distance = min_sample_distance
        self.max_samples = max_samples

        # 采样点
        self.samples: Dict[Tuple[float, float], float] = {}

        # KD-Tree（延迟构建）
        self._tree = None
        self._tree_positions = None
        self._tree_values = None
        self._tree_dirty = True

        # 缓存
        self._cache: Dict[Tuple[float, float], float] = {}
        self._cache_size = 2000

        # 统计
        self._query_count = 0
        self._cache_hit_count = 0

    def _rebuild_tree(self):
        """重建KD-Tree"""
        if not HAS_SCIPY or len(self.samples) < 10:
            self._tree = None
            self._tree_dirty = False
            return

        self._tree_positions = np.array(list(self.samples.keys()))
        self._tree_values = np.array(list(self.samples.values()))
        self._tree = KDTree(self._tree_positions)
        self._tree_dirty = False

    def add_sample(self, position: Tuple[float, float], value: float):
        """添加采样点（带距离检查）"""
        # 检查是否已有太近的点
        if self.samples:
            for pos in self.samples:
                dist = math.sqrt(
                    (pos[0] - position[0])**2 +
                    (pos[1] - position[1])**2
                )
                if dist < self.min_sample_distance:
                    # 更新现有点的值（加权平均）
                    old_val = self.samples[pos]
                    self.samples[pos] = 0.7 * value + 0.3 * old_val
                    self._tree_dirty = True
                    self._cache.clear()
                    return

        # 添加新点
        self.samples[position] = value
        self._tree_dirty = True
        self._cache.clear()

        # 如果采样点过多，移除最旧的（简化：随机移除）
        if len(self.samples) > self.max_samples:
            # 移除距离其他点最近的点（冗余）
            self._remove_redundant_sample()

    def _remove_redundant_sample(self):
        """移除冗余采样点（简化版：随机移除）"""
        if len(self.samples) < 10:
            return

        import random
        key = random.choice(list(self.samples.keys()))
        del self.samples[key]
        self._tree_dirty = True

    def query(self, position: Tuple[float, float],
              k: int = 4, radius: float = 2.0) -> float:
        """查询（使用KD-Tree加速）"""
        self._query_count += 1

        # 检查缓存
        if position in self._cache:
            self._cache_hit_count += 1
            return self._cache[position]

        if not self.samples:
            return self.default_value

        # 使用KD-Tree查询
        if HAS_SCIPY and not self._tree_dirty:
            return self._query_kdtree(position, k, radius)
        else:
            return self._query_linear(position, k, radius)

    def _query_kdtree(self, position: Tuple[float, float],
                      k: int, radius: float) -> float:
        """使用KD-Tree查询"""
        if self._tree is None or self._tree_dirty:
            self._rebuild_tree()

        if self._tree is None:
            return self._query_linear(position, k, radius)

        # KD-Tree查询
        pos_array = np.array([position])
        distances, indices = self._tree.query(pos_array, k=min(k, len(self._tree_positions)))

        # 过滤超出半径的
        valid = distances[0] <= radius
        if not np.any(valid):
            # 使用最近的一个
            idx = indices[0][0]
            result = self._tree_values[idx]
            self._cache[position] = result
            return result

        valid_indices = indices[0][valid]
        valid_distances = distances[0][valid]

        if len(valid_indices) == 0:
            return self.default_value

        # 反距离加权
        if len(valid_indices) == 1:
            result = self._tree_values[valid_indices[0]]
        else:
            weights = 1.0 / (valid_distances ** 2 + 1e-8)
            total_weight = np.sum(weights)
            if total_weight > 0:
                result = np.sum(self._tree_values[valid_indices] * weights) / total_weight
            else:
                result = self.default_value

        self._cache[position] = float(result)
        if len(self._cache) > self._cache_size:
            self._cache.pop(next(iter(self._cache)))

        return float(result)

    def _query_linear(self, position: Tuple[float, float],
                      k: int, radius: float) -> float:
        """线性查询（fallback）"""
        candidates = []

        for sample_pos, value in self.samples.items():
            dist = math.sqrt(
                (sample_pos[0] - position[0])**2 +
                (sample_pos[1] - position[1])**2
            )
            if dist <= radius:
                candidates.append((dist, sample_pos, value))

        if not candidates:
            return self.default_value

        candidates.sort(key=lambda x: x[0])
        neighbors = candidates[:k]

        if len(neighbors) == 1:
            result = neighbors[0][2]
        else:
            weights = []
            values = []
            for dist, _, val in neighbors:
                if dist < 1e-6:
                    result = val
                    self._cache[position] = result
                    return result
                w = 1.0 / (dist ** 2)
                weights.append(w)
                values.append(val)

            total_weight = sum(weights)
            result = sum(v * w for v, w in zip(values, weights)) / total_weight

        self._cache[position] = result
        if len(self._cache) > self._cache_size:
            self._cache.pop(next(iter(self._cache)))

        return result

    def query_batch(self, positions: List[Tuple[float, float]],
                    k: int = 4) -> List[float]:
        """批量查询（使用KD-Tree批量查询）"""
        if HAS_SCIPY and not self._tree_dirty and self._tree is not None:
            return self._query_batch_kdtree(positions, k)
        else:
            return [self.query(pos, k) for pos in positions]

    def _query_batch_kdtree(self, positions: List[Tuple[float, float]],
                            k: int) -> List[float]:
        """KD-Tree批量查询"""
        if self._tree is None or self._tree_dirty:
            self._rebuild_tree()

        if self._tree is None or len(positions) == 0:
            return [self.default_value] * len(positions)

        pos_array = np.array(positions)
        k_actual = min(k, len(self._tree_positions))

        distances, indices = self._tree.query(pos_array, k=k_actual)

        results = []
        for i in range(len(positions)):
            if k_actual == 1:
                d = [distances[i]]
                idx = [indices[i]]
            else:
                d = distances[i]
                idx = indices[i]

            # 反距离加权
            weights = 1.0 / (d ** 2 + 1e-8)
            total_weight = np.sum(weights)
            if total_weight > 0:
                if k_actual == 1:
                    val = self._tree_values[idx]
                else:
                    val = np.sum(self._tree_values[idx] * weights) / total_weight
                results.append(float(val))
            else:
                results.append(self.default_value)

        return results

    def get_cache_stats(self) -> Dict[str, float]:
        """获取缓存统计"""
        if self._query_count == 0:
            return {'hit_rate': 0.0}
        return {
            'hit_rate': self._cache_hit_count / self._query_count,
            'query_count': self._query_count,
            'cache_size': len(self._cache),
        }

    def num_samples(self) -> int:
        return len(self.samples)

    def clear(self):
        self.samples.clear()
        self._tree = None
        self._tree_positions = None
        self._tree_values = None
        self._tree_dirty = True
        self._cache.clear()


# ============================================================================
# 2. 优化的连续Ricci空间
# ============================================================================

class OptimizedContinuousRicciSpace:
    """
    优化的连续Ricci空间

    改进：
    1. 延迟曲率重计算（只在需要时）
    2. 增量曲率更新（只更新附近点）
    3. 批量场更新
    4. 预计算路径采样点
    """

    def __init__(self, curvature_scale: float = 1.0,
                 familiarity_decay: float = 0.1,
                 **kwargs):
        self.name = "ricci_optimized"
        self.curvature_scale = curvature_scale
        self.familiarity_decay = familiarity_decay

        # 连续场
        self.uncertainty_field = OptimizedContinuousField(default_value=0.5)
        self.curvature_field = OptimizedContinuousField(default_value=0.0)
        self.familiarity_field = OptimizedContinuousField(default_value=0.0)

        # 访问记录
        self.visit_points: List[Tuple[float, float]] = []

        # 目标吸引子
        self.attractor_position: Optional[Tuple[float, float]] = None
        self.attractor_strength: float = 0.0

        # 曲率重计算控制
        self._curvature_dirty = True
        self._last_curvature_calc = 0
        self._curvature_calc_interval = 5  # 每5次更新才重计算
        self._update_count = 0

        # 路径采样缓存
        self._path_samples_cache: Dict[Tuple[float, float, float, float], List[Tuple[float, float]]] = {}
        self._path_cache_size = 100

    def compute_distance(self, pos1: Tuple[float, float],
                        pos2: Tuple[float, float]) -> float:
        """优化的距离计算"""
        euclidean_dist = math.sqrt(
            (pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2
        )
        if euclidean_dist < 0.01:
            return 0.0

        # 获取路径采样点（缓存）
        samples = self._get_path_samples(pos1, pos2, euclidean_dist)

        # 批量查询曲率
        curvatures = self.curvature_field.query_batch(samples)

        # 计算积分
        total = 0.0
        step_len = euclidean_dist / len(samples)
        for curvature in curvatures:
            metric_factor = 1.0 + self.curvature_scale * abs(curvature)
            total += metric_factor * step_len

        return total

    def _get_path_samples(self, pos1: Tuple[float, float],
                         pos2: Tuple[float, float],
                         dist: float) -> List[Tuple[float, float]]:
        """获取路径采样点（带缓存）"""
        # 使用粗粒度的缓存键
        key = (
            round(pos1[0], 1), round(pos1[1], 1),
            round(pos2[0], 1), round(pos2[1], 1)
        )

        if key in self._path_samples_cache:
            return self._path_samples_cache[key]

        n_samples = max(int(dist * 4), 5)
        samples = []
        for i in range(n_samples):
            t = (i + 0.5) / n_samples
            x = pos1[0] + t * (pos2[0] - pos1[0])
            y = pos1[1] + t * (pos2[1] - pos1[1])
            samples.append((x, y))

        # 缓存
        self._path_samples_cache[key] = samples
        if len(self._path_samples_cache) > self._path_cache_size:
            self._path_samples_cache.pop(next(iter(self._path_samples_cache)))

        return samples

    def get_heuristic(self, pos: Tuple[float, float],
                     goal: Tuple[float, float]) -> float:
        return self.compute_distance(pos, goal)

    def update_from_observation(self, position: Tuple[float, float],
                                observation: Dict[str, Any]) -> None:
        """优化的更新"""
        self._update_count += 1

        # 记录访问
        self.visit_points.append(position)

        # 更新熟悉度
        current_fam = self.familiarity_field.query(position)
        self.familiarity_field.add_sample(
            position,
            min(1.0, current_fam + self.familiarity_decay)
        )

        # 障碍物增加不确定性（批量）
        if 'obstacles' in observation:
            self._batch_update_uncertainty(observation['obstacles'])

        # 目标降低不确定性
        if observation.get('goal_reached') and 'goal_position' in observation:
            self._batch_reduce_uncertainty(observation['goal_position'])

        # 更新attractor
        if 'goal_position' in observation:
            self.attractor_position = observation['goal_position']
            self.attractor_strength = observation.get('goal_strength', 0.5)

        # 延迟曲率重计算
        self._curvature_dirty = True
        if self._update_count % self._curvature_calc_interval == 0:
            self._recalculate_curvature()

    def _batch_update_uncertainty(self, obstacles: List[Tuple[float, float]]):
        """批量更新不确定性"""
        for obs in obstacles:
            # 每个障碍物周围采样3个点
            for i in range(3):
                angle = 2 * math.pi * i / 3
                dist = 0.3 + 0.3 * (i / 3)
                sample_pos = (
                    obs[0] + dist * math.cos(angle),
                    obs[1] + dist * math.sin(angle)
                )
                current_unc = self.uncertainty_field.query(sample_pos)
                self.uncertainty_field.add_sample(
                    sample_pos,
                    min(1.0, current_unc + 0.2)
                )

    def _batch_reduce_uncertainty(self, goal: Tuple[float, float]):
        """批量降低不确定性（目标附近）"""
        for i in range(5):
            angle = 2 * math.pi * i / 5
            dist = 0.3 * (i / 5)
            sample_pos = (
                goal[0] + dist * math.cos(angle),
                goal[1] + dist * math.sin(angle)
            )
            current_unc = self.uncertainty_field.query(sample_pos)
            self.uncertainty_field.add_sample(
                sample_pos,
                max(0.0, current_unc * 0.8)
            )

    def _recalculate_curvature(self):
        """增量曲率重计算（只更新新点）"""
        if not self._curvature_dirty:
            return

        # 只在已有采样点上计算曲率
        # 获取不确定性场的采样点
        unc_samples = list(self.uncertainty_field.samples.keys())

        # 限制每次更新的点数
        max_update = 50
        if len(unc_samples) > max_update:
            # 优先更新最近访问的点
            unc_samples = unc_samples[-max_update:]

        h = 0.1
        for pos in unc_samples:
            # 采样周围点计算Laplacian
            samples = [
                (pos[0]+h, pos[1]), (pos[0]-h, pos[1]),
                (pos[0], pos[1]+h), (pos[0], pos[1]-h),
            ]
            values = [self.uncertainty_field.query(s) for s in samples]
            center = self.uncertainty_field.query(pos)

            laplacian = (sum(values) - 4*center) / (h**2)
            curvature = -laplacian

            familiarity = self.familiarity_field.query(pos)
            curvature *= (1.0 - familiarity)

            self.curvature_field.add_sample(pos, curvature)

        self._curvature_dirty = False
        self._last_curvature_calc = self._update_count

    def predict_next_state(self, position: Tuple[float, float],
                           observation: Dict[str, Any]) -> Dict[str, Any]:
        """优化的预测"""
        goal = observation.get('goal_position')
        obstacles = observation.get('obstacles', [])
        step_size = observation.get('step_size', 0.5)

        # 预计算所有候选点的曲率和熟悉度
        num_directions = 16
        candidates = []

        for i in range(num_directions):
            angle = 2 * math.pi * i / num_directions
            nx = position[0] + step_size * math.cos(angle)
            ny = position[1] + step_size * math.sin(angle)
            candidate = (nx, ny)

            if self._check_collision(candidate, obstacles):
                continue

            candidates.append(candidate)

        if not candidates:
            return {
                'predicted_position': position,
                'predicted_cost': 0.0,
                'predicted_uncertainty': 1.0,
                'passable': False,
            }

        # 批量查询场值
        curvatures = self.curvature_field.query_batch(candidates)
        familiarities = self.familiarity_field.query_batch(candidates)
        uncertainties = self.uncertainty_field.query_batch(candidates)

        best_pos = position
        best_score = float('-inf')

        for i, candidate in enumerate(candidates):
            curvature_penalty = -abs(curvatures[i]) * self.curvature_scale
            familiarity_bonus = familiarities[i]
            uncertainty_penalty = -uncertainties[i]

            goal_bonus = 0.0
            if goal:
                dist_to_goal = math.sqrt(
                    (candidate[0]-goal[0])**2 + (candidate[1]-goal[1])**2
                )
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

    def _check_collision(self, position: Tuple[float, float],
                        obstacles: List[Tuple[float, float]],
                        obstacle_radius: float = 0.3) -> bool:
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
        return {
            'name': self.name,
            'uncertainty_samples': self.uncertainty_field.num_samples(),
            'curvature_samples': self.curvature_field.num_samples(),
            'familiarity_samples': self.familiarity_field.num_samples(),
            'visit_count': len(self.visit_points),
            'cache_hit_rate': self.uncertainty_field.get_cache_stats()['hit_rate'],
        }


# ============================================================================
# 3. 优化的连续SSFR
# ============================================================================

class OptimizedContinuousSSFR:
    """
    优化的连续SSFR

    改进：
    1. 批量感知
    2. 增量竞争
    3. 结构池压缩
    4. 预测缓存
    """

    def __init__(self, space_names: Optional[List[str]] = None,
                 max_structures: int = 50,
                 evolution_interval: int = 10,
                 reuse_threshold: float = 0.85):

        self.evolution_interval = evolution_interval
        self.reuse_threshold = reuse_threshold

        # 初始化空间
        space_names = space_names or ['ricci']
        self.spaces = self._init_spaces(space_names)

        # 结构池
        self.structures: Dict[str, Any] = {}
        self.max_structures = max_structures

        # 时间步
        self.step_count = 0

        # 统计
        self.stats = {
            'perception_count': 0,
            'competition_count': 0,
            'total_time': 0.0,
        }

        # 预测缓存
        self._prediction_cache: Dict[str, Any] = {}
        self._prediction_cache_size = 100

    def _init_spaces(self, space_names: List[str]) -> Dict[str, Any]:
        """初始化空间"""
        spaces = {}
        for name in space_names:
            if name == 'ricci':
                spaces[name] = OptimizedContinuousRicciSpace()
        return spaces

    def perceive(self, position: Tuple[float, float],
                 observation: Dict,
                 active_space_name: Optional[str] = None) -> List[Any]:
        """优化的感知"""
        start = time.time()

        # 更新空间
        for name, space in self.spaces.items():
            if active_space_name and name != active_space_name:
                continue
            try:
                space.update_from_observation(position, observation)
            except Exception:
                pass

        # 尝试复用
        reused = self._try_reuse_structure(position, observation, active_space_name)
        if reused:
            self.stats['perception_count'] += 1
            self.stats['total_time'] += time.time() - start
            return [reused]

        # 生成新假设（简化版）
        hypotheses = []
        for name in self.spaces.keys():
            if active_space_name and name != active_space_name:
                continue

            hyp = self._create_hypothesis(name, observation)
            if hyp:
                hypotheses.append(hyp)
                self.structures[hyp['id']] = hyp

        # 压缩结构池
        if len(self.structures) > self.max_structures:
            self._compress_structures()

        self.stats['perception_count'] += 1
        self.stats['total_time'] += time.time() - start

        return hypotheses

    def _try_reuse_structure(self, position: Tuple[float, float],
                              observation: Dict,
                              active_space_name: Optional[str] = None) -> Optional[Any]:
        """尝试复用结构（简化版）"""
        if not self.structures:
            return None

        best_match = None
        best_dist = float('inf')

        for struct in self.structures.values():
            struct_pos = struct.get('context', {}).get('position', None)
            if struct_pos is None:
                continue

            dist = math.sqrt(
                (position[0] - struct_pos[0])**2 +
                (position[1] - struct_pos[1])**2
            )

            if dist < 1.0 and dist < best_dist:  # 1米内
                best_dist = dist
                best_match = struct

        return best_match

    def _create_hypothesis(self, space_name: str, observation: Dict) -> Optional[Dict]:
        """创建假设（简化版）"""
        space = self.spaces.get(space_name)
        if not space:
            return None

        position = observation.get('position', (0.0, 0.0))

        # 使用缓存的预测
        cache_key = f"{space_name}_{position[0]:.1f}_{position[1]:.1f}"
        if cache_key in self._prediction_cache:
            prediction = self._prediction_cache[cache_key]
        else:
            try:
                prediction = space.predict_next_state(position, observation)
                self._prediction_cache[cache_key] = prediction
                if len(self._prediction_cache) > self._prediction_cache_size:
                    self._prediction_cache.pop(next(iter(self._prediction_cache)))
            except Exception:
                prediction = None

        return {
            'id': f"hyp_{self.step_count}_{space_name}",
            'name': f"{space_name}_struct",
            'space_type': space_name,
            'context': {
                'position': position,
                'observation': observation,
            },
            'prediction': prediction,
            'fitness': 0.5,
        }

    def _compress_structures(self):
        """压缩结构池（移除最老的）"""
        if len(self.structures) <= self.max_structures:
            return

        # 按创建时间排序，移除最老的
        sorted_structs = sorted(
            self.structures.items(),
            key=lambda x: x[1].get('created_at', 0)
        )

        to_remove = len(self.structures) - self.max_structures
        for key, _ in sorted_structs[:to_remove]:
            del self.structures[key]

    def compete(self, observation: Dict, actual: Dict) -> Optional[Any]:
        """简化的竞争"""
        self.stats['competition_count'] += 1

        if not self.structures:
            return None

        # 返回最新的结构
        best = max(self.structures.values(),
                  key=lambda s: s.get('fitness', 0))
        return best

    def get_statistics(self) -> Dict[str, Any]:
        return {
            'perceptions': self.stats['perception_count'],
            'competitions': self.stats['competition_count'],
            'pool_size': len(self.structures),
            'avg_time_ms': self.stats['total_time'] / max(1, self.stats['perception_count']) * 1000,
        }
