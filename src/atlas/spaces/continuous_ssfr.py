"""
Continuous SSFR - 连续状态SSFR

核心变化：
1. 位置: Tuple[int, int] → Tuple[float, float]
2. 结构池: 基于连续坐标的结构发现
3. 空间: 使用ContinuousCognitiveSpace
4. 预测: 连续路径积分替代网格邻居
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import uuid
import time
import math

from ..core.ssfr_enhanced import (
    StructureHypothesis, StructurePool, ValidationResult
)
from .continuous import (
    ContinuousCognitiveSpace,
    ContinuousRicciSpace,
    ContinuousFisherSpace,
    ContinuousWassersteinSpace,
    ContinuousEuclideanSpace,
)


# ============================================================================
# 1. 连续结构假设
# ============================================================================

@dataclass
class ContinuousStructureHypothesis:
    """连续结构假设"""
    id: str
    name: str
    representations: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    validation_history: List[Dict] = field(default_factory=list)
    computation_cost: float = 1.0
    maintenance_cost: float = 0.1
    created_at: float = 0.0
    last_used: float = 0.0
    usage_count: int = 0
    _fitness_cache: Optional[float] = None
    _fitness_stale: bool = True

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]

    @property
    def fitness(self) -> float:
        if self._fitness_stale or self._fitness_cache is None:
            self._fitness_cache = self._compute_fitness()
            self._fitness_stale = False
        return self._fitness_cache

    def _compute_fitness(self) -> float:
        if not self.validation_history:
            return 0.5

        recent = self.validation_history[-10:]
        weights = np.exp(np.linspace(-1, 0, len(recent)))
        errors = np.array([v.get('prediction_error', 1.0) for v in recent])
        weighted_error = np.sum(weights * errors) / np.sum(weights)
        accuracy = 1.0 / (1.0 + weighted_error)
        total_cost = self.computation_cost + self.maintenance_cost * self.usage_count
        return accuracy / (1.0 + total_cost * 0.01)

    def validate(self, observation: Dict, actual: Dict, timestamp: float = 0.0) -> Dict:
        prediction = self._generate_prediction(observation)
        error = self._compute_prediction_error(prediction, actual)

        result = {
            'hypothesis_id': self.id,
            'prediction_error': error,
            'fitness': self.fitness,
            'timestamp': timestamp
        }

        self.validation_history.append(result)
        self._fitness_stale = True
        self.last_used = timestamp
        self.usage_count += 1
        return result

    def _generate_prediction(self, observation: Dict) -> Dict:
        predictions = {}
        for space_name, rep in self.representations.items():
            if isinstance(rep, dict):
                if 'space_instance' in rep and 'observation' in rep:
                    space = rep['space_instance']
                    obs = rep['observation']
                    try:
                        pred = space.predict_next_state(
                            observation.get('position', (0.0, 0.0)),
                            obs
                        )
                        predictions[space_name] = pred
                    except Exception:
                        predictions[space_name] = self._fallback_prediction(obs)
                elif 'predictor' in rep and callable(rep['predictor']):
                    predictions[space_name] = rep['predictor'](observation)
                elif 'prediction' in rep:
                    predictions[space_name] = rep['prediction']
                else:
                    predictions[space_name] = self._fallback_prediction(observation)
            else:
                predictions[space_name] = self._fallback_prediction(observation)

        if len(predictions) > 1:
            predictions['_fused'] = self._fuse_predictions(predictions)
        return predictions

    def _fallback_prediction(self, observation: Dict) -> Dict:
        position = observation.get('position', (0.0, 0.0))
        goal = observation.get('goal_position')

        if goal:
            dx = goal[0] - position[0]
            dy = goal[1] - position[1]
            dist = math.sqrt(dx**2 + dy**2)
            if dist > 0.01:
                # 向目标方向移动一小步
                step = 0.5
                predicted_pos = (
                    position[0] + step * dx / dist,
                    position[1] + step * dy / dist
                )
            else:
                predicted_pos = position
        else:
            predicted_pos = position

        return {
            'predicted_position': predicted_pos,
            'predicted_cost': 1.0,
            'predicted_uncertainty': 0.5,
            'passable': True,
        }

    def _fuse_predictions(self, predictions: Dict[str, Dict]) -> Dict:
        positions = []
        weights = []
        for space_name, pred in predictions.items():
            if space_name.startswith('_'):
                continue
            if 'predicted_position' in pred:
                positions.append(pred['predicted_position'])
                unc = pred.get('predicted_uncertainty', 0.5)
                weights.append(1.0 / (1.0 + unc))

        if not positions:
            return self._fallback_prediction({})

        weights = np.array(weights)
        weights /= weights.sum()

        # 加权平均位置
        avg_x = sum(p[0] * w for p, w in zip(positions, weights))
        avg_y = sum(p[1] * w for p, w in zip(positions, weights))
        confidence = max(weights)

        return {
            'predicted_position': (avg_x, avg_y),
            'predicted_cost': 1.0,
            'predicted_uncertainty': 0.5,
            'passable': confidence > 0.3,
            'fusion_confidence': float(confidence),
        }

    def _compute_prediction_error(self, prediction: Dict, actual: Dict) -> float:
        errors = []
        for key in actual.keys():
            if key in prediction:
                pred_val = prediction[key]
                actual_val = actual[key]
                if isinstance(pred_val, (int, float)) and isinstance(actual_val, (int, float)):
                    errors.append(abs(pred_val - actual_val))
                elif isinstance(pred_val, tuple) and isinstance(actual_val, tuple):
                    # 位置误差
                    if len(pred_val) == len(actual_val) == 2:
                        dist = math.sqrt(
                            (pred_val[0] - actual_val[0])**2 +
                            (pred_val[1] - actual_val[1])**2
                        )
                        errors.append(min(dist / 10.0, 1.0))
        return np.mean(errors) if errors else 1.0


# ============================================================================
# 2. 连续结构池
# ============================================================================

class ContinuousStructurePool:
    """连续结构竞争池"""

    def __init__(self, max_structures: int = 100):
        self.structures: Dict[str, ContinuousStructureHypothesis] = {}
        self.max_structures = max_structures
        self.competition_history: List[Dict] = []
        self.generation = 0
        self._sorted_cache = None
        self._sorted_stale = True

    def add(self, hypothesis: ContinuousStructureHypothesis) -> None:
        self.structures[hypothesis.id] = hypothesis
        self._sorted_stale = True

        if len(self.structures) > self.max_structures * 1.2:
            self._eliminate_weakest(n=len(self.structures) - self.max_structures)

    def compete(self, observation: Dict, actual: Dict,
                timestamp: float = 0.0) -> Tuple[Optional[ContinuousStructureHypothesis], List]:
        """结构竞争"""
        if not self.structures:
            return None, []

        results = []
        top_structures = self._get_top_structures(n=min(20, len(self.structures)))

        for hyp in top_structures:
            result = hyp.validate(observation, actual, timestamp)
            results.append((hyp, result))

        results.sort(key=lambda x: x[1].get('fitness', 0), reverse=True)
        winner = results[0][0] if results else None

        self.competition_history.append({
            'timestamp': timestamp,
            'winner_id': winner.id if winner else None,
            'winner_fitness': results[0][1].get('fitness', 0) if results else 0,
            'num_competitors': len(results),
        })

        return winner, results

    def _get_top_structures(self, n: int = 20) -> List[ContinuousStructureHypothesis]:
        if self._sorted_stale or self._sorted_cache is None:
            self._sorted_cache = sorted(
                self.structures.values(),
                key=lambda h: h.fitness,
                reverse=True
            )
            self._sorted_stale = False
        return self._sorted_cache[:n]

    def _eliminate_weakest(self, n: int = 1) -> None:
        if len(self.structures) <= n:
            return

        sorted_ids = sorted(
            self.structures.keys(),
            key=lambda sid: self.structures[sid].fitness
        )

        for sid in sorted_ids[:n]:
            del self.structures[sid]

        self._sorted_stale = True

    def get_best(self, n: int = 5) -> List[ContinuousStructureHypothesis]:
        return self._get_top_structures(n=n)

    def get_statistics(self) -> Dict[str, Any]:
        if not self.structures:
            return {'num_structures': 0}

        fitnesses = [h.fitness for h in self.structures.values()]
        return {
            'num_structures': len(self.structures),
            'generation': self.generation,
            'avg_fitness': float(np.mean(fitnesses)),
            'max_fitness': float(np.max(fitnesses)),
            'min_fitness': float(np.min(fitnesses)),
            'std_fitness': float(np.std(fitnesses)),
            'total_competitions': len(self.competition_history),
        }


# ============================================================================
# 3. 连续SSFR
# ============================================================================

class ContinuousSSFR:
    """连续SSFR - 完全移除离散网格"""

    def __init__(self, space_names: Optional[List[str]] = None,
                 max_structures: int = 100,
                 evolution_interval: int = 10,
                 reuse_threshold: float = 0.85):

        self.evolution_interval = evolution_interval
        self.reuse_threshold = reuse_threshold

        # 初始化连续空间
        space_names = space_names or ['ricci', 'fisher', 'wasserstein']
        self.spaces = self._init_spaces(space_names)

        # 结构池
        self.structure_pool = ContinuousStructurePool(max_structures=max_structures)

        # 当前激活的结构
        self.active_structures: List[ContinuousStructureHypothesis] = []

        # 时间步
        self.step_count = 0

        # 统计
        self.perception_history: List[Dict] = []
        self.competition_history: List[Dict] = []

        # 性能计时
        self.timing = {
            'perceive': [],
            'compete': [],
            'evolve': [],
            'total': [],
        }

        # 复用统计
        self.reuse_stats = {
            'total_perceptions': 0,
            'reused_count': 0,
            'new_generated_count': 0,
            'reuse_rate_history': [],
        }

    def _init_spaces(self, space_names: List[str]) -> Dict[str, ContinuousCognitiveSpace]:
        """初始化连续空间"""
        spaces = {}
        space_classes = {
            'ricci': ContinuousRicciSpace,
            'fisher': ContinuousFisherSpace,
            'wasserstein': ContinuousWassersteinSpace,
            'euclidean': ContinuousEuclideanSpace,
        }

        for name in space_names:
            if name in space_classes:
                try:
                    spaces[name] = space_classes[name]()
                except Exception as e:
                    print(f"Warning: Could not create space {name}: {e}")

        return spaces

    def perceive(self, position: Tuple[float, float],
                 observation: Dict,
                 active_space_name: Optional[str] = None) -> List[ContinuousStructureHypothesis]:
        """感知：从观测中提取结构（连续版本）"""
        start = time.time()

        # 1. 更新空间
        if active_space_name and active_space_name in self.spaces:
            try:
                self.spaces[active_space_name].update_from_observation(position, observation)
            except Exception:
                pass
        else:
            for space in self.spaces.values():
                try:
                    space.update_from_observation(position, observation)
                except Exception:
                    pass

        # 2. 尝试复用现有结构
        self.reuse_stats['total_perceptions'] += 1
        reused = self._try_reuse_structure(position, observation, active_space_name)

        if reused:
            self.reuse_stats['reused_count'] += 1
            self.reuse_stats['reuse_rate_history'].append(1.0)
            self.active_structures = [reused]
            self.timing['perceive'].append(time.time() - start)
            return [reused]

        # 3. 生成新假设
        self.reuse_stats['new_generated_count'] += 1
        self.reuse_stats['reuse_rate_history'].append(0.0)
        hypotheses = []

        for space_name in self.spaces.keys():
            if active_space_name and space_name != active_space_name:
                continue

            hyp = self._generate_hypothesis_from_space(space_name, observation)
            if hyp:
                hypotheses.append(hyp)

        # 4. 添加到结构池
        for hyp in hypotheses:
            self.structure_pool.add(hyp)

        self.active_structures = hypotheses

        self.perception_history.append({
            'step': self.step_count,
            'position': position,
            'num_hypotheses': len(hypotheses),
            'active_space': active_space_name,
        })

        self.timing['perceive'].append(time.time() - start)
        return hypotheses

    def _try_reuse_structure(self, position: Tuple[float, float],
                              observation: Dict,
                              active_space_name: Optional[str] = None) -> Optional[ContinuousStructureHypothesis]:
        """尝试复用现有结构（连续版本）"""
        if not self.structure_pool.structures:
            return None

        current_features = self._extract_observation_features(observation)
        best_match = None
        best_similarity = 0.0

        for struct in self.structure_pool.structures.values():
            # 检查空间类型匹配
            if active_space_name:
                struct_space = struct.context.get('space_type', '')
                if struct_space != active_space_name:
                    continue

            # 检查位置接近（欧氏距离）
            struct_pos = struct.context.get('observation', {}).get('position', None)
            if struct_pos:
                dist = math.sqrt(
                    (position[0] - struct_pos[0])**2 +
                    (position[1] - struct_pos[1])**2
                )
                if dist > 3.0:  # 超过3米不算接近
                    continue

            # 计算特征相似度
            similarity = self._compute_observation_similarity(current_features, struct)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = struct

        if best_match and best_similarity >= self.reuse_threshold:
            best_match.last_used = self.step_count
            best_match.usage_count += 1
            best_match._fitness_stale = True
            return best_match

        return None

    def _extract_observation_features(self, observation: Dict) -> Dict:
        """提取观测特征"""
        features = {
            'position': observation.get('position', (0.0, 0.0)),
            'goal_position': observation.get('goal_position', None),
        }

        obstacles = observation.get('obstacles', [])
        if obstacles:
            pos = features['position']
            features['num_obstacles'] = len(obstacles)
            if obstacles:
                min_dist = min(
                    math.sqrt((pos[0] - ox)**2 + (pos[1] - oy)**2)
                    for ox, oy in obstacles
                )
                features['nearest_obstacle_dist'] = min_dist
            else:
                features['nearest_obstacle_dist'] = 999
        else:
            features['num_obstacles'] = 0
            features['nearest_obstacle_dist'] = 999

        return features

    def _compute_observation_similarity(self, current_features: Dict,
                                         struct: ContinuousStructureHypothesis) -> float:
        """计算观测相似度"""
        similarities = []

        # 位置相似度
        struct_pos = struct.context.get('observation', {}).get('position', None)
        if struct_pos and 'position' in current_features:
            curr_pos = current_features['position']
            dist = math.sqrt(
                (curr_pos[0] - struct_pos[0])**2 +
                (curr_pos[1] - struct_pos[1])**2
            )
            pos_similarity = max(0.0, 1.0 - dist / 5.0)
            similarities.append(pos_similarity)

        # 目标位置相似度
        struct_goal = struct.context.get('observation', {}).get('goal_position', None)
        curr_goal = current_features.get('goal_position', None)
        if struct_goal and curr_goal:
            goal_dist = math.sqrt(
                (curr_goal[0] - struct_goal[0])**2 +
                (curr_goal[1] - struct_goal[1])**2
            )
            goal_similarity = max(0.0, 1.0 - goal_dist / 10.0)
            similarities.append(goal_similarity)

        # 障碍物数量相似度
        struct_obs = struct.context.get('observation', {}).get('obstacles', [])
        curr_num_obs = current_features.get('num_obstacles', 0)
        struct_num_obs = len(list(struct_obs)) if struct_obs else 0
        obs_diff = abs(curr_num_obs - struct_num_obs)
        obs_similarity = max(0.0, 1.0 - obs_diff / 5.0)
        similarities.append(obs_similarity)

        if not similarities:
            return 0.0

        weights = [0.4, 0.3, 0.2][:len(similarities)]
        weighted_sum = sum(s * w for s, w in zip(similarities, weights))
        total_weight = sum(weights)

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _generate_hypothesis_from_space(self, space_name: str,
                                         observation: Dict) -> Optional[ContinuousStructureHypothesis]:
        """从空间生成假设"""
        space = self.spaces.get(space_name)

        # 提取特征
        features = self._extract_structure_features(space_name, observation)

        # 生成预测
        prediction = None
        if space:
            try:
                position = observation.get('position', (0.0, 0.0))
                prediction = space.predict_next_state(position, observation)
            except Exception:
                pass

        hyp = ContinuousStructureHypothesis(
            id=str(uuid.uuid4())[:8],
            name=f"{space_name}_struct",
            representations={
                space_name: {
                    'space_instance': space,
                    'observation': observation,
                    'features': features,
                    'prediction': prediction,
                }
            },
            context={
                'space_type': space_name,
                'features': features,
                'observation': observation,
                'has_prediction': prediction is not None,
            },
            created_at=self.step_count
        )

        return hyp

    def _extract_structure_features(self, space_name: str,
                                     observation: Dict) -> Dict:
        """提取结构特征"""
        features = {
            'type': 'unknown',
            'positions': [],
            'uncertainty_pattern': None,
        }

        # 从观测中提取
        if 'uncertainty' in observation:
            features['uncertainty_pattern'] = {
                'value': observation['uncertainty'],
            }

        return features

    def compete(self, observation: Dict, actual: Dict) -> Optional[ContinuousStructureHypothesis]:
        """竞争：选择最优结构"""
        start = time.time()

        winner, results = self.structure_pool.compete(
            observation, actual, self.step_count
        )

        if winner:
            self.competition_history.append({
                'step': self.step_count,
                'winner_id': winner.id,
                'num_competitors': len(results),
            })

        self.timing['compete'].append(time.time() - start)
        return winner

    def evolve(self) -> List[ContinuousStructureHypothesis]:
        """演化"""
        start = time.time()

        new_structures = []
        parents = self.structure_pool.get_best(n=5)

        # 变异
        for parent in parents:
            # 简化：创建新假设作为变异
            for space_name in self.spaces.keys():
                hyp = self._generate_hypothesis_from_space(space_name, parent.context.get('observation', {}))
                if hyp:
                    self.structure_pool.add(hyp)
                    new_structures.append(hyp)

        self.timing['evolve'].append(time.time() - start)
        return new_structures

    def step(self, position: Tuple[float, float],
             observation: Dict,
             actual: Optional[Dict] = None) -> Dict[str, Any]:
        """单步执行"""
        total_start = time.time()

        # 1. 感知
        hypotheses = self.perceive(position, observation)

        # 2. 竞争
        winner = None
        if actual:
            winner = self.compete(observation, actual)

        # 3. 演化
        new_structures = []
        if self.step_count % self.evolution_interval == 0:
            new_structures = self.evolve()

        self.step_count += 1
        self.timing['total'].append(time.time() - total_start)

        return {
            'hypotheses': [h.id for h in hypotheses],
            'winner_id': winner.id if winner else None,
            'new_structures': [s.id for s in new_structures],
            'pool_stats': self.structure_pool.get_statistics(),
        }

    def get_best_structures(self, n: int = 5) -> List[ContinuousStructureHypothesis]:
        """获取最佳结构"""
        return self.structure_pool.get_best(n)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'step_count': self.step_count,
            'num_spaces': len(self.spaces),
            'pool_stats': self.structure_pool.get_statistics(),
            'num_perceptions': len(self.perception_history),
            'num_competitions': len(self.competition_history),
        }

        total = self.reuse_stats['total_perceptions']
        if total > 0:
            stats['reuse_rate'] = self.reuse_stats['reused_count'] / total
            stats['new_generation_rate'] = self.reuse_stats['new_generated_count'] / total
        stats['total_reused'] = self.reuse_stats['reused_count']
        stats['total_new_generated'] = self.reuse_stats['new_generated_count']

        for key, times in self.timing.items():
            if times:
                stats[f'avg_{key}_time_ms'] = np.mean(times) * 1000
                stats[f'max_{key}_time_ms'] = np.max(times) * 1000

        return stats
