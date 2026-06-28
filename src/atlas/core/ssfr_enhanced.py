"""
SSFR Enhanced - 优化版本

核心优化：
1. 缓存空间实例，避免重复创建
2. 减少每步生成的假设数量（从"每个空间一个"到"按需生成"）
3. 异步演化（不在每步都做）
4. 批量验证（减少 fitness 计算次数）
5. 延迟淘汰（不在每次 add 时淘汰）
6. 预测结果缓存（避免重复调用 predict_next_state）
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict
import uuid
import copy
import time

from ..core.space import CognitiveSpace
from ..core.registry import create_space


# ============================================================================
# 1. 基础数据结构（不变）
# ============================================================================

@dataclass
class ValidationResult:
    """验证结果"""
    hypothesis_id: str
    prediction_error: float
    fitness: float
    timestamp: float

    def __post_init__(self):
        self.composite_score = self.fitness / (1.0 + self.prediction_error)


@dataclass
class StructureHypothesis:
    """结构假设"""
    id: str
    name: str
    representations: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    validation_history: List[ValidationResult] = field(default_factory=list)
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
        errors = np.array([v.prediction_error for v in recent])
        weighted_error = np.sum(weights * errors) / np.sum(weights)
        accuracy = 1.0 / (1.0 + weighted_error)
        total_cost = self.computation_cost + self.maintenance_cost * self.usage_count
        return accuracy / (1.0 + total_cost * 0.01)

    def validate(self, observation: Dict, actual: Dict, timestamp: float = 0.0) -> ValidationResult:
        prediction = self._generate_prediction(observation)
        error = self._compute_prediction_error(prediction, actual)
        result = ValidationResult(
            hypothesis_id=self.id,
            prediction_error=error,
            fitness=self.fitness,
            timestamp=timestamp
        )
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
                    position = obs.get('position', (0, 0))
                    try:
                        pred = space.predict_next_state(position, obs)
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
        position = observation.get('position', (0, 0))
        goal = observation.get('goal_position')
        predicted_pos = position
        if goal:
            dx = np.sign(goal[0] - position[0])
            dy = np.sign(goal[1] - position[1])
            if abs(goal[0] - position[0]) > abs(goal[1] - position[1]):
                predicted_pos = (position[0] + dx, position[1])
            else:
                predicted_pos = (position[0], position[1] + dy)
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

        pos_votes = {}
        for pos, w in zip(positions, weights):
            pos_key = pos
            pos_votes[pos_key] = pos_votes.get(pos_key, 0) + w

        best_pos = max(pos_votes.keys(), key=lambda k: pos_votes[k])
        confidence = pos_votes[best_pos]

        return {
            'predicted_position': best_pos,
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
                elif isinstance(pred_val, np.ndarray) and isinstance(actual_val, np.ndarray):
                    errors.append(np.mean(np.abs(pred_val - actual_val)))
        return np.mean(errors) if errors else 1.0

    def merge_with(self, other: 'StructureHypothesis') -> 'StructureHypothesis':
        merged_reps = copy.deepcopy(self.representations)
        merged_reps.update(other.representations)
        return StructureHypothesis(
            id=str(uuid.uuid4())[:8],
            name=f"merge({self.name},{other.name})",
            representations=merged_reps,
            context={**self.context, **other.context},
            computation_cost=self.computation_cost + other.computation_cost,
            created_at=max(self.created_at, other.created_at)
        )

    def mutate(self, mutation_rate: float = 0.1) -> 'StructureHypothesis':
        mutated_reps = copy.deepcopy(self.representations)
        for space_name in list(mutated_reps.keys()):
            if np.random.random() < mutation_rate:
                rep = mutated_reps[space_name]
                if isinstance(rep, dict) and 'params' in rep:
                    for param_key in rep['params']:
                        if np.random.random() < mutation_rate:
                            rep['params'][param_key] *= (1.0 + np.random.normal(0, 0.1))
        return StructureHypothesis(
            id=str(uuid.uuid4())[:8],
            name=f"mut({self.name})",
            representations=mutated_reps,
            context=self.context,
            computation_cost=self.computation_cost * 1.1,
            created_at=self.created_at
        )


# ============================================================================
# 2. 结构竞争池（优化版）
# ============================================================================

class StructurePool:
    """结构竞争池（优化版）"""

    def __init__(self,
                 max_structures: int = 100,
                 competition_threshold: float = 0.1,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.05,
                 elimination_rate: float = 0.1):
        self.structures: Dict[str, StructureHypothesis] = {}
        self.max_structures = max_structures
        self.competition_threshold = competition_threshold
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elimination_rate = elimination_rate
        self.competition_history: List[Dict] = []
        self.generation = 0
        # 缓存 fitness 排序结果
        self._sorted_cache = None
        self._sorted_stale = True

    def add(self, hypothesis: StructureHypothesis) -> None:
        """添加假设到池中"""
        self.structures[hypothesis.id] = hypothesis
        self._sorted_stale = True

        # 延迟淘汰：只在超过阈值时淘汰
        if len(self.structures) > self.max_structures * 1.2:
            self._eliminate_weakest(n=len(self.structures) - self.max_structures)

    def compete(self,
                observation: Dict,
                actual: Dict,
                timestamp: float = 0.0) -> Tuple[StructureHypothesis, List[Tuple[StructureHypothesis, ValidationResult]]]:
        """结构竞争（优化版）"""
        if not self.structures:
            return None, []

        # 批量验证：只验证前N个最佳结构（不是全部）
        results = []
        top_structures = self._get_top_structures(n=min(20, len(self.structures)))

        for hyp in top_structures:
            result = hyp.validate(observation, actual, timestamp)
            results.append((hyp, result))

        # 排序
        results.sort(key=lambda x: x[1].fitness, reverse=True)

        winner = results[0][0]

        self.competition_history.append({
            'timestamp': timestamp,
            'winner_id': winner.id,
            'winner_fitness': results[0][1].fitness,
            'num_competitors': len(results),
            'avg_fitness': np.mean([r[1].fitness for r in results])
        })

        return winner, results

    def _get_top_structures(self, n: int = 20) -> List[StructureHypothesis]:
        """获取前N个最佳结构（带缓存）"""
        if self._sorted_stale or self._sorted_cache is None:
            self._sorted_cache = sorted(
                self.structures.values(),
                key=lambda h: h.fitness,
                reverse=True
            )
            self._sorted_stale = False
        return self._sorted_cache[:n]

    def evolve(self, timestamp: float = 0.0) -> List[StructureHypothesis]:
        """结构演化（优化版）"""
        new_structures = []

        # 选择优秀结构作为父代
        parents = self._get_top_structures(n=5)

        # 变异
        for parent in parents:
            if np.random.random() < self.mutation_rate:
                child = parent.mutate(self.mutation_rate)
                self.add(child)
                new_structures.append(child)

        # 交叉
        if len(parents) >= 2 and np.random.random() < self.crossover_rate:
            child = parents[0].merge_with(parents[1])
            self.add(child)
            new_structures.append(child)

        # 定期淘汰（每10代）
        if self.generation % 10 == 0 and len(self.structures) > self.max_structures:
            self._eliminate_weakest(n=max(1, int(len(self.structures) * self.elimination_rate)))

        self.generation += 1
        return new_structures

    def _eliminate_weakest(self, n: int = 1) -> None:
        """淘汰最弱的结构"""
        if len(self.structures) <= n:
            return

        sorted_ids = sorted(
            self.structures.keys(),
            key=lambda sid: self.structures[sid].fitness
        )

        for sid in sorted_ids[:n]:
            del self.structures[sid]

        self._sorted_stale = True

    def get_best(self, n: int = 5) -> List[StructureHypothesis]:
        """获取最佳结构"""
        return self._get_top_structures(n=n)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
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
# 3. 多空间联合表示（优化版）
# ============================================================================

class MultiSpaceRepresentation:
    """多空间联合表示（优化版）"""

    def __init__(self,
                 spaces: List[CognitiveSpace],
                 consistency_threshold: float = 0.7):
        self.spaces = {s.name: s for s in spaces}
        self.consistency_threshold = consistency_threshold
        self.weights: Dict[str, float] = {name: 1.0 for name in self.spaces}
        self.consistency_history: List[Dict] = []
        # 缓存编码结果
        self._encode_cache = {}
        self._encode_cache_key = None

    def encode(self, observation: Dict) -> Dict[str, Any]:
        """将观测编码到所有空间（带缓存）"""
        # 简单缓存：基于观测哈希
        cache_key = str(observation.get('position', (0, 0)))
        if self._encode_cache_key == cache_key:
            return self._encode_cache

        representations = {}
        for name, space in self.spaces.items():
            fields = space.get_visualization_fields()
            rep = {
                'fields': fields,
                'statistics': space.get_statistics(),
                'space_type': name,
                'weight': self.weights.get(name, 1.0),
            }
            representations[name] = rep

        self._encode_cache = representations
        self._encode_cache_key = cache_key
        return representations

    def find_consistent_structure(self,
                                   representations: Dict[str, Any],
                                   observation: Dict) -> Optional[StructureHypothesis]:
        """在所有空间中寻找一致的结构"""
        structures_per_space = {}
        for space_name, rep in representations.items():
            features = self._extract_structure_features(rep, observation)
            structures_per_space[space_name] = features

        consistent = self._find_cross_space_consistency(structures_per_space)

        if consistent and consistent['confidence'] > self.consistency_threshold:
            hyp = StructureHypothesis(
                id=str(uuid.uuid4())[:8],
                name=f"consistent_{consistent['type']}",
                representations={name: rep for name, rep in representations.items()},
                context=observation
            )
            return hyp

        return None

    def _extract_structure_features(self,
                                     representation: Dict,
                                     observation: Dict) -> Dict:
        """从表示中提取结构特征"""
        features = {
            'type': 'unknown',
            'positions': [],
            'uncertainty_pattern': None,
            'curvature_pattern': None,
        }

        fields = representation.get('fields', {})

        if 'uncertainty' in fields:
            unc = fields['uncertainty']
            features['uncertainty_pattern'] = {
                'mean': float(np.mean(unc)),
                'std': float(np.std(unc)),
                'max': float(np.max(unc)),
                'min': float(np.min(unc)),
            }

        if 'curvature' in fields:
            curv = fields['curvature']
            features['curvature_pattern'] = {
                'mean': float(np.mean(curv)),
                'std': float(np.std(curv)),
                'max': float(np.max(curv)),
            }

        return features

    def _find_cross_space_consistency(self,
                                       structures_per_space: Dict[str, Dict]) -> Optional[Dict]:
        """找跨空间一致的结构"""
        if len(structures_per_space) < 2:
            return None

        space_names = list(structures_per_space.keys())
        consistencies = []

        for i in range(len(space_names)):
            for j in range(i + 1, len(space_names)):
                s1 = structures_per_space[space_names[i]]
                s2 = structures_per_space[space_names[j]]
                similarity = self._compute_feature_similarity(s1, s2)
                consistencies.append(similarity)

        if not consistencies:
            return None

        avg_consistency = np.mean(consistencies)

        return {
            'type': 'cross_space_consistent',
            'confidence': avg_consistency,
            'space_count': len(space_names),
        }

    def _compute_feature_similarity(self, f1: Dict, f2: Dict) -> float:
        """计算特征相似度"""
        similarities = []

        if 'uncertainty_pattern' in f1 and 'uncertainty_pattern' in f2:
            u1 = f1['uncertainty_pattern']
            u2 = f2['uncertainty_pattern']
            if u1 and u2:
                mean_diff = abs(u1.get('mean', 0) - u2.get('mean', 0))
                similarities.append(1.0 - min(mean_diff, 1.0))

        if 'curvature_pattern' in f1 and 'curvature_pattern' in f2:
            c1 = f1['curvature_pattern']
            c2 = f2['curvature_pattern']
            if c1 and c2:
                mean_diff = abs(c1.get('mean', 0) - c2.get('mean', 0))
                similarities.append(1.0 - min(mean_diff, 1.0))

        return np.mean(similarities) if similarities else 0.0

    def update_weights(self,
                       observation: Dict,
                       actual: Dict,
                       space_errors: Dict[str, float]) -> None:
        """动态更新空间权重"""
        for name in self.weights:
            if name in space_errors:
                error = space_errors[name]
                self.weights[name] *= (1.0 + 0.1 * (1.0 - error))

        total = sum(self.weights.values())
        if total > 0:
            for name in self.weights:
                self.weights[name] /= total


# ============================================================================
# 4. 增强版 SSFR（优化版）
# ============================================================================

class SSFREnhanced:
    """增强版 SSFR（优化版）"""

    def __init__(self,
                 width: int = 40,
                 height: int = 20,
                 space_names: Optional[List[str]] = None,
                 max_structures: int = 100,
                 evolution_interval: int = 10,
                 reuse_threshold: float = 0.85):
        self.width = width
        self.height = height
        self.evolution_interval = evolution_interval
        self.reuse_threshold = reuse_threshold

        # 初始化空间
        space_names = space_names or ['ricci', 'fisher', 'wasserstein', 'conformal']
        self.spaces = self._init_spaces(space_names)

        # 多空间表示
        self.multi_space = MultiSpaceRepresentation(list(self.spaces.values()))

        # 结构池
        self.structure_pool = StructurePool(max_structures=max_structures)

        # 当前激活的结构
        self.active_structures: List[StructureHypothesis] = []

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

    def _init_spaces(self, space_names: List[str]) -> Dict[str, CognitiveSpace]:
        """初始化空间（缓存实例）"""
        spaces = {}
        for name in space_names:
            try:
                space = create_space(name, self.width, self.height)
                spaces[name] = space
            except Exception as e:
                print(f"Warning: Could not create space {name}: {e}")
        return spaces

    def perceive(self, position: Tuple[int, int],
                 observation: Dict,
                 active_space_name: Optional[str] = None) -> List[StructureHypothesis]:
        """感知：从观测中提取结构（优化版 + 复用机制）

        Args:
            position: 当前位置
            observation: 观测数据
            active_space_name: 如果指定，只更新和生成该空间的假设（大幅提速）
        """
        start = time.time()

        # 1. 更新空间（优化：只更新活跃空间或全部）
        if active_space_name and active_space_name in self.spaces:
            # 只更新活跃空间（快10x）
            try:
                self.spaces[active_space_name].update_from_observation(position, observation)
            except Exception:
                pass
        else:
            # 更新所有空间（完整但慢）
            for space in self.spaces.values():
                try:
                    space.update_from_observation(position, observation)
                except Exception:
                    pass

        # 2. 多空间编码（带缓存）
        representations = self.multi_space.encode(observation)

        # 3. 尝试复用现有结构（新增：复用检查）
        self.reuse_stats['total_perceptions'] += 1
        reused = self._try_reuse_structure(position, observation, active_space_name)

        if reused:
            self.reuse_stats['reused_count'] += 1
            self.reuse_stats['reuse_rate_history'].append(1.0)
            self.active_structures = [reused]
            self.timing['perceive'].append(time.time() - start)
            return [reused]

        # 4. 生成新假设（当无法复用时）
        self.reuse_stats['new_generated_count'] += 1
        self.reuse_stats['reuse_rate_history'].append(0.0)
        hypotheses = []

        if active_space_name and active_space_name in representations:
            # 只生成活跃空间的假设
            hyp = self._generate_hypothesis_from_space(
                active_space_name, representations[active_space_name], observation
            )
            if hyp:
                hypotheses.append(hyp)
        else:
            # 生成所有空间的假设
            for space_name, rep in representations.items():
                hyp = self._generate_hypothesis_from_space(space_name, rep, observation)
                if hyp:
                    hypotheses.append(hyp)

        # 5. 跨空间一致性检查（只在有足够空间时做）
        if len(representations) >= 2 and not active_space_name:
            consistent = self.multi_space.find_consistent_structure(
                representations, observation
            )
            if consistent:
                hypotheses.append(consistent)

        # 6. 添加到结构池（批量添加）
        for hyp in hypotheses:
            self.structure_pool.add(hyp)

        self.active_structures = hypotheses

        # 记录
        self.perception_history.append({
            'step': self.step_count,
            'position': position,
            'num_hypotheses': len(hypotheses),
            'active_space': active_space_name,
        })

        self.timing['perceive'].append(time.time() - start)
        return hypotheses

    def _try_reuse_structure(self, position: Tuple[int, int],
                              observation: Dict,
                              active_space_name: Optional[str] = None) -> Optional[StructureHypothesis]:
        """尝试复用现有结构

        复用条件：
        1. 位置接近（曼哈顿距离 <= 3）
        2. 空间类型匹配
        3. 特征相似度 > reuse_threshold
        """
        if not self.structure_pool.structures:
            return None

        # 提取当前观测的特征
        current_features = self._extract_observation_features(observation)

        best_match = None
        best_similarity = 0.0

        for struct in self.structure_pool.structures.values():
            # 1. 检查空间类型匹配
            if active_space_name:
                struct_space = struct.context.get('space_type', '')
                if struct_space != active_space_name:
                    continue

            # 2. 检查位置接近
            struct_pos = struct.context.get('observation', {}).get('position', None)
            if struct_pos:
                manhattan_dist = abs(position[0] - struct_pos[0]) + abs(position[1] - struct_pos[1])
                if manhattan_dist > 3:
                    continue

            # 3. 计算特征相似度
            similarity = self._compute_observation_similarity(current_features, struct)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = struct

        # 如果最佳匹配超过阈值，复用它
        if best_match and best_similarity >= self.reuse_threshold:
            # 更新复用统计
            best_match.last_used = self.step_count
            best_match.usage_count += 1
            best_match._fitness_stale = True
            return best_match

        return None

    def _extract_observation_features(self, observation: Dict) -> Dict:
        """从观测中提取用于相似度比较的特征"""
        features = {
            'position': observation.get('position', (0, 0)),
            'goal_position': observation.get('goal_position', None),
        }

        # 障碍物特征（简化：障碍物数量和最近距离）
        obstacles = observation.get('obstacles', [])
        if obstacles:
            pos = features['position']
            obstacles_list = list(obstacles)
            features['num_obstacles'] = len(obstacles_list)
            # 计算到最近障碍物的距离
            min_dist = min(
                abs(pos[0] - ox) + abs(pos[1] - oy)
                for ox, oy in obstacles_list
            ) if obstacles_list else 999
            features['nearest_obstacle_dist'] = min_dist
        else:
            features['num_obstacles'] = 0
            features['nearest_obstacle_dist'] = 999

        return features

    def _compute_observation_similarity(self, current_features: Dict,
                                         struct: StructureHypothesis) -> float:
        """计算当前观测与结构的相似度"""
        similarities = []

        # 1. 位置相似度（基于曼哈顿距离的衰减）
        struct_pos = struct.context.get('observation', {}).get('position', None)
        if struct_pos and 'position' in current_features:
            curr_pos = current_features['position']
            manhattan_dist = abs(curr_pos[0] - struct_pos[0]) + abs(curr_pos[1] - struct_pos[1])
            pos_similarity = max(0.0, 1.0 - manhattan_dist / 5.0)
            similarities.append(pos_similarity)

        # 2. 目标位置相似度
        struct_goal = struct.context.get('observation', {}).get('goal_position', None)
        curr_goal = current_features.get('goal_position', None)
        if struct_goal and curr_goal:
            goal_dist = abs(curr_goal[0] - struct_goal[0]) + abs(curr_goal[1] - struct_goal[1])
            goal_similarity = max(0.0, 1.0 - goal_dist / 10.0)
            similarities.append(goal_similarity)

        # 3. 障碍物数量相似度
        struct_obs = struct.context.get('observation', {}).get('obstacles', [])
        curr_num_obs = current_features.get('num_obstacles', 0)
        struct_num_obs = len(list(struct_obs)) if struct_obs else 0
        obs_diff = abs(curr_num_obs - struct_num_obs)
        obs_similarity = max(0.0, 1.0 - obs_diff / 5.0)
        similarities.append(obs_similarity)

        # 4. 特征模式相似度（如果结构有特征）
        struct_features = struct.context.get('features', {})
        if 'uncertainty_pattern' in struct_features:
            # 结构有不确定性模式，检查当前位置是否在相似区域
            unc_pattern = struct_features['uncertainty_pattern']
            if unc_pattern and 'mean' in unc_pattern:
                # 简化：假设当前位置的不确定性均值与结构记录相似
                # 实际实现中可以从空间实例获取当前不确定性
                similarities.append(0.8)  # 保守估计

        # 综合相似度（加权平均）
        if not similarities:
            return 0.0

        # 位置相似度权重最高
        weights = [0.4, 0.3, 0.2, 0.1][:len(similarities)]
        weighted_sum = sum(s * w for s, w in zip(similarities, weights))
        total_weight = sum(weights)

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _generate_hypothesis_from_space(self,
                                         space_name: str,
                                         representation: Dict,
                                         observation: Dict) -> Optional[StructureHypothesis]:
        """从单个空间表示生成假设"""
        space = self.spaces.get(space_name)

        # 提取关键特征
        features = self.multi_space._extract_structure_features(representation, observation)

        # 生成预测（调用空间的 predict_next_state）
        prediction = None
        if space:
            try:
                position = observation.get('position', (0, 0))
                prediction = space.predict_next_state(position, observation)
            except Exception:
                pass

        # 构建假设
        hyp = StructureHypothesis(
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

    def compete(self, observation: Dict, actual: Dict) -> Optional[StructureHypothesis]:
        """竞争：选择最优结构（优化版）"""
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

    def evolve(self) -> List[StructureHypothesis]:
        """演化：结构自我改进（优化版）"""
        start = time.time()

        new_structures = self.structure_pool.evolve(self.step_count)

        self.timing['evolve'].append(time.time() - start)
        return new_structures

    def step(self, position: Tuple[int, int],
             observation: Dict,
             actual: Optional[Dict] = None) -> Dict[str, Any]:
        """单步执行：感知 + 竞争 + 演化（优化版）"""
        total_start = time.time()

        # 1. 感知
        hypotheses = self.perceive(position, observation)

        # 2. 竞争（如果有实际结果）
        winner = None
        if actual:
            winner = self.compete(observation, actual)

        # 3. 演化（只在间隔到达时）
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

    def get_best_structures(self, n: int = 5) -> List[StructureHypothesis]:
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

        # 添加复用统计
        total = self.reuse_stats['total_perceptions']
        if total > 0:
            stats['reuse_rate'] = self.reuse_stats['reused_count'] / total
            stats['new_generation_rate'] = self.reuse_stats['new_generated_count'] / total
        stats['total_reused'] = self.reuse_stats['reused_count']
        stats['total_new_generated'] = self.reuse_stats['new_generated_count']

        # 添加性能统计
        for key, times in self.timing.items():
            if times:
                stats[f'avg_{key}_time_ms'] = np.mean(times) * 1000
                stats[f'max_{key}_time_ms'] = np.max(times) * 1000

        return stats
