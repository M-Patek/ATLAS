"""
SSFR Enhanced: 增强版稳定结构优先表示

核心设计：
1. 结构 = 假设（不是聚类结果）
2. 多空间联合表示（不是单空间选择）
3. 结构竞争（不是静态存储）
4. 结构演化（不是固定不变）
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict
import uuid
import copy

from ..core.space import CognitiveSpace
from ..core.registry import create_space


# ============================================================================
# 1. 基础数据结构
# ============================================================================

@dataclass
class ValidationResult:
    """验证结果"""
    hypothesis_id: str
    prediction_error: float
    fitness: float
    timestamp: float

    def __post_init__(self):
        # 适应度越高越好，误差越低越好
        self.composite_score = self.fitness / (1.0 + self.prediction_error)


@dataclass
class StructureHypothesis:
    """
    结构假设

    不是静态的聚类结果，而是可验证、可竞争、可演化的假设。
    每个假设包含多种数学空间的表示，以及预测和验证能力。
    """
    id: str
    name: str

    # 多空间表示: {space_name: representation_data}
    representations: Dict[str, Any] = field(default_factory=dict)

    # 生成假设的上下文
    context: Dict[str, Any] = field(default_factory=dict)

    # 验证历史
    validation_history: List[ValidationResult] = field(default_factory=list)

    # 代价（计算 + 维护）
    computation_cost: float = 1.0
    maintenance_cost: float = 0.1

    # 生命周期
    created_at: float = 0.0
    last_used: float = 0.0
    usage_count: int = 0

    # 竞争状态
    _fitness_cache: Optional[float] = None
    _fitness_stale: bool = True

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]

    @property
    def fitness(self) -> float:
        """适应度 = 预测准确度 / 总代价"""
        if self._fitness_stale or self._fitness_cache is None:
            self._fitness_cache = self._compute_fitness()
            self._fitness_stale = False
        return self._fitness_cache

    def _compute_fitness(self) -> float:
        """计算适应度"""
        if not self.validation_history:
            return 0.5  # 默认值

        # 最近验证的加权平均
        recent = self.validation_history[-10:]
        weights = np.exp(np.linspace(-1, 0, len(recent)))
        errors = np.array([v.prediction_error for v in recent])

        weighted_error = np.sum(weights * errors) / np.sum(weights)

        # 适应度 = 准确度 / 代价
        accuracy = 1.0 / (1.0 + weighted_error)
        total_cost = self.computation_cost + self.maintenance_cost * self.usage_count

        return accuracy / (1.0 + total_cost * 0.01)

    def validate(self, observation: Dict, actual: Dict, timestamp: float = 0.0) -> ValidationResult:
        """验证预测，更新适应度"""
        # 基于表示生成预测
        prediction = self._generate_prediction(observation)

        # 计算误差
        error = self._compute_prediction_error(prediction, actual)

        # 创建验证结果
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
        """
        基于表示生成预测

        核心改进：每个空间表示必须包含真正的预测器，
        调用空间的 predict_next_state 方法生成有意义的预测。
        """
        predictions = {}

        for space_name, rep in self.representations.items():
            if isinstance(rep, dict):
                # 情况1: 表示中包含空间实例和观测数据
                if 'space_instance' in rep and 'observation' in rep:
                    space = rep['space_instance']
                    obs = rep['observation']
                    position = obs.get('position', (0, 0))
                    try:
                        pred = space.predict_next_state(position, obs)
                        predictions[space_name] = pred
                    except Exception as e:
                        # 回退到简单预测
                        predictions[space_name] = self._fallback_prediction(obs)

                # 情况2: 表示中包含预计算的预测
                elif 'predictor' in rep and callable(rep['predictor']):
                    predictions[space_name] = rep['predictor'](observation)

                # 情况3: 表示中包含预测结果缓存
                elif 'prediction' in rep:
                    predictions[space_name] = rep['prediction']

                else:
                    # 默认回退
                    predictions[space_name] = self._fallback_prediction(observation)
            else:
                predictions[space_name] = self._fallback_prediction(observation)

        # 多空间融合预测：如果多个空间都有预测，取加权平均
        if len(predictions) > 1:
            predictions['_fused'] = self._fuse_predictions(predictions)

        return predictions

    def _fallback_prediction(self, observation: Dict) -> Dict:
        """回退预测：基于观测做简单推断"""
        position = observation.get('position', (0, 0))
        goal = observation.get('goal_position')

        # 简单预测：朝向目标移动一步
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
        """融合多个空间的预测"""
        # 收集所有预测的位置
        positions = []
        weights = []

        for space_name, pred in predictions.items():
            if space_name.startswith('_'):
                continue
            if 'predicted_position' in pred:
                positions.append(pred['predicted_position'])
                # 权重基于不确定性（越低越好）
                unc = pred.get('predicted_uncertainty', 0.5)
                weights.append(1.0 / (1.0 + unc))

        if not positions:
            return self._fallback_prediction({})

        # 加权投票选择位置
        weights = np.array(weights)
        weights /= weights.sum()

        # 找多数同意的位置
        pos_votes = {}
        for pos, w in zip(positions, weights):
            pos_key = pos
            pos_votes[pos_key] = pos_votes.get(pos_key, 0) + w

        best_pos = max(pos_votes.keys(), key=lambda k: pos_votes[k])
        confidence = pos_votes[best_pos]

        # 融合其他字段
        fused_cost = np.mean([
            pred.get('predicted_cost', 1.0)
            for pred in predictions.values()
            if not pred.get('_fused')
        ]) if predictions else 1.0

        fused_unc = np.mean([
            pred.get('predicted_uncertainty', 0.5)
            for pred in predictions.values()
            if not pred.get('_fused')
        ]) if predictions else 0.5

        return {
            'predicted_position': best_pos,
            'predicted_cost': float(fused_cost),
            'predicted_uncertainty': float(fused_unc),
            'passable': confidence > 0.3,
            'fusion_confidence': float(confidence),
            'agreeing_spaces': sum(1 for p in positions if p == best_pos),
            'total_spaces': len(positions),
        }

    def _compute_prediction_error(self, prediction: Dict, actual: Dict) -> float:
        """计算预测误差"""
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
        """与另一个假设合并"""
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
        """变异"""
        mutated_reps = copy.deepcopy(self.representations)

        # 随机变异表示
        for space_name in list(mutated_reps.keys()):
            if np.random.random() < mutation_rate:
                rep = mutated_reps[space_name]
                if isinstance(rep, dict) and 'params' in rep:
                    # 变异参数
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
# 2. 结构竞争池
# ============================================================================

class StructurePool:
    """
    结构竞争池

    管理多个假设的竞争、选择、演化。
    核心思想：不是静态存储结构，而是让结构在竞争中自我优化。
    """

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

        # 竞争统计
        self.competition_history: List[Dict] = []
        self.generation = 0

    def add(self, hypothesis: StructureHypothesis) -> None:
        """添加假设到池中"""
        if len(self.structures) >= self.max_structures:
            # 淘汰最弱的
            self._eliminate_weakest(n=1)

        self.structures[hypothesis.id] = hypothesis

    def compete(self,
                observation: Dict,
                actual: Dict,
                timestamp: float = 0.0) -> Tuple[StructureHypothesis, List[Tuple[StructureHypothesis, ValidationResult]]]:
        """
        结构竞争

        所有结构都尝试预测，适应度高的胜出。

        Returns:
            (winner, all_results)
        """
        if not self.structures:
            return None, []

        # 所有结构验证
        results = []
        for hyp in self.structures.values():
            result = hyp.validate(observation, actual, timestamp)
            results.append((hyp, result))

        # 排序
        results.sort(key=lambda x: x[1].fitness, reverse=True)

        # 胜者
        winner = results[0][0]

        # 记录历史
        self.competition_history.append({
            'timestamp': timestamp,
            'winner_id': winner.id,
            'winner_fitness': results[0][1].fitness,
            'num_competitors': len(results),
            'avg_fitness': np.mean([r[1].fitness for r in results])
        })

        return winner, results

    def evolve(self, timestamp: float = 0.0) -> List[StructureHypothesis]:
        """
        结构演化

        基于竞争结果演化结构。

        Returns:
            新生成的结构
        """
        new_structures = []

        # 选择优秀结构作为父代
        parents = self._select_parents(n=5)

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

        # 定期淘汰
        if self.generation % 10 == 0:
            self._eliminate_weakest(n=max(1, int(len(self.structures) * self.elimination_rate)))

        self.generation += 1

        return new_structures

    def _select_parents(self, n: int = 5) -> List[StructureHypothesis]:
        """选择优秀父代"""
        if not self.structures:
            return []

        # 按适应度排序
        sorted_structs = sorted(
            self.structures.values(),
            key=lambda h: h.fitness,
            reverse=True
        )

        # 返回前n个
        return sorted_structs[:n]

    def _eliminate_weakest(self, n: int = 1) -> None:
        """淘汰最弱的结构"""
        if len(self.structures) <= n:
            return

        # 按适应度排序
        sorted_ids = sorted(
            self.structures.keys(),
            key=lambda sid: self.structures[sid].fitness
        )

        # 淘汰最弱的
        for sid in sorted_ids[:n]:
            del self.structures[sid]

    def get_best(self, n: int = 5) -> List[StructureHypothesis]:
        """获取最佳结构"""
        sorted_structs = sorted(
            self.structures.values(),
            key=lambda h: h.fitness,
            reverse=True
        )
        return sorted_structs[:n]

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
# 3. 多空间联合表示
# ============================================================================

class MultiSpaceRepresentation:
    """
    多空间联合表示

    不是"选一个空间"，而是"所有空间共同表示"。
    核心思想：如果多个空间都"看到"了相同的结构，那这个结构更可能是稳定的。
    """

    def __init__(self,
                 spaces: List[CognitiveSpace],
                 consistency_threshold: float = 0.7):
        self.spaces = {s.name: s for s in spaces}
        self.consistency_threshold = consistency_threshold

        # 空间间的权重（动态调整）
        self.weights: Dict[str, float] = {name: 1.0 for name in self.spaces}

        # 一致性历史
        self.consistency_history: List[Dict] = []

    def encode(self, observation: Dict) -> Dict[str, Any]:
        """
        将观测编码到所有空间

        Returns:
            {space_name: representation}
        """
        representations = {}

        for name, space in self.spaces.items():
            # 获取空间的场数据作为表示
            fields = space.get_visualization_fields()

            # 构建表示
            rep = {
                'fields': fields,
                'statistics': space.get_statistics(),
                'space_type': name,
                'weight': self.weights.get(name, 1.0),
            }

            representations[name] = rep

        return representations

    def find_consistent_structure(self,
                                   representations: Dict[str, Any],
                                   observation: Dict) -> Optional[StructureHypothesis]:
        """
        在所有空间中寻找一致的结构

        核心思想：跨空间一致的结构更可信
        """
        # 从每个空间中提取结构特征
        structures_per_space = {}
        for space_name, rep in representations.items():
            features = self._extract_structure_features(rep, observation)
            structures_per_space[space_name] = features

        # 找跨空间一致的结构
        consistent = self._find_cross_space_consistency(structures_per_space)

        if consistent and consistent['confidence'] > self.consistency_threshold:
            # 构建假设
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

        # 提取关键特征
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

        # 简单的一致性检查：比较特征
        space_names = list(structures_per_space.keys())
        consistencies = []

        for i in range(len(space_names)):
            for j in range(i + 1, len(space_names)):
                s1 = structures_per_space[space_names[i]]
                s2 = structures_per_space[space_names[j]]

                # 比较特征
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

        # 比较 uncertainty pattern
        if 'uncertainty_pattern' in f1 and 'uncertainty_pattern' in f2:
            u1 = f1['uncertainty_pattern']
            u2 = f2['uncertainty_pattern']
            if u1 and u2:
                # 简单的均值比较
                mean_diff = abs(u1.get('mean', 0) - u2.get('mean', 0))
                similarities.append(1.0 - min(mean_diff, 1.0))

        # 比较 curvature pattern
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
        """
        动态更新空间权重

        基于各空间的预测误差调整权重
        """
        for name in self.weights:
            if name in space_errors:
                error = space_errors[name]
                # 误差小的空间权重增加
                self.weights[name] *= (1.0 + 0.1 * (1.0 - error))

        # 归一化
        total = sum(self.weights.values())
        if total > 0:
            for name in self.weights:
                self.weights[name] /= total


# ============================================================================
# 4. 增强版 SSFR
# ============================================================================

class SSFREnhanced:
    """
    增强版 SSFR

    核心改进：
    1. 结构 = 假设（不是聚类结果）
    2. 多空间联合表示（不是单空间选择）
    3. 结构竞争（不是静态存储）
    4. 结构演化（不是固定不变）

    使用方式：
        ssfr = SSFREnhanced(width=40, height=20)

        # 感知并发现结构
        hypotheses = ssfr.perceive(observation)

        # 竞争选择最优结构
        winner = ssfr.compete(observation, actual)

        # 演化结构
        ssfr.evolve()
    """

    def __init__(self,
                 width: int = 40,
                 height: int = 20,
                 space_names: Optional[List[str]] = None,
                 max_structures: int = 100,
                 evolution_interval: int = 10):
        self.width = width
        self.height = height
        self.evolution_interval = evolution_interval

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

    def _init_spaces(self, space_names: List[str]) -> Dict[str, CognitiveSpace]:
        """初始化空间"""
        spaces = {}
        for name in space_names:
            try:
                space = create_space(name, self.width, self.height)
                spaces[name] = space
            except Exception as e:
                print(f"Warning: Could not create space {name}: {e}")

        return spaces

    def perceive(self, position: Tuple[int, int],
                 observation: Dict) -> List[StructureHypothesis]:
        """
        感知：从观测中提取结构

        不是聚类，而是生成假设。
        """
        # 1. 更新所有空间
        for space in self.spaces.values():
            try:
                space.update_from_observation(position, observation)
            except Exception as e:
                pass  # 某些空间可能不支持某些观测

        # 2. 多空间编码
        representations = self.multi_space.encode(observation)

        # 3. 生成假设
        hypotheses = []

        # 从每个空间生成假设
        for space_name, rep in representations.items():
            hyp = self._generate_hypothesis_from_space(space_name, rep, observation)
            if hyp:
                hypotheses.append(hyp)

        # 4. 跨空间一致性检查
        consistent = self.multi_space.find_consistent_structure(
            representations, observation
        )
        if consistent:
            hypotheses.append(consistent)

        # 5. 添加到结构池
        for hyp in hypotheses:
            self.structure_pool.add(hyp)

        self.active_structures = hypotheses

        # 记录
        self.perception_history.append({
            'step': self.step_count,
            'position': position,
            'num_hypotheses': len(hypotheses),
        })

        return hypotheses

    def _generate_hypothesis_from_space(self,
                                         space_name: str,
                                         representation: Dict,
                                         observation: Dict) -> Optional[StructureHypothesis]:
        """从单个空间表示生成假设"""
        # 获取空间实例
        space = self.spaces.get(space_name)

        # 提取关键特征
        features = self.multi_space._extract_structure_features(representation, observation)

        # 生成预测：调用空间的 predict_next_state
        prediction = None
        if space:
            try:
                position = observation.get('position', (0, 0))
                prediction = space.predict_next_state(position, observation)
            except Exception as e:
                pass  # 某些空间可能不支持预测

        # 构建假设：表示中包含空间实例和观测，供后续预测使用
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
        """
        竞争：选择最优结构

        所有结构竞争，适应度高的胜出。
        """
        winner, results = self.structure_pool.compete(
            observation, actual, self.step_count
        )

        if winner:
            self.competition_history.append({
                'step': self.step_count,
                'winner_id': winner.id,
                'num_competitors': len(results),
            })

        return winner

    def evolve(self) -> List[StructureHypothesis]:
        """
        演化：结构自我改进
        """
        new_structures = self.structure_pool.evolve(self.step_count)

        # 定期演化
        if self.step_count % self.evolution_interval == 0:
            # 额外的演化操作
            pass

        return new_structures

    def step(self, position: Tuple[int, int],
             observation: Dict,
             actual: Optional[Dict] = None) -> Dict[str, Any]:
        """
        单步执行：感知 + 竞争 + 演化

        Returns:
            包含当前状态的字典
        """
        # 1. 感知
        hypotheses = self.perceive(position, observation)

        # 2. 竞争（如果有实际结果）
        winner = None
        if actual:
            winner = self.compete(observation, actual)

        # 3. 演化
        new_structures = self.evolve()

        self.step_count += 1

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
        return {
            'step_count': self.step_count,
            'num_spaces': len(self.spaces),
            'pool_stats': self.structure_pool.get_statistics(),
            'num_perceptions': len(self.perception_history),
            'num_competitions': len(self.competition_history),
        }
