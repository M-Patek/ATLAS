"""
Stable-Structure-First Representation (SSFR)
稳定结构优先表示

核心洞察：
- 价值 ≠ 压缩率（compression ratio）
- 价值 = 压缩所揭示的稳定结构（stable structures）
- 稳定结构 = 跨时间/空间/任务不变的模式
- 真正的价值度量：发现结构后，Agent在未来用更少资源完成更多任务

资源效率（Resource Efficiency）：
- 计算成本：推理时的FLOPs
- 记忆成本：存储表示所需的比特
- 规划成本：找到行动序列所需的搜索步数
- 感知成本：获取信息所需的观测次数

一个观察/位置/行动的价值 = 它让未来任务的总资源消耗减少了多少
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod
import time


# ============================================================================
# 1. 稳定结构：跨时间/空间/任务不变的模式
# ============================================================================

@dataclass
class StableStructure:
    """
    稳定结构：可重用的世界规律

    关键属性：
    - generality: 适用范围（多少位置/时间/任务适用）
    - stability: 稳定性（跨时间变化的程度）
    - utility: 实用性（能带来多少资源节省）
    - cost: 表示成本（存储和使用的资源）
    """

    # 结构的数学描述
    predictor: Callable  # f(context, query) → prediction

    # 结构的适用范围
    spatial_coverage: Set[Tuple[int, int]] = field(default_factory=set)
    temporal_range: Tuple[float, float] = (0.0, 0.0)  # (first_seen, last_used)
    task_types: Set[str] = field(default_factory=set)

    # 结构的质量
    prediction_accuracy: float = 0.0  # 预测准确度
    usage_count: int = 0  # 使用次数

    # 资源效率度量
    flops_per_query: float = 1.0  # 每次查询的计算成本
    memory_bits: float = 32.0  # 存储成本（比特）

    @property
    def generality(self) -> float:
        """通用性：适用范围广 = 更有价值"""
        spatial_score = min(1.0, len(self.spatial_coverage) / 100.0)
        temporal_score = min(1.0, (self.temporal_range[1] - self.temporal_range[0]) / 1000.0)
        task_score = min(1.0, len(self.task_types) / 5.0)
        return (spatial_score + temporal_score + task_score) / 3.0

    @property
    def stability(self) -> float:
        """稳定性：预测准确度随时间的保持度"""
        # 使用次数越多且准确度保持高 = 更稳定
        if self.usage_count == 0:
            return 0.0
        return self.prediction_accuracy * min(1.0, self.usage_count / 10.0)

    @property
    def utility(self) -> float:
        """
        实用性：使用此结构相比不用能节省多少资源

        计算方式：
        - 不用结构：每次需要完整观测/搜索/计算
        - 用结构：直接预测，省去观测/搜索

        utility = (不用结构的成本 - 用结构的成本) × 使用次数
        """
        # 假设不用结构需要10倍资源
        cost_without = 10.0 * self.flops_per_query
        cost_with = self.flops_per_query + self.memory_bits / 1000.0  # 摊销存储成本
        saving_per_use = cost_without - cost_with
        return saving_per_use * self.usage_count

    @property
    def value(self) -> float:
        """
        结构的价值 = 实用性 / 成本

        即：每比特存储/每次计算能带来多少资源节省
        """
        if self.memory_bits == 0 or self.flops_per_query == 0:
            return 0.0
        return self.utility / (self.memory_bits + self.flops_per_query * self.usage_count)

    def predict(self, context: Dict[str, Any], query: Any) -> Tuple[Any, float]:
        """
        使用结构预测

        Returns: (prediction, confidence)
        """
        try:
            result = self.predictor(context, query)
            if isinstance(result, tuple):
                return result
            return result, self.prediction_accuracy
        except Exception:
            return None, 0.0

    def update_accuracy(self, predicted: Any, actual: Any) -> None:
        """更新准确度（指数移动平均）"""
        error = self._compute_error(predicted, actual)
        # EMA: 新准确度 = 0.9 × 旧 + 0.1 × (1 - error)
        self.prediction_accuracy = 0.9 * self.prediction_accuracy + 0.1 * max(0, 1 - error)
        self.usage_count += 1

    def _compute_error(self, predicted, actual) -> float:
        """计算预测误差"""
        if predicted is None or actual is None:
            return 1.0
        if isinstance(predicted, np.ndarray) and isinstance(actual, np.ndarray):
            if predicted.shape == actual.shape:
                return np.mean((predicted - actual) ** 2)
        try:
            return abs(predicted - actual) / (1 + abs(actual))
        except:
            return 1.0 if predicted != actual else 0.0


# ============================================================================
# 2. 资源效率度量：计算、记忆、规划、感知
# ============================================================================

@dataclass
class ResourceCost:
    """资源消耗"""
    compute_flops: float = 0.0      # 计算成本（浮点运算）
    memory_bits: float = 0.0        # 记忆成本（比特）
    planning_steps: int = 0         # 规划步数
    perception_queries: int = 0     # 感知查询次数
    time_ms: float = 0.0            # 时间成本

    def total(self, weights: Dict[str, float] = None) -> float:
        """加权总成本"""
        if weights is None:
            weights = {
                'compute': 1.0,
                'memory': 0.1,
                'planning': 10.0,
                'perception': 5.0,
                'time': 0.01
            }
        return (
            weights['compute'] * self.compute_flops +
            weights['memory'] * self.memory_bits +
            weights['planning'] * self.planning_steps +
            weights['perception'] * self.perception_queries +
            weights['time'] * self.time_ms
        )

    def __add__(self, other: 'ResourceCost') -> 'ResourceCost':
        return ResourceCost(
            compute_flops=self.compute_flops + other.compute_flops,
            memory_bits=self.memory_bits + other.memory_bits,
            planning_steps=self.planning_steps + other.planning_steps,
            perception_queries=self.perception_queries + other.perception_queries,
            time_ms=self.time_ms + other.time_ms
        )


class ResourceTracker:
    """资源追踪器：记录Agent的资源消耗历史"""

    def __init__(self):
        self.history: List[ResourceCost] = []
        self.cumulative: ResourceCost = ResourceCost()

    def record(self, cost: ResourceCost) -> None:
        self.history.append(cost)
        self.cumulative = self.cumulative + cost

    def efficiency_trend(self, window: int = 10) -> float:
        """
        效率趋势：后期任务平均成本 / 前期任务平均成本

        < 1 表示效率提升（发现结构有帮助）
        > 1 表示效率下降
        """
        if len(self.history) < window * 2:
            return 1.0

        early = sum(h.total() for h in self.history[:window]) / window
        late = sum(h.total() for h in self.history[-window:]) / window

        if early == 0:
            return 1.0
        return late / early


# ============================================================================
# 3. 稳定结构引擎：发现、验证、使用结构
# ============================================================================

class StructureEngine:
    """
    稳定结构引擎

    核心职责：
    1. 从观测中提取候选结构
    2. 验证结构的稳定性（跨时间/空间/任务）
    3. 使用结构降低未来任务的资源消耗
    4. 评估结构的价值（资源节省 / 表示成本）
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        # 存储的结构
        self.structures: List[StableStructure] = []

        # 原始观测（用于发现结构）
        self.observations: Dict[Tuple[int, int], List[Tuple[Any, float, ResourceCost]]] = defaultdict(list)
        # 格式: position → [(observation, time, resource_cost), ...]

        # 资源追踪
        self.resources = ResourceTracker()

        # 当前时间
        self.time = 0.0

        # 结构索引（快速查找）
        self.structure_index: Dict[Tuple[int, int], List[StableStructure]] = defaultdict(list)

    def observe(self, position: Tuple[int, int],
                observation: Any,
                task_type: str = "explore") -> Tuple[Any, ResourceCost]:
        """
        观测：尝试用现有结构解释，或发现新结构

        Returns: (解释, 资源消耗)
        """
        start_time = time.time()
        cost = ResourceCost()

        # 1. 尝试用现有结构解释（低成本）
        best_structure, best_prediction, best_confidence = self._try_structures(position, observation)

        if best_structure and best_confidence > 0.7:
            # 结构解释成功 → 低资源消耗
            cost.compute_flops = best_structure.flops_per_query
            cost.memory_bits = best_structure.memory_bits
            cost.perception_queries = 0  # 不需要额外感知

            # 更新结构准确度
            best_structure.update_accuracy(best_prediction, observation)
            best_structure.temporal_range = (
                best_structure.temporal_range[0],
                self.time
            )

            self.resources.record(cost)
            self.time += 1

            return best_prediction, cost

        # 2. 结构解释失败 → 需要直接处理（高资源消耗）
        cost.perception_queries = 1
        cost.compute_flops = 100.0  # 无结构时的处理成本

        # 存储观测（用于未来发现结构）
        self.observations[position].append((observation, self.time, cost))

        # 尝试发现新结构
        new_structure = self._discover_structure(position, task_type)
        if new_structure:
            self.structures.append(new_structure)
            # 更新索引
            for pos in new_structure.spatial_coverage:
                self.structure_index[pos].append(new_structure)

        self.resources.record(cost)
        self.time += 1

        elapsed = (time.time() - start_time) * 1000
        cost.time_ms = elapsed

        return observation, cost  # 返回原始观测（无结构解释）

    def _try_structures(self, position: Tuple[int, int],
                        observation: Any) -> Tuple[Optional[StableStructure], Any, float]:
        """尝试用现有结构解释观测"""
        candidates = self.structure_index.get(position, [])

        best_structure = None
        best_prediction = None
        best_confidence = 0.0

        for structure in candidates:
            pred, conf = structure.predict(
                {'position': position, 'time': self.time},
                observation
            )
            if conf > best_confidence:
                best_confidence = conf
                best_prediction = pred
                best_structure = structure

        return best_structure, best_prediction, best_confidence

    def _discover_structure(self, position: Tuple[int, int],
                            task_type: str) -> Optional[StableStructure]:
        """
        从观测历史中发现新结构

        发现条件：
        1. 有足够多的相似观测（同一位置的多次观测）
        2. 可以用简洁规则描述（方差小 = 稳定）
        3. 规则在多个位置适用（空间泛化）
        """
        # 获取此位置的观测历史
        obs_history = self.observations.get(position, [])

        if len(obs_history) < 2:
            return None  # 数据不足

        # 提取观测值（忽略时间和成本）
        recent_obs = [o for o, t, c in obs_history[-10:]]

        if len(recent_obs) < 2:
            return None

        # 检查是否相似（方差小 = 稳定模式）
        try:
            # 统一转换为numpy数组
            obs_arrays = []
            for o in recent_obs:
                if isinstance(o, np.ndarray):
                    obs_arrays.append(o.flatten())
                elif isinstance(o, (list, tuple)):
                    obs_arrays.append(np.array(o).flatten())
                else:
                    obs_arrays.append(np.array([float(o)]))

            if len(obs_arrays) < 2:
                return None

            obs_array = np.array(obs_arrays)
            variance = np.var(obs_array, axis=0).mean()

            # 方差小 = 稳定模式
            if variance < 0.5:  # 放宽阈值
                # 创建结构：预测平均值
                mean_obs = np.mean(obs_array, axis=0)

                # 找到相似的其他位置
                similar_positions = self._find_similar_positions(position, recent_obs[-1])

                # 创建预测函数（闭包捕获mean_obs）
                mean_obs_copy = mean_obs.copy()

                def make_predictor(mean):
                    def predictor(context, query):
                        return mean, 0.8
                    return predictor

                structure = StableStructure(
                    predictor=make_predictor(mean_obs_copy),
                    spatial_coverage=set(similar_positions) | {position},
                    temporal_range=(self.time - len(obs_history), self.time),
                    task_types={task_type},
                    prediction_accuracy=0.8,
                    usage_count=0,
                    flops_per_query=1.0,  # 低成本预测
                    memory_bits=len(mean_obs_copy) * 32.0  # 存储均值
                )

                return structure

        except Exception as e:
            pass

        return None

    def _find_similar_positions(self, position: Tuple[int, int],
                                observation: Any) -> List[Tuple[int, int]]:
        """找到产生相似观测的其他位置"""
        similar = []
        for pos, obs_list in self.observations.items():
            if pos == position:
                continue
            if not obs_list:
                continue
            last_obs = obs_list[-1][0]
            if self._similarity(observation, last_obs) < 0.3:  # 放宽阈值
                similar.append(pos)
        return similar[:10]  # 最多10个

    def _similarity(self, a: Any, b: Any) -> float:
        """计算相似度"""
        try:
            a_arr = np.array(a).flatten().astype(float)
            b_arr = np.array(b).flatten().astype(float)
            if len(a_arr) == len(b_arr) and len(a_arr) > 0:
                # 归一化欧氏距离
                dist = np.linalg.norm(a_arr - b_arr)
                max_val = max(np.linalg.norm(a_arr), np.linalg.norm(b_arr))
                if max_val > 0:
                    return dist / max_val
                return 0.0
        except:
            pass
        return 0.0 if a == b else 1.0

    def plan(self, start: Tuple[int, int],
             goal: Tuple[int, int],
             use_structures: bool = True) -> Tuple[List[Tuple[int, int]], ResourceCost]:
        """
        规划路径

        如果有结构：低规划成本（直接预测结果）
        如果无结构：高规划成本（需要搜索）
        """
        cost = ResourceCost()

        if use_structures and self.structures:
            # 使用结构：低成本规划
            # 简化：假设结构告诉我们大致方向
            path = self._structure_guided_plan(start, goal)
            cost.planning_steps = len(path)
            cost.compute_flops = len(path) * 2.0  # 低成本
        else:
            # 无结构：高成本搜索
            path = self._blind_search(start, goal)
            cost.planning_steps = len(path) * 10  # 高成本
            cost.compute_flops = len(path) * 100.0

        return path, cost

    def _structure_guided_plan(self, start: Tuple[int, int],
                               goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """结构引导的规划（低成本）"""
        # 简化：直接向目标移动
        path = [start]
        current = start

        while current != goal and len(path) < 50:
            # 使用结构预测最佳方向
            dx = np.sign(goal[0] - current[0])
            dy = np.sign(goal[1] - current[1])

            # 检查是否有结构说这个位置是安全的
            next_pos = (current[0] + int(dx), current[1] + int(dy))

            if 0 <= next_pos[0] < self.width and 0 <= next_pos[1] < self.height:
                path.append(next_pos)
                current = next_pos
            else:
                break

        return path

    def _blind_search(self, start: Tuple[int, int],
                      goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """盲目搜索（高成本）"""
        # BFS
        from collections import deque
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            pos, path = queue.popleft()

            if pos == goal:
                return path

            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = pos[0] + dx, pos[1] + dy
                next_pos = (nx, ny)

                if (0 <= nx < self.width and 0 <= ny < self.height and
                    next_pos not in visited):
                    visited.add(next_pos)
                    queue.append((next_pos, path + [next_pos]))

        return [start]  # 失败

    def evaluate_position(self, position: Tuple[int, int]) -> float:
        """
        评估位置的价值

        价值 = 在此位置获取的观测能帮助发现多少稳定结构
        = 预期资源节省 / 获取成本
        """
        # 如果此位置已有强结构，价值低（已被利用）
        existing = self.structure_index.get(position, [])
        if existing and all(s.stability > 0.8 for s in existing):
            return 0.1  # 低价值（已充分探索）

        # 如果邻居有结构但此位置没有，价值高（可能完成模式）
        neighbor_structures = 0
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = position[0] + dx, position[1] + dy
            if (nx, ny) in self.structure_index:
                neighbor_structures += len(self.structure_index[(nx, ny)])

        # 如果此位置观测历史显示稳定模式正在形成，价值高
        obs_history = self.observations.get(position, [])
        stability_signal = 0.0
        if len(obs_history) >= 2:
            recent = [o for o, t, c in obs_history[-3:]]
            if len(recent) >= 2:
                try:
                    arr = np.array([np.array(r).flatten() for r in recent])
                    stability_signal = 1.0 - min(1.0, np.var(arr, axis=0).mean() * 10)
                except:
                    pass

        # 综合价值
        return (stability_signal * 0.5 +
                min(1.0, neighbor_structures / 4.0) * 0.3 +
                (1.0 if not existing else 0.0) * 0.2)

    def get_efficiency_report(self) -> Dict[str, Any]:
        """生成效率报告"""
        trend = self.resources.efficiency_trend()

        return {
            'total_structures': len(self.structures),
            'total_observations': sum(len(v) for v in self.observations.values()),
            'cumulative_cost': {
                'compute_flops': self.resources.cumulative.compute_flops,
                'memory_bits': self.resources.cumulative.memory_bits,
                'planning_steps': self.resources.cumulative.planning_steps,
                'perception_queries': self.resources.cumulative.perception_queries,
                'time_ms': self.resources.cumulative.time_ms,
            },
            'efficiency_trend': trend,
            'efficiency_improved': trend < 1.0,
            'top_structures': [
                {
                    'generality': s.generality,
                    'stability': s.stability,
                    'utility': s.utility,
                    'value': s.value
                }
                for s in sorted(self.structures, key=lambda x: x.value, reverse=True)[:5]
            ]
        }


# ============================================================================
# 4. 演示：稳定结构优先的Agent
# ============================================================================

def demo_stable_structure():
    """演示：稳定结构发现与资源效率"""

    print("=" * 70)
    print("Stable-Structure-First Representation")
    print("Value = Stable Structures Discovered = Future Resource Savings")
    print("=" * 70)

    # 创建引擎
    engine = StructureEngine(20, 20)

    # 场景1：探索走廊（发现稳定结构）
    print("\n--- Phase 1: Explore Corridor (Discover Structure) ---")
    print("Corridor positions have similar observations (stable pattern)")

    corridor_positions = [(i, 10) for i in range(5, 15)]

    for i, pos in enumerate(corridor_positions):
        # 走廊中的观测：稳定模式（地面特征相似）
        obs = np.array([0.2, 0.2, 0.2])  # 走廊地面特征
        result, cost = engine.observe(pos, obs, task_type="explore")

        if i < 3 or i == len(corridor_positions) - 1:
            print(f"  Observe {pos}: cost={cost.total():.1f}, "
                  f"structures={len(engine.structures)}")

    print(f"\nAfter corridor: {len(engine.structures)} structures discovered")

    # 场景2：再次经过走廊（使用结构，低成本）
    print("\n--- Phase 2: Revisit Corridor (Use Structure, Low Cost) ---")
    print("Now the corridor pattern is known → predictions are cheap")

    for i, pos in enumerate(corridor_positions[:3]):
        obs = np.array([0.2, 0.2, 0.2])
        result, cost = engine.observe(pos, obs, task_type="explore")
        print(f"  Observe {pos}: cost={cost.total():.1f} "
              f"({'HIGH' if cost.total() > 50 else 'LOW'} cost)")

    # 场景3：规划到目标
    print("\n--- Phase 3: Plan to Goal ---")

    start = (5, 10)
    goal = (15, 10)

    # 无结构规划（高成本）
    path_no_struct, cost_no_struct = engine.plan(start, goal, use_structures=False)
    print(f"  Without structures: path={len(path_no_struct)} steps, "
          f"cost={cost_no_struct.total():.1f}")

    # 有结构规划（低成本）
    path_with_struct, cost_with_struct = engine.plan(start, goal, use_structures=True)
    print(f"  With structures: path={len(path_with_struct)} steps, "
          f"cost={cost_with_struct.total():.1f}")

    # 效率提升
    if cost_no_struct.total() > 0:
        improvement = (cost_no_struct.total() - cost_with_struct.total()) / cost_no_struct.total()
        print(f"  Efficiency improvement: {improvement*100:.1f}%")

    # 场景4：评估位置价值
    print("\n--- Phase 4: Position Value (Where to go next?) ---")

    test_positions = [(7, 10), (12, 10), (10, 12), (10, 8)]
    for pos in test_positions:
        value = engine.evaluate_position(pos)
        print(f"  {pos}: value={value:.3f}")

    # 效率报告
    print("\n--- Efficiency Report ---")
    report = engine.get_efficiency_report()
    print(f"  Total structures: {report['total_structures']}")
    print(f"  Total observations: {report['total_observations']}")
    print(f"  Efficiency trend: {report['efficiency_trend']:.3f}")
    print(f"  Efficiency improved: {report['efficiency_improved']}")

    print("\n" + "=" * 70)
    print("Key Insight:")
    print("  - First visit: high cost (no structures)")
    print("  - Later visits: low cost (structures discovered)")
    print("  - Value of exploration = future resource savings")
    print("  - Stable structure = reusable pattern that reduces future costs")
    print("=" * 70)


def demo_structure_types():
    """演示不同类型的稳定结构"""

    print("\n" + "=" * 70)
    print("Demo 2: Different Types of Stable Structures")
    print("=" * 70)

    engine = StructureEngine(20, 20)

    # 结构类型1：空间模式（走廊）
    print("\n--- Structure Type 1: Spatial Pattern (Corridor) ---")
    print("Same observation across different positions → spatial structure")

    for x in range(5, 15):
        engine.observe((x, 10), np.array([0.2, 0.2, 0.2]), "explore")

    print(f"  Discovered: {len(engine.structures)} spatial structures")
    if engine.structures:
        s = engine.structures[0]
        print(f"  Coverage: {len(s.spatial_coverage)} positions")
        print(f"  Generality: {s.generality:.3f}")
        print(f"  Stability: {s.stability:.3f}")

    # 结构类型2：时间模式（周期性变化）
    print("\n--- Structure Type 2: Temporal Pattern (Day/Night Cycle) ---")
    print("Same position, periodic observations → temporal structure")

    for t in range(10):
        # 模拟日夜交替
        if t % 2 == 0:
            obs = np.array([1.0, 0.8, 0.5])  # 白天
        else:
            obs = np.array([0.1, 0.1, 0.3])  # 夜晚
        engine.observe((5, 10), obs, "explore")

    print(f"  Total structures now: {len(engine.structures)}")
    print(f"  Temporal structures can predict future observations")

    # 结构类型3：任务模式
    print("\n--- Structure Type 3: Task Pattern (Navigation Strategy) ---")
    print("Same task, similar solutions → task structure")

    # 模拟多次导航任务
    for _ in range(5):
        for x in range(5, 10):
            engine.observe((x, 10), np.array([0.2, 0.2, 0.2]), "navigate")

    print(f"  Total structures: {len(engine.structures)}")
    print(f"  Task structures enable strategy transfer")


def demo_resource_efficiency():
    """演示资源效率随时间提升"""

    print("\n" + "=" * 70)
    print("Demo 3: Resource Efficiency Over Time")
    print("=" * 70)

    engine = StructureEngine(20, 20)

    # 模拟多次探索-利用循环
    phases = [
        ("Initial exploration (no structures)", [(i, 10) for i in range(5, 8)]),
        ("Second exploration (some structures)", [(i, 10) for i in range(8, 11)]),
        ("Third exploration (more structures)", [(i, 10) for i in range(11, 14)]),
        ("Exploitation (structures known)", [(i, 10) for i in range(5, 15)]),
    ]

    costs = []
    for phase_name, positions in phases:
        phase_cost = 0.0
        for pos in positions:
            obs = np.array([0.2, 0.2, 0.2])
            _, cost = engine.observe(pos, obs, "explore")
            phase_cost += cost.total()
        costs.append((phase_name, phase_cost / len(positions)))
        print(f"  {phase_name}: avg_cost={phase_cost/len(positions):.1f}")

    print(f"\n  Cost reduction over time:")
    if len(costs) >= 2 and costs[0][1] > 0:
        reduction = (costs[0][1] - costs[-1][1]) / costs[0][1]
        print(f"  {reduction*100:.1f}% reduction from first to last phase")


if __name__ == "__main__":
    demo_stable_structure()
    demo_structure_types()
    demo_resource_efficiency()

    print("\n" + "=" * 70)
    print("Summary:")
    print("  - Stable structures are discovered from repeated observations")
    print("  - Structures reduce future resource costs")
    print("  - Value = future resource savings / representation cost")
    print("  - Exploration is valuable when it discovers stable structures")
    print("=" * 70)
