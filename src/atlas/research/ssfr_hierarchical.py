"""
SSFR Hierarchical Structures: Fiber Bundle and Meta-Patterns
SSFR层次结构：纤维丛与Meta-Patterns

核心思想：
1. 基础层：原始观测模式（如"走廊"）
2. 元层：模式的模式（如"线性结构"）
3. 元元层：元模式的模式（如"几何规律"）

数学框架：
- 纤维丛：E = (B, F, π)
  - B: 基空间（模式类型）
  - F: 纤维（具体实例）
  - π: 投影（实例→类型）
- 层次信息几何：每层有自己的统计流形
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy.linalg as la


# ============================================================================
# 1. 纤维丛结构
# ============================================================================

class FiberBundle:
    """
    纤维丛：E = (Base, Fiber, Projection)

    在SSFR中：
    - 基空间 B：模式类型（如"线性"、"周期"、"随机"）
    - 纤维 F：具体实例（如"走廊A"、"走廊B"）
    - 投影 π：将实例映射到类型

    示例：
    - 基空间：{线性, 周期, 随机}
    - 纤维（线性）：{走廊, 道路, 通道}
    - 投影：走廊 → 线性
    """

    def __init__(self):
        # 基空间：模式类型
        self.base_space: Set[str] = set()

        # 纤维：每个基点的纤维
        self.fibers: Dict[str, List['MetaPattern']] = defaultdict(list)

        # 投影映射
        self.projection: Dict[int, str] = {}  # pattern_id -> base_type

    def add_base_point(self, pattern_type: str):
        """添加基点（模式类型）"""
        self.base_space.add(pattern_type)

    def add_fiber_point(self, pattern: 'MetaPattern', pattern_type: str):
        """添加纤维点（具体实例）"""
        self.fibers[pattern_type].append(pattern)
        self.projection[id(pattern)] = pattern_type

    def get_fiber(self, pattern_type: str) -> List['MetaPattern']:
        """获取某类型的所有实例"""
        return self.fibers.get(pattern_type, [])

    def project(self, pattern: 'MetaPattern') -> str:
        """投影：实例 → 类型"""
        return self.projection.get(id(pattern), "unknown")


# ============================================================================
# 2. Meta-Pattern（模式的模式）
# ============================================================================

@dataclass
class MetaPattern:
    """
    元模式：描述一组模式的共同结构

    属性：
    - pattern_type: 模式类型（如"线性"）
    - abstraction_level: 抽象层级（0=原始，1=元，2=元元）
    - common_features: 共同特征（统计量）
    - instances: 实例列表
    - meta_pattern: 更高层的元模式（递归）
    """

    pattern_type: str
    abstraction_level: int
    common_features: np.ndarray = field(default_factory=lambda: np.zeros(0))
    instances: List[int] = field(default_factory=list)  # 实例ID列表
    meta_pattern: Optional['MetaPattern'] = None

    # 统计量
    feature_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    feature_cov: np.ndarray = field(default_factory=lambda: np.eye(1))

    def similarity(self, other: 'MetaPattern') -> float:
        """
        两个元模式的相似度

        基于共同特征的余弦相似度
        """
        if len(self.common_features) == 0 or len(other.common_features) == 0:
            return 0.0

        # 归一化
        f1 = self.common_features / (np.linalg.norm(self.common_features) + 1e-10)
        f2 = other.common_features / (np.linalg.norm(other.common_features) + 1e-10)

        return np.dot(f1, f2)

    @property
    def stability(self) -> float:
        """
        稳定性 = 实例数量 × 特征一致性

        实例越多、特征越一致，稳定性越高
        """
        if len(self.instances) == 0:
            return 0.0

        # 特征一致性（协方差矩阵的迹的倒数）
        trace = np.trace(self.feature_cov)
        consistency = 1.0 / (1.0 + trace)

        # 实例数量（对数缩放）
        count_factor = np.log(1 + len(self.instances)) / 5.0

        return min(1.0, consistency * count_factor)

    @property
    def value(self) -> float:
        """
        价值 = 稳定性 / 表示成本

        表示成本 = 特征维度 × 32比特
        """
        cost = len(self.common_features) * 32
        if cost == 0:
            return 0.0
        return self.stability / cost


# ============================================================================
# 3. 层次结构引擎
# ============================================================================

class HierarchicalSSFREngine:
    """
    层次SSFR引擎：管理多层结构

    层级：
    - Level 0: 原始观测模式（如"走廊"）
    - Level 1: 元模式（如"线性结构"）
    - Level 2: 元元模式（如"几何规律"）
    - Level N: 更高层...

    每层都是信息几何的一个子流形。
    """

    def __init__(self, max_levels: int = 3):
        self.max_levels = max_levels

        # 每层的数据结构
        self.levels: Dict[int, Dict[int, MetaPattern]] = {
            i: {} for i in range(max_levels)
        }

        # 纤维丛
        self.fiber_bundle = FiberBundle()

        # 层级映射：下层 → 上层
        self.upward_mapping: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # 统计
        self.pattern_counter = 0

    def add_pattern(self, features: np.ndarray, level: int = 0) -> int:
        """
        添加模式到指定层级

        如果层级 > 0，自动聚合下层模式
        """
        pattern_id = self.pattern_counter
        self.pattern_counter += 1

        if level == 0:
            # 原始模式
            pattern = MetaPattern(
                pattern_type=f"pattern_{pattern_id}",
                abstraction_level=0,
                common_features=features,
                instances=[pattern_id]
            )
            self.levels[0][pattern_id] = pattern

        else:
            # 元模式：从下层聚合
            lower_patterns = list(self.levels[level - 1].values())

            if len(lower_patterns) == 0:
                return -1

            # 找到最相似的实例，聚合成元模式
            pattern_type = self._infer_pattern_type(lower_patterns)

            # 计算共同特征
            all_features = np.array([p.common_features for p in lower_patterns])
            mean_features = np.mean(all_features, axis=0)
            cov_features = np.cov(all_features.T) if len(all_features) > 1 else np.eye(len(mean_features))

            pattern = MetaPattern(
                pattern_type=pattern_type,
                abstraction_level=level,
                common_features=mean_features,
                instances=[id(p) for p in lower_patterns],
                feature_mean=mean_features,
                feature_cov=cov_features
            )

            self.levels[level][pattern_id] = pattern

            # 更新纤维丛
            self.fiber_bundle.add_base_point(pattern_type)
            self.fiber_bundle.add_fiber_point(pattern, pattern_type)

            # 更新向上映射
            for lower in lower_patterns:
                self.upward_mapping[(level - 1, id(lower))] = (level, pattern_id)

        return pattern_id

    def _infer_pattern_type(self, patterns: List[MetaPattern]) -> str:
        """
        推断模式类型

        基于特征的统计特性：
        - 线性：特征变化均匀
        - 周期：特征重复
        - 随机：特征无规律
        """
        if len(patterns) < 2:
            return "singleton"

        # 提取特征序列
        features = np.array([p.common_features for p in patterns])

        if len(features.shape) == 1:
            features = features.reshape(-1, 1)

        # 计算变化率
        if len(features) > 1:
            diffs = np.diff(features, axis=0)
            mean_diff = np.mean(np.abs(diffs))
            std_diff = np.std(np.abs(diffs))

            # 判断类型
            if std_diff < 0.1 * mean_diff:
                return "linear"
            elif self._check_periodicity(features):
                return "periodic"
            else:
                return "random"
        else:
            return "singleton"

    def _check_periodicity(self, features: np.ndarray) -> bool:
        """检查周期性"""
        if len(features) < 4:
            return False

        # 简化的周期检测：检查自相关
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)

        # 计算一维特征的周期
        if features.shape[1] == 1:
            f = features.flatten()
            # 检查是否有重复模式
            for period in range(2, len(f) // 2):
                correlation = np.corrcoef(f[:-period], f[period:])[0, 1]
                if correlation > 0.8:
                    return True

        return False

    def get_meta_pattern(self, pattern_id: int, level: int) -> Optional[MetaPattern]:
        """获取指定层级的元模式"""
        return self.levels[level].get(pattern_id)

    def get_hierarchy(self, pattern_id: int, level: int = 0) -> List[MetaPattern]:
        """
        获取模式的完整层级链

        从底层向上追溯
        """
        hierarchy = []
        current_id = pattern_id
        current_level = level

        while current_level < self.max_levels:
            pattern = self.levels[current_level].get(current_id)
            if pattern is None:
                break

            hierarchy.append(pattern)

            # 向上查找
            upward = self.upward_mapping.get((current_level, current_id))
            if upward is None:
                break

            current_level, current_id = upward

        return hierarchy

    def cross_level_similarity(self, pattern1_id: int, level1: int,
                               pattern2_id: int, level2: int) -> float:
        """
        跨层级相似度

        如果两个模式在不同层级，先提升到同一层级再比较
        """
        # 获取层级链
        hierarchy1 = self.get_hierarchy(pattern1_id, level1)
        hierarchy2 = self.get_hierarchy(pattern2_id, level2)

        # 找到共同层级
        types1 = {p.abstraction_level: p for p in hierarchy1}
        types2 = {p.abstraction_level: p for p in hierarchy2}

        common_levels = set(types1.keys()) & set(types2.keys())

        if len(common_levels) == 0:
            return 0.0

        # 在最高共同层级比较
        max_level = max(common_levels)
        return types1[max_level].similarity(types2[max_level])

    @property
    def total_value(self) -> float:
        """总价值（所有层级）"""
        total = 0.0
        count = 0
        for level in self.levels.values():
            for pattern in level.values():
                total += pattern.value
                count += 1
        return total / max(1, count)

    def print_hierarchy(self, pattern_id: int, level: int = 0):
        """打印层级结构"""
        hierarchy = self.get_hierarchy(pattern_id, level)

        print(f"\nHierarchy for pattern {pattern_id} (level {level}):")
        for i, pattern in enumerate(hierarchy):
            indent = "  " * i
            print(f"{indent}Level {pattern.abstraction_level}: {pattern.pattern_type}")
            print(f"{indent}  Stability: {pattern.stability:.4f}")
            print(f"{indent}  Value: {pattern.value:.4f}")
            print(f"{indent}  Instances: {len(pattern.instances)}")


# ============================================================================
# 4. 层次信息几何
# ============================================================================

class HierarchicalInformationGeometry:
    """
    层次信息几何

    每层都是信息几何的一个子流形。
    层与层之间通过"粗粒化"映射连接。

    数学结构：
    - Level 0: M_0 = {p(·|θ_0)}（精细流形）
    - Level 1: M_1 = {p(·|θ_1)}（粗粒化流形）
    - 投影 π: M_0 → M_1（粗粒化）
    - 提升 ι: M_1 → M_0（细化）
    """

    def __init__(self, n_levels: int = 3):
        self.n_levels = n_levels

        # 每层的统计流形
        self.manifolds: Dict[int, Dict] = {}

        # 粗粒化映射
        self.coarsening: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def add_point(self, level: int, theta: np.ndarray, info: Dict):
        """添加点到指定层级的流形"""
        if level not in self.manifolds:
            self.manifolds[level] = {}

        point_id = len(self.manifolds[level])
        self.manifolds[level][point_id] = {
            'theta': theta,
            'info': info
        }

        return point_id

    def coarse_grain(self, level: int, point_id: int) -> Optional[Tuple[int, int]]:
        """
        粗粒化：从精细层到粗粒层

        π: M_level → M_{level+1}

        粗粒化 = 聚合相似点
        """
        if level >= self.n_levels - 1:
            return None

        point = self.manifolds[level].get(point_id)
        if point is None:
            return None

        # 简化的粗粒化：平均附近点
        theta = point['theta']

        # 在下一层创建粗粒化点
        next_level = level + 1
        coarse_theta = theta * 0.5  # 简化：缩放

        coarse_id = self.add_point(next_level, coarse_theta, {
            'source_level': level,
            'source_id': point_id
        })

        self.coarsening[(level, point_id)] = (next_level, coarse_id)

        return (next_level, coarse_id)

    def fisher_information_hierarchical(self, level: int, point_id: int) -> np.ndarray:
        """
        层次Fisher信息

        I_level(θ) = E[∇_θ log p_level · ∇_θ log p_level^T]

        高层的信息是低层信息的粗粒化。
        """
        point = self.manifolds[level].get(point_id)
        if point is None:
            return np.eye(1)

        theta = point['theta']

        # 简化的Fisher信息（单位矩阵缩放）
        dim = len(theta)
        return np.eye(dim) * (1.0 / (level + 1))


# ============================================================================
# 5. 演示
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SSFR Hierarchical Structures: Fiber Bundle and Meta-Patterns")
    print("=" * 70)

    # 创建层次引擎
    engine = HierarchicalSSFREngine(max_levels=3)

    # 添加原始模式（Level 0）
    print("\n--- Adding Level 0 Patterns (Raw Observations) ---")

    # 模式1：走廊（线性）
    for i in range(5):
        features = np.array([i * 0.2, 0.5, 0.1])  # 线性变化
        pid = engine.add_pattern(features, level=0)
        print(f"  Pattern {pid}: features={features}")

    # 模式2：房间（随机）
    for i in range(5):
        features = np.array([np.random.randn(), np.random.randn(), 0.5])
        pid = engine.add_pattern(features, level=0)
        print(f"  Pattern {pid}: features={features}")

    # 添加元模式（Level 1）
    print("\n--- Adding Level 1 Meta-Patterns ---")
    meta_id1 = engine.add_pattern(np.zeros(3), level=1)
    print(f"  Meta-pattern {meta_id1}: type={engine.levels[1][meta_id1].pattern_type}")

    # 添加元元模式（Level 2）
    print("\n--- Adding Level 2 Meta-Meta-Patterns ---")
    meta_meta_id = engine.add_pattern(np.zeros(3), level=2)
    print(f"  Meta-meta-pattern {meta_meta_id}: type={engine.levels[2][meta_meta_id].pattern_type}")

    # 打印层级结构
    print("\n" + "=" * 70)
    print("Hierarchy Analysis")
    print("=" * 70)

    for pid in list(engine.levels[0].keys())[:3]:
        engine.print_hierarchy(pid, level=0)

    # 统计
    print("\n" + "=" * 70)
    print("Statistics")
    print("=" * 70)

    for level in range(3):
        patterns = engine.levels[level]
        if len(patterns) > 0:
            avg_stability = np.mean([p.stability for p in patterns.values()])
            avg_value = np.mean([p.value for p in patterns.values()])
            print(f"  Level {level}: {len(patterns)} patterns")
            print(f"    Avg stability: {avg_stability:.4f}")
            print(f"    Avg value: {avg_value:.4f}")

    print(f"\n  Total value: {engine.total_value:.4f}")

    # 纤维丛
    print("\n" + "=" * 70)
    print("Fiber Bundle")
    print("=" * 70)

    print(f"  Base space: {engine.fiber_bundle.base_space}")
    for btype in engine.fiber_bundle.base_space:
        fibers = engine.fiber_bundle.get_fiber(btype)
        print(f"    {btype}: {len(fibers)} instances")

    # 层次信息几何
    print("\n" + "=" * 70)
    print("Hierarchical Information Geometry")
    print("=" * 70)

    hig = HierarchicalInformationGeometry(n_levels=3)

    # 添加点
    for level in range(3):
        theta = np.array([1.0, 0.5]) * (level + 1)
        pid = hig.add_point(level, theta, {'level': level})
        print(f"  Level {level}: theta = {theta}")

        # Fisher信息
        fisher = hig.fisher_information_hierarchical(level, pid)
        print(f"    Fisher info trace: {np.trace(fisher):.4f}")

    print("\n" + "=" * 70)
    print("Key Insights:")
    print("  - Hierarchical SSFR uses fiber bundles for multi-level patterns")
    print("  - Meta-patterns capture common structure across instances")
    print("  - Each level is a submanifold of information geometry")
    print("  - Coarse-graining connects levels (fine -> coarse)")
    print("  - Value at each level = stability / cost")
    print("=" * 70)
