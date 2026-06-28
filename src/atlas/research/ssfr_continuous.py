"""
SSFR Continuous Extension: Infinite-Dimensional Statistical Manifold
SSFR连续扩展：无限维统计流形

核心思想：
1. 离散SSFR：参数空间 Θ ⊆ R^k（有限维）
2. 连续SSFR：参数空间 Θ ⊆ H（无限维Hilbert空间）

数学框架：
- 泛函分析：Banach空间、Hilbert空间、Sobolev空间
- 核方法：RKHS（再生核Hilbert空间）
- 变分推断：变分Bayes、高斯过程
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy.linalg as la
from scipy.spatial.distance import cdist


# ============================================================================
# 1. 无限维参数空间
# ============================================================================

class InfiniteDimensionalSpace:
    """
    无限维参数空间

    在离散SSFR中，参数是有限维向量 θ ∈ R^k。
    在连续SSFR中，参数是函数 f ∈ H，其中 H 是Hilbert空间。

    关键区别：
    - 有限维：θ = (θ_1, ..., θ_k)
    - 无限维：f(x) 对所有 x ∈ X 定义
    """

    def __init__(self, domain: Tuple[float, float] = (0.0, 1.0)):
        self.domain = domain

    def inner_product(self, f: Callable, g: Callable) -> float:
        """
        L^2内积：<f, g> = ∫ f(x)g(x) dx

        离散近似：Σ f(x_i)g(x_i) Δx
        """
        # 数值积分（梯形法则）
        x = np.linspace(self.domain[0], self.domain[1], 1000)
        fx = np.array([f(xi) for xi in x])
        gx = np.array([g(xi) for xi in x])
        dx = x[1] - x[0]
        return np.trapz(fx * gx, x)

    def norm(self, f: Callable) -> float:
        """L^2范数：||f|| = sqrt(<f, f>)"""
        return np.sqrt(self.inner_product(f, f))


# ============================================================================
# 2. RKHS（再生核Hilbert空间）
# ============================================================================

class RKHS:
    """
    再生核Hilbert空间

    核函数 k(x, x') 定义了函数空间中的内积：
        <f, g>_k = Σ_i Σ_j α_i β_j k(x_i, x_j)

    其中 f(x) = Σ_i α_i k(x, x_i)，g(x) = Σ_j β_j k(x, x_j)

    关键性质：
    1. 再生性：f(x) = <f, k(x, ·)>
    2. 有限表示：无限维函数由有限个系数表示
    """

    def __init__(self, kernel: str = "rbf", bandwidth: float = None):
        self.kernel_type = kernel
        # 自动带宽：如果没有指定，使用数据范围
        self.bandwidth = bandwidth

        # 支持点（有限表示）
        self.support_points: List[np.ndarray] = []
        self.coefficients: List[float] = []

        # 数据范围（用于自动带宽）
        self.data_range = None

    def kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """核函数"""
        if self.kernel_type == "rbf":
            # 高斯核：k(x,y) = exp(-||x-y||^2 / (2σ^2))
            dist_sq = np.sum((x - y)**2)
            return np.exp(-dist_sq / (2 * self.bandwidth**2))
        elif self.kernel_type == "laplace":
            # Laplace核
            dist = np.sqrt(np.sum((x - y)**2))
            return np.exp(-dist / self.bandwidth)
        elif self.kernel_type == "polynomial":
            # 多项式核
            return (1 + np.dot(x, y))**2
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_type}")

    def evaluate(self, x: np.ndarray) -> float:
        """
        在点x处求值：f(x) = Σ_i α_i k(x, x_i)

        这是RKHS的"再生性"：f(x) = <f, k(x, ·)>
        """
        if len(self.support_points) == 0:
            return 0.0

        result = 0.0
        for alpha, x_i in zip(self.coefficients, self.support_points):
            result += alpha * self.kernel(x, x_i)
        return result

    def add_support_point(self, x: np.ndarray, alpha: float = 1.0):
        """添加支持点"""
        self.support_points.append(x.copy())
        self.coefficients.append(alpha)

    def gram_matrix(self) -> np.ndarray:
        """
        Gram矩阵：K_ij = k(x_i, x_j)

        这是RKHS中的"度量"，对应有限维中的内积矩阵。
        """
        n = len(self.support_points)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel(self.support_points[i], self.support_points[j])
        return K

    def norm(self) -> float:
        """
        RKHS范数：||f||_k = sqrt(α^T K α)

        其中 α 是系数向量，K 是Gram矩阵。
        """
        if len(self.coefficients) == 0:
            return 0.0

        alpha = np.array(self.coefficients)
        K = self.gram_matrix()
        return np.sqrt(alpha @ K @ alpha)


# ============================================================================
# 3. 连续SSFR结构
# ============================================================================

class ContinuousSSFR:
    """
    连续SSFR：无限维统计流形上的结构

    与离散SSFR的区别：
    - 离散：θ ∈ R^k（有限参数向量）
    - 连续：f ∈ H（函数，无限维）

    但RKHS允许我们用有限个系数表示无限维函数！
    """

    def __init__(self, domain: Tuple[float, float] = (0.0, 1.0),
                 kernel: str = "rbf", bandwidth: float = None):
        self.domain = domain
        self.rkhs = RKHS(kernel=kernel, bandwidth=bandwidth)

        # 覆盖区域（连续）
        self.coverage_region: List[Tuple[float, float]] = []

        # 使用统计
        self.usage_count = 0
        self.accuracy = 0.0

        # 观测值范围
        self.y_min = float('inf')
        self.y_max = float('-inf')

    def _update_bandwidth(self):
        """根据数据自动更新带宽"""
        if self.rkhs.bandwidth is not None:
            return

        # 自动带宽：数据范围的中位数距离
        if len(self.rkhs.support_points) < 2:
            self.rkhs.bandwidth = 1.0
            return

        points = np.array(self.rkhs.support_points).flatten()
        if len(points.shape) == 1:
            # 一维数据
            distances = []
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    distances.append(abs(points[i] - points[j]))
            if distances:
                self.rkhs.bandwidth = np.median(distances)
            else:
                self.rkhs.bandwidth = 1.0
        else:
            self.rkhs.bandwidth = 1.0

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        """
        预测：f(x) = Σ_i α_i k(x, x_i)

        返回：(预测值, 不确定性)
        """
        mean = self.rkhs.evaluate(x)

        # 不确定性：与最近支持点的距离
        if len(self.rkhs.support_points) == 0:
            uncertainty = 1.0
        else:
            distances = [np.sqrt(np.sum((x - sp)**2))
                        for sp in self.rkhs.support_points]
            min_dist = min(distances)
            uncertainty = 1.0 - np.exp(-min_dist / self.rkhs.bandwidth)

        return mean, uncertainty

    def update(self, x: np.ndarray, observation: float, merge_threshold: float = 0.1):
        """
        更新：添加新的支持点或合并到现有支持点
        """
        # 更新观测值范围
        self.y_min = min(self.y_min, observation)
        self.y_max = max(self.y_max, observation)

        # 计算残差
        pred, _ = self.predict(x)
        residual = observation - pred

        # 检查是否需要添加新支持点
        if len(self.rkhs.support_points) > 0:
            distances = [np.sqrt(np.sum((x - sp)**2))
                        for sp in self.rkhs.support_points]
            min_dist = min(distances)
            min_idx = distances.index(min_dist)

            if min_dist < merge_threshold:
                # 合并到现有支持点：更新系数
                self.rkhs.coefficients[min_idx] += residual
                return

        # 添加新支持点
        self.rkhs.add_support_point(x, alpha=residual)

        # 更新带宽
        self._update_bandwidth()

        # 更新覆盖区域
        self.coverage_region.append((float(x[0]), float(observation)))
        self.usage_count += 1

    @property
    def representation_cost(self) -> int:
        """
        表示成本：支持点的数量 × 每个点的成本

        在连续SSFR中，表示成本是支持点的数量。
        这与离散SSFR中的参数数量对应。
        """
        return len(self.rkhs.support_points) * 32  # 每个支持点32比特

    @property
    def fisher_information_operator(self) -> Callable:
        """
        Fisher信息算子（无限维版本）

        在有限维中：I(θ) = E[∇log p · ∇log p^T]（矩阵）
        在无限维中：I(f) 是算子，作用在函数上

        在RKHS中，Fisher信息算子可以用Gram矩阵近似。
        """
        def operator(g: Callable) -> Callable:
            """
            算子作用：I(f)[g] = Σ_i Σ_j g(x_i) K^{-1}_{ij} k(·, x_j)
            """
            if len(self.rkhs.support_points) == 0:
                return lambda x: 0.0

            K = self.rkhs.gram_matrix()
            try:
                K_inv = la.inv(K)
            except la.LinAlgError:
                K_inv = la.pinv(K)

            # 在支持点处求值
            g_values = np.array([g(xi) for xi in self.rkhs.support_points])

            # 计算 I(f)[g]
            result_coeffs = K_inv @ g_values

            def result_function(x):
                return sum(c * self.rkhs.kernel(x, xi)
                          for c, xi in zip(result_coeffs, self.rkhs.support_points))

            return result_function

        return operator

    @property
    def stability(self) -> float:
        """
        稳定性 = Fisher信息算子的迹 / 维度

        在无限维中，"迹"需要正则化。
        这里用Gram矩阵的迹近似。
        """
        if len(self.rkhs.support_points) == 0:
            return 0.0

        K = self.rkhs.gram_matrix()
        trace = np.trace(K)
        n = len(self.rkhs.support_points)

        # 正则化：除以支持点数量
        return min(1.0, trace / n * 0.1)

    @property
    def value(self) -> float:
        """
        价值 = 信息增益 / 表示成本

        在连续SSFR中：
        - 信息增益 = Fisher信息算子的迹
        - 表示成本 = 支持点数量
        """
        if len(self.rkhs.support_points) == 0:
            return 0.0

        K = self.rkhs.gram_matrix()
        info_gain = np.trace(K)
        cost = self.representation_cost

        if cost == 0:
            return 0.0
        return info_gain / cost


# ============================================================================
# 4. 连续SSFR引擎
# ============================================================================

class ContinuousSSFREngine:
    """
    连续SSFR引擎：管理多个连续结构
    """

    def __init__(self, domain: Tuple[float, float] = (0.0, 1.0)):
        self.domain = domain
        self.structures: List[ContinuousSSFR] = []

    def observe(self, x: np.ndarray, observation: float) -> Tuple[int, float]:
        """
        观测：找到最佳结构或创建新结构

        返回：(结构索引, 预测误差)
        """
        if len(self.structures) == 0:
            # 创建第一个结构
            structure = ContinuousSSFR(domain=self.domain)
            structure.update(x, observation)
            self.structures.append(structure)
            return 0, 0.0

        # 找到最佳匹配的结构
        best_idx = -1
        best_error = float('inf')

        for i, structure in enumerate(self.structures):
            pred, uncertainty = structure.predict(x)
            error = abs(observation - pred)

            # 考虑不确定性
            weighted_error = error + uncertainty

            if weighted_error < best_error:
                best_error = weighted_error
                best_idx = i

        # 如果误差太大，创建新结构
        if best_error > 0.5:
            structure = ContinuousSSFR(domain=self.domain)
            structure.update(x, observation)
            self.structures.append(structure)
            return len(self.structures) - 1, best_error
        else:
            # 更新现有结构
            self.structures[best_idx].update(x, observation)
            return best_idx, best_error

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        """预测：所有结构的加权平均"""
        if len(self.structures) == 0:
            return 0.0, 1.0

        total_weight = 0.0
        weighted_sum = 0.0
        weighted_uncertainty = 0.0

        for structure in self.structures:
            pred, uncertainty = structure.predict(x)
            weight = structure.stability * (1.0 - uncertainty)

            weighted_sum += weight * pred
            weighted_uncertainty += weight * uncertainty
            total_weight += weight

        if total_weight == 0:
            return 0.0, 1.0

        return weighted_sum / total_weight, weighted_uncertainty / total_weight

    @property
    def total_value(self) -> float:
        """总价值"""
        if len(self.structures) == 0:
            return 0.0
        return sum(s.value for s in self.structures) / len(self.structures)


# ============================================================================
# 5. 演示
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SSFR Continuous Extension: Infinite-Dimensional Statistical Manifold")
    print("=" * 70)

    # 创建连续SSFR引擎
    engine = ContinuousSSFREngine(domain=(0.0, 10.0))

    # 生成数据：两个正弦波模式
    print("\n--- Generating Data: Two Sine Wave Patterns ---")

    np.random.seed(42)

    # 模式1：sin(x) + 噪声
    for i in range(20):
        x = np.array([i * 0.5])
        y = np.sin(x[0]) + np.random.randn() * 0.1
        idx, error = engine.observe(x, y)
        print(f"  x={x[0]:.1f}, y={y:.3f}, structure={idx}, error={error:.3f}")

    # 模式2：cos(x) + 噪声（不同区域）
    print("\n--- Switching to Cosine Pattern ---")
    for i in range(20):
        x = np.array([5.0 + i * 0.25])
        y = np.cos(x[0]) + np.random.randn() * 0.1
        idx, error = engine.observe(x, y)
        print(f"  x={x[0]:.1f}, y={y:.3f}, structure={idx}, error={error:.3f}")

    # 总结
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Total structures: {len(engine.structures)}")
    print(f"  Total value: {engine.total_value:.4f}")

    for i, structure in enumerate(engine.structures):
        print(f"\n  Structure {i}:")
        print(f"    Support points: {len(structure.rkhs.support_points)}")
        print(f"    Stability: {structure.stability:.4f}")
        print(f"    Value: {structure.value:.4f}")
        print(f"    Representation cost: {structure.representation_cost} bits")

    # 预测测试
    print("\n" + "=" * 70)
    print("Prediction Test")
    print("=" * 70)

    test_points = np.linspace(0, 10, 11)
    for x in test_points:
        pred, uncertainty = engine.predict(np.array([x]))
        true_sin = np.sin(x)
        true_cos = np.cos(x)
        print(f"  x={x:.1f}: pred={pred:.3f}, uncertainty={uncertainty:.3f}, "
              f"true_sin={true_sin:.3f}, true_cos={true_cos:.3f}")

    print("\n" + "=" * 70)
    print("Key Insights:")
    print("  - Continuous SSFR uses RKHS for infinite-dimensional representation")
    print("  - Support points = finite representation of infinite-dimensional function")
    print("  - Structure discovery = finding optimal support points")
    print("  - Value = information gain / representation cost (same as discrete)")
    print("=" * 70)
