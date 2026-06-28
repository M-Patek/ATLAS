"""
SSFR Continuous Extension: Improved Version
SSFR连续扩展：改进版

改进点：
1. 使用高斯过程回归（GPR）替代简单的RKHS求和
2. 自动带宽选择
3. 正则化防止过拟合
4. 预测时使用完整的后验分布
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Callable
from dataclasses import dataclass
import numpy.linalg as la


class GaussianProcessSSFR:
    """
    高斯过程SSFR：使用GP回归进行连续函数逼近

    改进点：
    1. 使用GP后验分布进行预测
    2. 自动带宽选择
    3. 正则化防止过拟合
    4. 预测不确定性估计
    """

    def __init__(self, kernel: str = "rbf", noise_level: float = 0.1):
        self.kernel_type = kernel
        self.noise_level = noise_level

        # 训练数据
        self.X_train: List[np.ndarray] = []
        self.y_train: List[float] = []

        # 核函数带宽
        self.bandwidth = None

        # 缓存的Gram矩阵
        self.K = None
        self.K_inv = None

    def _kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """核函数"""
        dist_sq = np.sum((x1 - x2)**2)
        return np.exp(-dist_sq / (2 * self.bandwidth**2))

    def _compute_gram_matrix(self):
        """计算Gram矩阵"""
        n = len(self.X_train)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self._kernel(self.X_train[i], self.X_train[j])

        # 添加噪声项
        K += self.noise_level**2 * np.eye(n)
        return K

    def _update_bandwidth(self):
        """自动带宽选择"""
        if len(self.X_train) < 2:
            self.bandwidth = 1.0
            return

        # 使用中位数距离
        distances = []
        for i in range(len(self.X_train)):
            for j in range(i+1, len(self.X_train)):
                dist = np.sqrt(np.sum((self.X_train[i] - self.X_train[j])**2))
                distances.append(dist)

        if distances:
            self.bandwidth = np.median(distances)
        else:
            self.bandwidth = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练GP模型

        X: 输入数据 (n, d)
        y: 输出数据 (n,)
        """
        self.X_train = [X[i] for i in range(len(X))]
        self.y_train = [y[i] for i in range(len(y))]

        # 更新带宽
        self._update_bandwidth()

        # 计算Gram矩阵
        self.K = self._compute_gram_matrix()

        # 计算逆矩阵
        try:
            self.K_inv = la.inv(self.K)
        except la.LinAlgError:
            self.K_inv = la.pinv(self.K)

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        """
        预测：返回均值和方差

        使用GP后验：
        μ(x) = k(x, X) K^{-1} y
        σ²(x) = k(x, x) - k(x, X) K^{-1} k(X, x)
        """
        if len(self.X_train) == 0:
            return 0.0, 1.0

        # 计算k(x, X)
        k_x = np.array([self._kernel(x, xi) for xi in self.X_train])

        # 预测均值
        mean = k_x @ self.K_inv @ np.array(self.y_train)

        # 预测方差
        k_xx = self._kernel(x, x)
        variance = k_xx - k_x @ self.K_inv @ k_x
        variance = max(0.0, variance)  # 确保非负

        return mean, np.sqrt(variance)

    @property
    def representation_cost(self) -> int:
        """表示成本：训练数据数量 × 每个点的成本"""
        return len(self.X_train) * 32

    @property
    def stability(self) -> float:
        """稳定性：基于训练数据量"""
        if len(self.X_train) == 0:
            return 0.0
        return min(1.0, len(self.X_train) / 10.0)

    @property
    def value(self) -> float:
        """价值 = 稳定性 / 表示成本"""
        cost = self.representation_cost
        if cost == 0:
            return 0.0
        return self.stability / cost


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Gaussian Process SSFR Test")
    print("=" * 70)

    # 生成数据
    np.random.seed(42)
    x_train = np.linspace(0, 10, 20)
    y_train = np.sin(x_train) + np.random.randn(20) * 0.1

    # 创建GP-SSFR
    gp = GaussianProcessSSFR(kernel="rbf", noise_level=0.1)
    gp.fit(x_train.reshape(-1, 1), y_train)

    # 预测
    x_test = np.linspace(0, 10, 100)
    predictions = []
    uncertainties = []

    for x in x_test:
        mean, std = gp.predict(np.array([x]))
        predictions.append(mean)
        uncertainties.append(std)

    # 评估
    y_true = np.sin(x_test)
    mse = np.mean((np.array(predictions) - y_true)**2)
    mae = np.mean(np.abs(np.array(predictions) - y_true))

    print(f"\nMSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Stability: {gp.stability:.4f}")
    print(f"Value: {gp.value:.6f}")
    print(f"Representation cost: {gp.representation_cost} bits")

    # 与多项式拟合比较
    coeffs = np.polyfit(x_train, y_train, 5)
    poly_preds = np.polyval(coeffs, x_test)
    poly_mse = np.mean((poly_preds - y_true)**2)

    print(f"\nPolynomial MSE: {poly_mse:.6f}")
    print(f"GP improvement: {((poly_mse - mse) / poly_mse * 100):+.1f}%")
