"""
ATLAS Learning: Bayesian Optimization
贝叶斯参数优化

用于自动优化认知空间的参数
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from dataclasses import dataclass
import warnings


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: Dict[str, float]
    best_score: float
    all_trials: List[Dict[str, Any]]
    n_iterations: int


class BayesianOptimizer:
    """
    贝叶斯优化器

    使用高斯过程代理模型优化黑盒函数
    """

    def __init__(self,
                 param_bounds: Dict[str, Tuple[float, float]],
                 n_initial_points: int = 5,
                 acquisition: str = 'ei',  # 'ei', 'ucb', 'pi'
                 xi: float = 0.01,  # EI探索参数
                 kappa: float = 2.0):  # UCB探索参数
        """
        Args:
            param_bounds: 参数范围 {param_name: (min, max)}
            n_initial_points: 初始随机采样点数
            acquisition: 采集函数类型
            xi: EI探索参数
            kappa: UCB探索参数
        """
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.n_initial_points = n_initial_points
        self.acquisition = acquisition
        self.xi = xi
        self.kappa = kappa

        # 历史记录
        self.X: List[np.ndarray] = []  # 参数
        self.y: List[float] = []  # 目标值

        # GP模型
        self.has_sklearn = False
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, WhiteKernel
            self.GaussianProcessRegressor = GaussianProcessRegressor
            self.Matern = Matern
            self.WhiteKernel = WhiteKernel
            self.has_sklearn = True
        except ImportError:
            warnings.warn("scikit-learn not available, using random search fallback")

        self.gp = None
        self._best_params = None
        self._best_score = float('-inf')

    def _normalize(self, params: np.ndarray) -> np.ndarray:
        """归一化参数到 [0, 1]"""
        normalized = np.zeros_like(params)
        for i, name in enumerate(self.param_names):
            low, high = self.param_bounds[name]
            normalized[i] = (params[i] - low) / (high - low)
        return normalized

    def _denormalize(self, normalized: np.ndarray) -> np.ndarray:
        """反归一化"""
        params = np.zeros_like(normalized)
        for i, name in enumerate(self.param_names):
            low, high = self.param_bounds[name]
            params[i] = normalized[i] * (high - low) + low
        return params

    def _fit_gp(self):
        """拟合高斯过程"""
        if not self.has_sklearn or len(self.X) < 2:
            return

        X_array = np.array([self._normalize(x) for x in self.X])
        y_array = np.array(self.y)

        # Matern核 + 白噪声
        kernel = self.Matern(length_scale=0.5, nu=2.5) + self.WhiteKernel(noise_level=0.1)

        self.gp = self.GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=2
        )

        try:
            self.gp.fit(X_array, y_array)
        except Exception as e:
            warnings.warn(f"GP fitting failed: {e}")
            self.gp = None

    def _acquisition_ei(self, x: np.ndarray, y_best: float) -> float:
        """期望改进采集函数"""
        if self.gp is None:
            return np.random.random()

        x_norm = self._normalize(x).reshape(1, -1)
        mu, sigma = self.gp.predict(x_norm, return_std=True)

        sigma = max(sigma[0], 1e-9)

        improvement = mu[0] - y_best - self.xi
        z = improvement / sigma

        from scipy.stats import norm
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        return ei

    def _acquisition_ucb(self, x: np.ndarray) -> float:
        """上置信界采集函数"""
        if self.gp is None:
            return np.random.random()

        x_norm = self._normalize(x).reshape(1, -1)
        mu, sigma = self.gp.predict(x_norm, return_std=True)

        return mu[0] + self.kappa * sigma[0]

    def _suggest_next(self) -> np.ndarray:
        """建议下一个采样点"""
        if len(self.X) < self.n_initial_points or self.gp is None:
            # 随机采样
            return np.array([
                np.random.uniform(low, high)
                for low, high in self.param_bounds.values()
            ])

        # 优化采集函数
        best_x = None
        best_acq = float('-inf')

        # 随机起点优化
        for _ in range(100):
            x0 = np.array([
                np.random.uniform(low, high)
                for low, high in self.param_bounds.values()
            ])

            # 简单的爬山法
            x = x0.copy()
            for _ in range(50):
                # 扰动
                dx = np.random.randn(len(x)) * 0.1
                x_new = x + dx

                # 边界约束
                for i, name in enumerate(self.param_names):
                    low, high = self.param_bounds[name]
                    x_new[i] = np.clip(x_new[i], low, high)

                # 评估
                if self.acquisition == 'ei':
                    acq_new = self._acquisition_ei(x_new, max(self.y))
                    acq_old = self._acquisition_ei(x, max(self.y))
                else:  # ucb
                    acq_new = self._acquisition_ucb(x_new)
                    acq_old = self._acquisition_ucb(x)

                if acq_new > acq_old:
                    x = x_new

            # 记录最佳
            if self.acquisition == 'ei':
                acq = self._acquisition_ei(x, max(self.y))
            else:
                acq = self._acquisition_ucb(x)

            if acq > best_acq:
                best_acq = acq
                best_x = x

        return best_x if best_x is not None else x0

    def optimize(self,
                 objective_fn: Callable[[Dict[str, float]], float],
                 n_iterations: int = 20,
                 verbose: bool = True) -> OptimizationResult:
        """
        执行优化

        Args:
            objective_fn: 目标函数，接收参数字典，返回评分（越高越好）
            n_iterations: 迭代次数
            verbose: 是否打印进度

        Returns:
            OptimizationResult
        """
        trials = []

        for i in range(n_iterations):
            # 建议参数
            x = self._suggest_next()
            params = {name: x[j] for j, name in enumerate(self.param_names)}

            # 评估
            try:
                score = objective_fn(params)
            except Exception as e:
                warnings.warn(f"Objective evaluation failed: {e}")
                score = float('-inf')

            # 记录
            self.X.append(x)
            self.y.append(score)

            trial = {
                'iteration': i,
                'params': params.copy(),
                'score': score
            }
            trials.append(trial)

            # 更新最佳
            if score > self._best_score:
                self._best_score = score
                self._best_params = params.copy()
                if verbose:
                    print(f"  Iter {i}: New best! Score={score:.4f}, Params={params}")
            elif verbose:
                print(f"  Iter {i}: Score={score:.4f}")

            # 更新GP
            if i >= self.n_initial_points - 1:
                self._fit_gp()

        return OptimizationResult(
            best_params=self._best_params or {},
            best_score=self._best_score,
            all_trials=trials,
            n_iterations=n_iterations
        )


class SpaceOptimizer:
    """
    空间参数优化器

    专门用于优化认知空间的参数
    """

    def __init__(self, space_type: str, param_bounds: Dict[str, Tuple[float, float]]):
        self.space_type = space_type
        self.param_bounds = param_bounds
        self.optimizer = BayesianOptimizer(param_bounds)

    def optimize_for_task(self,
                         task_evaluator: Callable[[Any], float],
                         base_space_params: Dict[str, Any],
                         n_iterations: int = 20) -> OptimizationResult:
        """
        为特定任务优化空间参数

        Args:
            task_evaluator: 接收空间实例，返回任务表现评分
            base_space_params: 空间基础参数（不优化的部分）
            n_iterations: 优化迭代次数
        """
        from ..core.registry import create_space

        def objective_fn(params: Dict[str, float]) -> float:
            # 合并参数
            full_params = {**base_space_params, **params}

            # 创建空间
            try:
                space = create_space(self.space_type, **full_params)
            except Exception as e:
                warnings.warn(f"Space creation failed: {e}")
                return float('-inf')

            # 评估
            try:
                score = task_evaluator(space)
            except Exception as e:
                warnings.warn(f"Task evaluation failed: {e}")
                return float('-inf')

            return score

        return self.optimizer.optimize(objective_fn, n_iterations)


class MultiObjectiveOptimizer:
    """
    多目标优化器

    同时优化多个目标（如效率和探索）
    """

    def __init__(self,
                 param_bounds: Dict[str, Tuple[float, float]],
                 objectives: List[str]):
        self.param_bounds = param_bounds
        self.objectives = objectives
        self.optimizers = {
            obj: BayesianOptimizer(param_bounds)
            for obj in objectives
        }

        # Pareto前沿
        self.pareto_front: List[Dict] = []

    def optimize(self,
                 objective_fns: Dict[str, Callable[[Dict[str, float]], float]],
                 n_iterations: int = 30) -> List[Dict]:
        """
        多目标优化

        Returns:
            Pareto前沿解集
        """
        all_solutions = []

        for obj_name, obj_fn in objective_fns.items():
            print(f"\nOptimizing for {obj_name}...")
            result = self.optimizers[obj_name].optimize(obj_fn, n_iterations // len(self.objectives))

            all_solutions.append({
                'params': result.best_params,
                obj_name: result.best_score
            })

        # 计算Pareto前沿
        self.pareto_front = self._compute_pareto_front(all_solutions)
        return self.pareto_front

    def _compute_pareto_front(self, solutions: List[Dict]) -> List[Dict]:
        """计算Pareto前沿"""
        pareto = []

        for s in solutions:
            dominated = False
            for other in solutions:
                if s is other:
                    continue

                # 检查是否被支配
                better_in_all = True
                for obj in self.objectives:
                    if other.get(obj, float('-inf')) <= s.get(obj, float('-inf')):
                        better_in_all = False
                        break

                if better_in_all:
                    dominated = True
                    break

            if not dominated:
                pareto.append(s)

        return pareto
