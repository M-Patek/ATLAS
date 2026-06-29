"""
Neural Natural Gradient - KFAC Implementation
神经自然梯度 - KFAC近似实现

基于Kronecker-Factored Approximate Curvature的自然梯度计算。
核心思想：将Fisher信息矩阵近似为两个较小矩阵的Kronecker积。

数学框架:
- Fisher信息矩阵: F = E[∇log p · ∇log p^T]
- KFAC近似: F ≈ A ⊗ G
  - A: 激活协方差矩阵 (输入维度)
  - G: 梯度协方差矩阵 (输出维度)
- 自然梯度: Δθ = F^{-1} ∇L ≈ (A^{-1} ⊗ G^{-1}) ∇L
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy.linalg as la


@dataclass
class KroneckerFactors:
    """Kronecker因子对 (A, G)"""
    A: np.ndarray  # 激活协方差 (input_dim x input_dim)
    G: np.ndarray  # 梯度协方差 (output_dim x output_dim)
    damping: float = 0.001  # 阻尼系数

    def compute_inverse(self) -> Tuple[np.ndarray, np.ndarray]:
        """计算A^{-1}和G^{-1}（带阻尼）"""
        # 添加阻尼项确保可逆
        A_damped = self.A + self.damping * np.eye(self.A.shape[0])
        G_damped = self.G + self.damping * np.eye(self.G.shape[0])

        try:
            A_inv = la.inv(A_damped)
            G_inv = la.inv(G_damped)
        except la.LinAlgError:
            # 使用伪逆作为后备
            A_inv = la.pinv(A_damped)
            G_inv = la.pinv(G_damped)

        return A_inv, G_inv

    def apply_inverse_to_gradient(self, grad: np.ndarray) -> np.ndarray:
        """
        应用逆Fisher矩阵到梯度: (A^{-1} ⊗ G^{-1}) vec(grad)

        利用Kronecker积性质避免显式构造大矩阵:
        (A ⊗ G) vec(X) = vec(G X A^T)
        因此: (A^{-1} ⊗ G^{-1}) vec(grad) = vec(G^{-1} grad A^{-T})
        """
        A_inv, G_inv = self.compute_inverse()

        # 将梯度reshape为矩阵形式
        orig_shape = grad.shape

        # 确保维度匹配
        # 对于权重矩阵 (out_dim, in_dim)，其中 A 是 (in_dim, in_dim)，G 是 (out_dim, out_dim)
        if G_inv.shape[0] == grad.shape[0] and A_inv.shape[0] == grad.shape[1]:
            grad_matrix = grad
        elif G_inv.shape[0] == grad.shape[1] and A_inv.shape[0] == grad.shape[0]:
            # 转置以匹配维度
            grad_matrix = grad.T
        else:
            # 如果维度不匹配，返回原始梯度
            return grad

        # 应用 (A^{-1} ⊗ G^{-1}) vec(grad) = vec(G^{-1} grad A^{-T})
        result = G_inv @ grad_matrix @ A_inv.T

        return result.reshape(orig_shape)


class LayerWiseNaturalGradient:
    """
    层-wise自然梯度计算

    对每一层维护独立的(A, G)因子。
    """

    def __init__(self, layer_shapes: Dict[str, Tuple[Tuple[int, int], Tuple[int]]],
                 damping: float = 0.001,
                 update_freq: int = 10):
        """
        Args:
            layer_shapes: {layer_name: ((in_dim, out_dim), (out_dim,))}
            damping: 阻尼系数
            update_freq: Fisher矩阵更新频率
        """
        self.layer_shapes = layer_shapes
        self.damping = damping
        self.update_freq = update_freq

        # 初始化Kronecker因子
        self.factors: Dict[str, KroneckerFactors] = {}
        self._init_factors()

        # 统计信息
        self.update_count = 0
        self.step_count = 0

    def _init_factors(self):
        """初始化因子矩阵"""
        for name, (w_shape, b_shape) in self.layer_shapes.items():
            in_dim, out_dim = w_shape

            # 初始化A和G为单位矩阵
            A = np.eye(in_dim)
            G = np.eye(out_dim)

            self.factors[name] = KroneckerFactors(A, G, self.damping)

    def update_factors(self,
                       layer_name: str,
                       activations: np.ndarray,
                       output_gradients: np.ndarray,
                       momentum: float = 0.9):
        """
        更新Kronecker因子

        Args:
            layer_name: 层名称
            activations: 输入激活 (batch_size, in_dim)
            output_gradients: 输出梯度 (batch_size, out_dim)
            momentum: 移动平均动量
        """
        if layer_name not in self.factors:
            return

        # 计算激活协方差: A = E[a a^T]
        # 批处理平均
        A_new = (activations.T @ activations) / activations.shape[0]

        # 计算梯度协方差: G = E[∇h ∇h^T]
        G_new = (output_gradients.T @ output_gradients) / output_gradients.shape[0]

        # 移动平均更新
        factors = self.factors[layer_name]
        factors.A = momentum * factors.A + (1 - momentum) * A_new
        factors.G = momentum * factors.G + (1 - momentum) * G_new

        self.update_count += 1

    def compute_natural_gradient(self,
                                  layer_name: str,
                                  weight_grad: np.ndarray,
                                  bias_grad: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        计算自然梯度

        Args:
            layer_name: 层名称
            weight_grad: 权重梯度
            bias_grad: 偏置梯度（可选）

        Returns:
            (nat_weight_grad, nat_bias_grad)
        """
        if layer_name not in self.factors:
            return weight_grad, bias_grad

        factors = self.factors[layer_name]

        # 应用逆Fisher到权重梯度
        nat_weight_grad = factors.apply_inverse_to_gradient(weight_grad)

        # 应用逆Fisher到偏置梯度
        nat_bias_grad = None
        if bias_grad is not None:
            # 对于偏置，A是1x1（因为输入是常数1）
            # 简化为: nat_bias = G^{-1} bias_grad
            _, G_inv = factors.compute_inverse()
            nat_bias_grad = G_inv @ bias_grad

        return nat_weight_grad, nat_bias_grad


class NeuralNaturalGradient:
    """
    神经自然梯度优化器

    使用KFAC近似实现自然梯度下降。
    适用于神经网络参数化空间的端到端学习。

    Example:
        # 定义网络结构
        layer_shapes = {
            'fc1': ((784, 256), (256,)),
            'fc2': ((256, 128), (128,)),
            'fc3': ((128, 10), (10,))
        }

        # 创建优化器
        nng = NeuralNaturalGradient(layer_shapes, damping=0.001)

        # 训练循环
        for batch in data_loader:
            # 前向传播
            activations = forward(batch)

            # 反向传播
            grads = backward(loss)

            # 更新Fisher因子
            nng.update(model, batch)

            # 自然梯度更新
            nng.step(model, grads, lr=0.01)
    """

    def __init__(self,
                 model: Optional[Any] = None,
                 damping: float = 0.001,
                 update_freq: int = 10,
                 lr_scale: float = 1.0):
        """
        Args:
            model: 神经网络模型（可选，用于自动推断结构）
            damping: Fisher矩阵阻尼系数
            update_freq: Fisher矩阵更新频率（每N步）
            lr_scale: 学习率缩放因子
        """
        self.damping = damping
        self.update_freq = update_freq
        self.lr_scale = lr_scale

        # 层-wise自然梯度计算器
        self.layer_wise_ng: Optional[LayerWiseNaturalGradient] = None

        # 激活和梯度缓存
        self.activations: Dict[str, np.ndarray] = {}
        self.output_gradients: Dict[str, np.ndarray] = {}

        # 统计
        self.step_count = 0
        self.fisher_update_count = 0
        self.gradient_norms: List[float] = []

        if model is not None:
            self._init_from_model(model)

    def _init_from_model(self, model: Any):
        """从模型自动推断层结构"""
        layer_shapes = {}

        # 假设模型有layers属性
        if hasattr(model, 'layers'):
            for i, layer in enumerate(model.layers):
                if hasattr(layer, 'W') and hasattr(layer, 'b'):
                    name = f'layer_{i}'
                    layer_shapes[name] = (layer.W.shape, layer.b.shape)

        # 或者假设模型有get_parameters方法
        elif hasattr(model, 'get_parameters'):
            params = model.get_parameters()
            for name, param in params.items():
                if 'W' in name:
                    layer_name = name.replace('_W', '')
                    w_shape = param.shape
                    b_shape = (w_shape[1],)
                    layer_shapes[layer_name] = (w_shape, b_shape)

        if layer_shapes:
            self.layer_wise_ng = LayerWiseNaturalGradient(
                layer_shapes, self.damping, self.update_freq
            )

    def register_layer(self, name: str, input_dim: int, output_dim: int):
        """手动注册层"""
        if self.layer_wise_ng is None:
            self.layer_wise_ng = LayerWiseNaturalGradient(
                {}, self.damping, self.update_freq
            )

        self.layer_wise_ng.layer_shapes[name] = ((input_dim, output_dim), (output_dim,))
        self.layer_wise_ng._init_factors()

    def cache_activations(self, layer_name: str, activations: np.ndarray):
        """缓存层的激活值"""
        self.activations[layer_name] = activations

    def cache_output_gradients(self, layer_name: str, gradients: np.ndarray):
        """缓存层的输出梯度"""
        self.output_gradients[layer_name] = gradients

    def update(self, model: Optional[Any] = None, data: Optional[np.ndarray] = None):
        """
        更新Fisher信息矩阵估计

        这应该在每个训练步骤调用，但实际更新按update_freq间隔执行。
        """
        self.step_count += 1

        # 按频率更新
        if self.step_count % self.update_freq != 0:
            return

        if self.layer_wise_ng is None:
            return

        # 使用缓存的激活和梯度更新因子
        for layer_name in self.activations.keys():
            if layer_name in self.output_gradients:
                acts = self.activations[layer_name]
                grads = self.output_gradients[layer_name]

                self.layer_wise_ng.update_factors(
                    layer_name, acts, grads
                )

        self.fisher_update_count += 1

    def compute_natural_gradient(self,
                                  layer_name: str,
                                  weight_grad: np.ndarray,
                                  bias_grad: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """计算指定层的自然梯度"""
        if self.layer_wise_ng is None:
            return weight_grad, bias_grad

        return self.layer_wise_ng.compute_natural_gradient(
            layer_name, weight_grad, bias_grad
        )

    def step(self,
             model: Any,
             gradients: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]],
             lr: float = 0.01) -> Dict[str, float]:
        """
        执行自然梯度更新

        Args:
            model: 神经网络模型（需支持set_parameters或类似方法）
            gradients: {layer_name: (weight_grad, bias_grad)}
            lr: 学习率

        Returns:
            更新统计信息
        """
        if self.layer_wise_ng is None:
            return {'error': 'Layer-wise NG not initialized'}

        scaled_lr = lr * self.lr_scale
        update_stats = {}

        natural_gradients = {}

        # 计算自然梯度
        for layer_name, (w_grad, b_grad) in gradients.items():
            nat_w_grad, nat_b_grad = self.compute_natural_gradient(
                layer_name, w_grad, b_grad
            )
            natural_gradients[layer_name] = (nat_w_grad, nat_b_grad)

            # 记录梯度范数
            w_norm = np.linalg.norm(nat_w_grad)
            update_stats[f'{layer_name}_grad_norm'] = float(w_norm)

        # 应用更新
        if hasattr(model, 'layers'):
            for i, layer in enumerate(model.layers):
                layer_name = f'layer_{i}'
                if layer_name in natural_gradients:
                    nat_w_grad, nat_b_grad = natural_gradients[layer_name]

                    # 更新权重
                    if hasattr(layer, 'W'):
                        layer.W -= scaled_lr * nat_w_grad

                    # 更新偏置
                    if hasattr(layer, 'b') and nat_b_grad is not None:
                        layer.b -= scaled_lr * nat_b_grad

        elif hasattr(model, 'set_parameters'):
            # 通过参数接口更新
            new_params = {}
            for layer_name, (nat_w_grad, nat_b_grad) in natural_gradients.items():
                new_params[f'{layer_name}_W'] = -scaled_lr * nat_w_grad
                if nat_b_grad is not None:
                    new_params[f'{layer_name}_b'] = -scaled_lr * nat_b_grad

            model.set_parameters(new_params)

        update_stats['learning_rate'] = scaled_lr
        update_stats['step'] = self.step_count

        return update_stats

    def get_fisher_information(self, layer_name: Optional[str] = None) -> Optional[np.ndarray]:
        """
        获取Fisher信息矩阵（或其近似）

        注意：返回的是A ⊗ G的近似，不是完整矩阵（太大）
        """
        if self.layer_wise_ng is None:
            return None

        if layer_name is None:
            # 返回所有层的Fisher迹之和
            total_trace = 0.0
            for name, factors in self.layer_wise_ng.factors.items():
                # tr(A ⊗ G) = tr(A) * tr(G)
                trace = np.trace(factors.A) * np.trace(factors.G)
                total_trace += trace
            return np.array([total_trace])

        if layer_name not in self.layer_wise_ng.factors:
            return None

        factors = self.layer_wise_ng.factors[layer_name]
        # 返回A和G的迹作为Fisher信息的代理
        return np.array([np.trace(factors.A), np.trace(factors.G)])

    def get_statistics(self) -> Dict[str, Any]:
        """获取优化器统计信息"""
        return {
            'step_count': self.step_count,
            'fisher_update_count': self.fisher_update_count,
            'damping': self.damping,
            'update_freq': self.update_freq,
            'num_layers': len(self.layer_wise_ng.factors) if self.layer_wise_ng else 0,
            'avg_gradient_norm': np.mean(self.gradient_norms) if self.gradient_norms else 0.0,
        }


class AmortizedNaturalGradient:
    """
    摊销自然梯度

    通过神经网络学习自然梯度的近似，避免每次迭代都计算Fisher矩阵。
    适用于在线学习场景。
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2):
        """
        Args:
            input_dim: 梯度向量的维度
            hidden_dim: 隐层维度
            num_layers: 网络层数
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 构建网络
        self.weights = []
        self.biases = []

        dims = [input_dim] + [hidden_dim] * num_layers + [input_dim]
        for i in range(len(dims) - 1):
            # Xavier初始化
            scale = np.sqrt(2.0 / (dims[i] + dims[i+1]))
            self.weights.append(np.random.randn(dims[i], dims[i+1]) * scale)
            self.biases.append(np.zeros(dims[i+1]))

        # 训练缓冲区
        self.training_buffer: List[Tuple[np.ndarray, np.ndarray]] = []
        self.max_buffer_size = 1000

    def forward(self, gradient: np.ndarray) -> np.ndarray:
        """
        前向传播: 原始梯度 -> 预处理梯度

        这学习了一个预处理器，近似F^{-1/2}的作用
        """
        x = gradient.flatten()

        # 确保维度匹配
        if len(x) < self.input_dim:
            x = np.pad(x, (0, self.input_dim - len(x)))
        elif len(x) > self.input_dim:
            x = x[:self.input_dim]

        activations = [x]

        # 隐藏层
        for i in range(len(self.weights) - 1):
            x = x @ self.weights[i] + self.biases[i]
            x = np.maximum(0, x)  # ReLU
            activations.append(x)

        # 输出层
        x = x @ self.weights[-1] + self.biases[-1]

        return x

    def preprocess_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """预处理梯度（摊销的自然梯度近似）"""
        return self.forward(gradient)

    def train_step(self,
                   raw_gradient: np.ndarray,
                   true_natural_gradient: np.ndarray,
                   lr: float = 0.001) -> float:
        """
        训练步骤: 学习近似自然梯度

        Args:
            raw_gradient: 原始梯度
            true_natural_gradient: 真实自然梯度（从KFAC计算）
            lr: 学习率

        Returns:
            损失值
        """
        # 前向
        pred_ng = self.forward(raw_gradient)

        # 损失: MSE
        loss = np.mean((pred_ng - true_natural_gradient.flatten()) ** 2)

        # 添加到训练缓冲区
        self.training_buffer.append((raw_gradient.copy(), true_natural_gradient.copy()))
        if len(self.training_buffer) > self.max_buffer_size:
            self.training_buffer.pop(0)

        # 简化更新（实际应该用反向传播）
        # 这里使用简单的梯度下降作为演示
        grad_error = 2 * (pred_ng - true_natural_gradient.flatten())

        # 更新最后一层
        self.weights[-1] -= lr * np.outer(self.weights[-2].T @ raw_gradient[:self.weights[-2].shape[0]], grad_error)[:self.weights[-1].shape[0], :self.weights[-1].shape[1]]

        return float(loss)


# ============================================================================
# 辅助函数
# ============================================================================

def compute_fisher_information_matrix(log_likelihood_fn: Callable,
                                       params: np.ndarray,
                                       data: np.ndarray,
                                       num_samples: int = 100) -> np.ndarray:
    """
    经验Fisher信息矩阵的蒙特卡洛估计

    F = (1/N) Σ ∇log p(x_i|θ) ∇log p(x_i|θ)^T

    Args:
        log_likelihood_fn: 对数似然函数 log p(x|θ)
        params: 当前参数
        data: 数据样本
        num_samples: 用于估计的样本数

    Returns:
        Fisher信息矩阵估计
    """
    n_params = len(params)
    fisher = np.zeros((n_params, n_params))

    # 采样子集
    indices = np.random.choice(len(data), min(num_samples, len(data)), replace=False)

    for idx in indices:
        x = data[idx]

        # 数值计算梯度
        eps = 1e-5
        grad = np.zeros(n_params)

        for i in range(n_params):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps

            ll_plus = log_likelihood_fn(x, params_plus)
            ll_minus = log_likelihood_fn(x, params_minus)

            grad[i] = (ll_plus - ll_minus) / (2 * eps)

        # 外积
        fisher += np.outer(grad, grad)

    fisher /= len(indices)

    return fisher


def natural_gradient_step(params: np.ndarray,
                          grad: np.ndarray,
                          fisher: np.ndarray,
                          lr: float = 0.01,
                          damping: float = 0.001) -> np.ndarray:
    """
    执行自然梯度更新

    θ' = θ - lr * F^{-1} ∇L

    Args:
        params: 当前参数
        grad: 梯度
        fisher: Fisher信息矩阵
        lr: 学习率
        damping: 阻尼系数

    Returns:
        更新后的参数
    """
    # 添加阻尼
    fisher_damped = fisher + damping * np.eye(len(fisher))

    try:
        fisher_inv = la.inv(fisher_damped)
    except la.LinAlgError:
        fisher_inv = la.pinv(fisher_damped)

    # 自然梯度方向
    nat_grad = fisher_inv @ grad

    # 更新
    return params - lr * nat_grad


# ============================================================================
# 演示
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Neural Natural Gradient - KFAC Implementation")
    print("=" * 70)

    # 1. 测试Kronecker因子
    print("\n--- Test 1: Kronecker Factors ---")
    A = np.array([[2.0, 0.5], [0.5, 1.5]])
    G = np.array([[3.0, 0.3], [0.3, 2.0]])

    factors = KroneckerFactors(A, G, damping=0.01)

    # 测试梯度
    grad = np.array([[1.0, 0.5], [0.3, 0.8]])
    nat_grad = factors.apply_inverse_to_gradient(grad)

    print(f"Original gradient:\n{grad}")
    print(f"Natural gradient:\n{nat_grad}")

    # 2. 测试层-wise自然梯度
    print("\n--- Test 2: Layer-wise Natural Gradient ---")
    layer_shapes = {
        'fc1': ((10, 5), (5,)),
        'fc2': ((5, 3), (3,))
    }

    lng = LayerWiseNaturalGradient(layer_shapes, damping=0.001)

    # 模拟激活和梯度
    batch_size = 32
    activations_fc1 = np.random.randn(batch_size, 10)
    gradients_fc1 = np.random.randn(batch_size, 5)

    lng.update_factors('fc1', activations_fc1, gradients_fc1)

    # 计算自然梯度
    w_grad = np.random.randn(10, 5)
    b_grad = np.random.randn(5)

    nat_w, nat_b = lng.compute_natural_gradient('fc1', w_grad, b_grad)

    print(f"Weight gradient norm: {np.linalg.norm(w_grad):.4f}")
    print(f"Natural weight gradient norm: {np.linalg.norm(nat_w):.4f}")
    print(f"Bias gradient norm: {np.linalg.norm(b_grad):.4f}")
    print(f"Natural bias gradient norm: {np.linalg.norm(nat_b):.4f}")

    # 3. 测试完整优化器
    print("\n--- Test 3: Neural Natural Gradient Optimizer ---")

    # 简单模型
    class SimpleModel:
        def __init__(self):
            self.layers = [
                type('Layer', (), {'W': np.random.randn(784, 256) * 0.01,
                                   'b': np.zeros(256)}),
                type('Layer', (), {'W': np.random.randn(256, 10) * 0.01,
                                   'b': np.zeros(10)})
            ]

    model = SimpleModel()
    nng = NeuralNaturalGradient(model, damping=0.001, update_freq=5)

    # 模拟训练
    for step in range(20):
        # 模拟梯度
        gradients = {
            'layer_0': (np.random.randn(784, 256), np.random.randn(256)),
            'layer_1': (np.random.randn(256, 10), np.random.randn(10))
        }

        # 缓存激活和输出梯度
        nng.cache_activations('layer_0', np.random.randn(32, 784))
        nng.cache_output_gradients('layer_0', np.random.randn(32, 256))

        nng.cache_activations('layer_1', np.random.randn(32, 256))
        nng.cache_output_gradients('layer_1', np.random.randn(32, 10))

        # 更新Fisher
        nng.update()

        # 执行更新
        stats = nng.step(model, gradients, lr=0.01)

        if step % 5 == 0:
            print(f"Step {step}: {stats}")

    print("\n--- Final Statistics ---")
    print(nng.get_statistics())

    print("\n" + "=" * 70)
    print("Key Insights:")
    print("  - KFAC approximates Fisher as A (x) G (Kronecker product)")
    print("  - Natural gradient: d_theta = (A^{-1} (x) G^{-1}) grad_L")
    print("  - Efficient computation using vec(G X A^T) property")
    print("  - Amortized version learns to approximate F^{-1} online")
    print("=" * 70)
