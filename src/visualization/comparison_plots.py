"""
ATLAS Visualization: Comparison Plots
对比分析图

用于对比不同空间、算法、配置的性能
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class ComparisonPlotter:
    """
    实验结果对比绘图器
    """

    def __init__(self, style: str = 'default'):
        self.style = style

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            self.plt = plt
            self.sns = sns
            self._has_libs = True
        except ImportError:
            self.plt = None
            self._has_libs = False

    def _check_libs(self):
        if not self._has_libs:
            raise ImportError("matplotlib and seaborn required for comparison plots")

    def plot_performance_comparison(self, results: Dict[str, List[Dict]],
                                   metrics: List[str] = ['steps', 'time_ms', 'success'],
                                   save_path: Optional[str] = None,
                                   show: bool = True):
        """
        对比不同配置的性能

        Args:
            results: {
                'config_A': [{'steps': 10, 'time_ms': 5, ...}, ...],
                'config_B': [...],
            }
        """
        self._check_libs()

        n_metrics = len(metrics)
        fig, axes = self.plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]

        configs = list(results.keys())

        for ax, metric in zip(axes, metrics):
            data = []
            labels = []

            for config, trials in results.items():
                values = [t.get(metric, 0) for t in trials if metric in t]
                if values:
                    data.append(values)
                    labels.append(config)

            if data:
                bp = ax.boxplot(data, labels=labels, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.7)

                ax.set_ylabel(metric)
                ax.set_title(f'{metric} by Configuration')
                ax.tick_params(axis='x', rotation=45)

        self.plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            self.plt.show()
        else:
            self.plt.close()

    def plot_pareto_frontier(self, solutions: List[Dict],
                            obj1: str = 'cost',
                            obj2: str = 'time_ms',
                            save_path: Optional[str] = None,
                            show: bool = True):
        """
        绘制Pareto前沿（多目标优化）

        Args:
            solutions: [{'name': 'A', 'cost': 10, 'time_ms': 5}, ...]
        """
        self._check_libs()

        fig, ax = self.plt.subplots(figsize=(8, 6))

        x = [s[obj1] for s in solutions]
        y = [s[obj2] for s in solutions]
        names = [s.get('name', f'S{i}') for i, s in enumerate(solutions)]

        # 绘制所有解
        ax.scatter(x, y, s=100, alpha=0.6)

        # 标注
        for i, name in enumerate(names):
            ax.annotate(name, (x[i], y[i]), xytext=(5, 5),
                       textcoords='offset points', fontsize=9)

        # 计算并绘制Pareto前沿
        pareto = self._compute_pareto_frontier(solutions, obj1, obj2)
        if pareto:
            px = [s[obj1] for s in pareto]
            py = [s[obj2] for s in pareto]
            px, py = zip(*sorted(zip(px, py)))
            ax.plot(px, py, 'r--', linewidth=2, label='Pareto Frontier')

        ax.set_xlabel(obj1)
        ax.set_ylabel(obj2)
        ax.set_title('Pareto Frontier Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            self.plt.show()
        else:
            self.plt.close()

    def _compute_pareto_frontier(self, solutions: List[Dict],
                                 obj1: str, obj2: str) -> List[Dict]:
        """计算Pareto前沿"""
        pareto = []
        for s in solutions:
            dominated = False
            for other in solutions:
                if other is s:
                    continue
                if (other[obj1] <= s[obj1] and other[obj2] <= s[obj2] and
                    (other[obj1] < s[obj1] or other[obj2] < s[obj2])):
                    dominated = True
                    break
            if not dominated:
                pareto.append(s)
        return pareto

    def plot_convergence(self, convergence_data: Dict[str, List[float]],
                        save_path: Optional[str] = None,
                        show: bool = True):
        """
        绘制算法收敛过程
        """
        self._check_libs()

        fig, ax = self.plt.subplots(figsize=(10, 5))

        for name, values in convergence_data.items():
            ax.plot(values, label=name, linewidth=2)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        ax.set_title('Convergence Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            self.plt.show()
        else:
            self.plt.close()

    def plot_heatmap_comparison(self, spaces: List,
                               positions: List[Tuple[int, int]],
                               save_path: Optional[str] = None,
                               show: bool = True):
        """
        热力图对比不同空间在各位置的距离度量
        """
        self._check_libs()

        n_spaces = len(spaces)
        n_positions = len(positions)

        if n_positions == 0:
            return

        fig, axes = self.plt.subplots(1, n_spaces, figsize=(5*n_spaces, 5))
        if n_spaces == 1:
            axes = [axes]

        for ax, space in zip(axes, spaces):
            # 构建距离矩阵
            dist_matrix = np.zeros((n_positions, n_positions))

            for i, p1 in enumerate(positions):
                for j, p2 in enumerate(positions):
                    if i != j:
                        dist_matrix[i, j] = space.compute_distance(p1, p2)

            im = ax.imshow(dist_matrix, cmap='viridis', aspect='auto')
            ax.set_title(f'{space.name} Distance Matrix')
            self.plt.colorbar(im, ax=ax)

        self.plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            self.plt.show()
        else:
            self.plt.close()

    def create_latex_table(self, results: Dict[str, Dict[str, float]],
                          output_path: str):
        """
        生成LaTeX表格
        """
        if not results:
            return

        configs = list(results.keys())
        metrics = list(results[configs[0]].keys())

        lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\begin{tabular}{l" + "c" * len(metrics) + "}",
            "\\toprule",
        ]

        # 表头
        header = "Config & " + " & ".join(metrics) + " \\\\"
        lines.append(header)
        lines.append("\\midrule")

        # 数据行
        for config in configs:
            values = [f"{results[config].get(m, 0):.3f}" for m in metrics]
            row = f"{config} & " + " & ".join(values) + " \\\\"
            lines.append(row)

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Performance Comparison}",
            "\\end{table}",
        ])

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"LaTeX table saved to {output_path}")

    def plot_space_projection(self, space,
                             method: str = 'pca',
                             save_path: Optional[str] = None,
                             show: bool = True):
        """
        将高维空间投影到2D可视化
        """
        self._check_libs()

        # 采样位置
        w, h = space.width, space.height
        positions = [(x, y) for x in range(0, w, max(1, w//10))
                            for y in range(0, h, max(1, h//10))]

        if len(positions) < 3:
            print("Not enough positions for projection")
            return

        # 构建距离矩阵
        n = len(positions)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = space.compute_distance(positions[i], positions[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # MDS投影
        try:
            from sklearn.manifold import MDS
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            coords = mds.fit_transform(dist_matrix)
        except ImportError:
            # 简单PCA替代
            from numpy.linalg import eigh
            # 双重中心化
            J = np.eye(n) - np.ones((n, n)) / n
            B = -0.5 * J @ (dist_matrix ** 2) @ J
            eigvals, eigvecs = eigh(B)
            idx = np.argsort(eigvals)[::-1][:2]
            coords = eigvecs[:, idx] * np.sqrt(np.maximum(eigvals[idx], 0))

        fig, ax = self.plt.subplots(figsize=(8, 6))
        ax.scatter(coords[:, 0], coords[:, 1], c=range(n), cmap='viridis', s=100)

        # 标注
        for i, pos in enumerate(positions):
            ax.annotate(str(pos), (coords[i, 0], coords[i, 1]),
                       xytext=(3, 3), textcoords='offset points', fontsize=8)

        ax.set_title(f'{space.name} Space Projection (MDS)')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            self.plt.show()
        else:
            self.plt.close()
