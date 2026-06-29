"""
ATLAS Visualization: Space Visualizer
空间场可视化
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


class SpaceVisualizer:
    """
    认知空间可视化器

    可视化空间的内部场：曲率、uncertainty、置信度等
    """

    def __init__(self, style: str = 'default'):
        self.style = style
        self._has_matplotlib = False

        # 延迟导入
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
            self._has_matplotlib = True
            self.plt = plt
            self.colors = LinearSegmentedColormap
        except ImportError:
            self.plt = None

    def _check_matplotlib(self):
        if not self._has_matplotlib:
            raise ImportError("matplotlib is required for visualization. "
                            "Install with: pip install matplotlib")

    def visualize_space(self, space,
                       fields: Optional[List[str]] = None,
                       overlay: Optional[Dict] = None,
                       save_path: Optional[str] = None,
                       show: bool = True):
        """
        可视化空间场

        Args:
            space: CognitiveSpace 实例
            fields: 要可视化的字段列表，None表示全部
            overlay: 叠加层数据 {'path': [...], 'obstacles': [...], 'goal': (x,y)}
            save_path: 保存路径
            show: 是否显示
        """
        self._check_matplotlib()

        # 获取可视化字段
        viz_fields = space.get_visualization_fields()

        if fields is None:
            fields = list(viz_fields.keys())

        n_fields = len(fields)
        if n_fields == 0:
            return

        # 创建子图
        fig, axes = self.plt.subplots(1, n_fields, figsize=(5*n_fields, 4))
        if n_fields == 1:
            axes = [axes]

        for ax, field_name in zip(axes, fields):
            if field_name not in viz_fields:
                continue

            data = viz_fields[field_name]

            # 绘制热力图
            im = ax.imshow(data.T, origin='lower', cmap='viridis', aspect='auto')
            ax.set_title(f'{space.name}.{field_name}')
            self.plt.colorbar(im, ax=ax)

            # 叠加层
            if overlay:
                if 'obstacles' in overlay:
                    for (x, y) in overlay['obstacles']:
                        if 0 <= x < data.shape[0] and 0 <= y < data.shape[1]:
                            ax.plot(x, y, 'rx', markersize=8)

                if 'path' in overlay:
                    path = overlay['path']
                    if path:
                        xs = [p[0] for p in path]
                        ys = [p[1] for p in path]
                        ax.plot(xs, ys, 'w-', linewidth=2, alpha=0.8)
                        ax.plot(xs[0], ys[0], 'go', markersize=10, label='start')
                        ax.plot(xs[-1], ys[-1], 'ro', markersize=10, label='goal')

        self.plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            self.plt.show()
        else:
            self.plt.close()

    def visualize_metrics(self, space,
                         save_path: Optional[str] = None,
                         show: bool = True):
        """
        可视化空间统计指标
        """
        self._check_matplotlib()

        stats = space.get_statistics()

        fig, (ax1, ax2) = self.plt.subplots(1, 2, figsize=(12, 4))

        # 数值指标
        numeric_stats = {k: v for k, v in stats.items()
                        if isinstance(v, (int, float)) and k != 'name'}

        if numeric_stats:
            ax1.bar(range(len(numeric_stats)), list(numeric_stats.values()))
            ax1.set_xticks(range(len(numeric_stats)))
            ax1.set_xticklabels(list(numeric_stats.keys()), rotation=45, ha='right')
            ax1.set_title(f'{space.name} Statistics')
            ax1.set_ylabel('Value')

        # 字段分布直方图
        fields = space.get_visualization_fields()
        if fields:
            field_name = list(fields.keys())[0]
            data = fields[field_name].flatten()
            data = data[data > 0]  # 只显示非零值
            ax2.hist(data, bins=50, alpha=0.7)
            ax2.set_title(f'{field_name} Distribution')
            ax2.set_xlabel('Value')
            ax2.set_ylabel('Count')

        self.plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            self.plt.show()
        else:
            self.plt.close()

    def compare_spaces(self, spaces: List,
                      field: str = 'metric',
                      titles: Optional[List[str]] = None,
                      save_path: Optional[str] = None,
                      show: bool = True):
        """
        并排对比多个空间
        """
        self._check_matplotlib()

        n_spaces = len(spaces)
        if n_spaces == 0:
            return

        fig, axes = self.plt.subplots(1, n_spaces, figsize=(5*n_spaces, 4))
        if n_spaces == 1:
            axes = [axes]

        for i, (space, ax) in enumerate(zip(spaces, axes)):
            fields = space.get_visualization_fields()

            if field in fields:
                data = fields[field]
            elif len(fields) > 0:
                data = list(fields.values())[0]
            else:
                data = np.ones((space.width, space.height))

            im = ax.imshow(data.T, origin='lower', cmap='viridis', aspect='auto')
            ax.set_title(titles[i] if titles and i < len(titles) else space.name)
            self.plt.colorbar(im, ax=ax)

        self.plt.suptitle(f'Space Comparison: {field}', fontsize=14)
        self.plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            self.plt.show()
        else:
            self.plt.close()

    def create_animation_frames(self, space_history: List[Tuple[Any, Dict]],
                               output_dir: str,
                               field: str = 'metric'):
        """
        创建动画帧序列

        Args:
            space_history: [(space, info), ...] 空间历史
            output_dir: 输出目录
            field: 要可视化的字段
        """
        self._check_matplotlib()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i, (space, info) in enumerate(space_history):
            save_path = output_path / f'frame_{i:04d}.png'
            overlay = info.get('overlay', {})

            self.visualize_space(
                space,
                fields=[field],
                overlay=overlay,
                save_path=str(save_path),
                show=False
            )

        print(f"Generated {len(space_history)} frames in {output_dir}")
