"""
ATLAS Visualization: Path Animator
路径动画

展示路径规划过程和演变
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from collections import deque


class PathAnimator:
    """
    路径动画生成器

    支持：
    - 路径演化动画
    - 多智能体路径对比
    - 空间场随时间变化
    """

    def __init__(self, style: str = 'default'):
        self.style = style
        self._has_matplotlib = False

        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation, PillowWriter
            import matplotlib.patches as patches
            self._has_matplotlib = True
            self.plt = plt
            self.FuncAnimation = FuncAnimation
            self.PillowWriter = PillowWriter
            self.patches = patches
        except ImportError:
            self.plt = None

    def _check_matplotlib(self):
        if not self._has_matplotlib:
            raise ImportError("matplotlib is required for animation")

    def animate_path_planning(self, space,
                             planning_process: List[Dict],
                             save_path: Optional[str] = None,
                             interval: int = 200):
        """
        动画展示路径规划过程

        Args:
            space: 认知空间
            planning_process: [
                {'expanded': [(x,y),...], 'frontier': [(x,y),...], 'current': (x,y)},
                ...
            ]
            save_path: 保存路径 (.gif)
            interval: 帧间隔（毫秒）
        """
        self._check_matplotlib()

        fig, ax = self.plt.subplots(figsize=(8, 6))

        # 背景场
        fields = space.get_visualization_fields()
        bg_field = list(fields.values())[0] if fields else np.ones((space.width, space.height))

        im = ax.imshow(bg_field.T, origin='lower', cmap='viridis', alpha=0.5)

        # 绘制元素
        expanded_scatter = ax.scatter([], [], c='blue', s=20, alpha=0.3, label='expanded')
        frontier_scatter = ax.scatter([], [], c='orange', s=30, alpha=0.6, label='frontier')
        current_point = ax.scatter([], [], c='red', s=100, marker='*', label='current')

        ax.set_xlim(0, space.width)
        ax.set_ylim(0, space.height)
        ax.set_title('Path Planning Process')
        ax.legend(loc='upper right')

        def update(frame_data):
            expanded = frame_data.get('expanded', [])
            frontier = frame_data.get('frontier', [])
            current = frame_data.get('current')

            if expanded:
                expanded_scatter.set_offsets(expanded)
            if frontier:
                frontier_scatter.set_offsets(frontier)
            if current:
                current_point.set_offsets([current])

            return expanded_scatter, frontier_scatter, current_point

        anim = self.FuncAnimation(fig, update, frames=planning_process,
                                 interval=interval, blit=True, repeat=True)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            writer = self.PillowWriter(fps=1000//interval)
            anim.save(save_path, writer=writer)
            print(f"Animation saved to {save_path}")

        self.plt.show()
        return anim

    def animate_navigation(self, space,
                          trajectory: List[Tuple[int, int]],
                          space_history: Optional[List] = None,
                          obstacles_history: Optional[List] = None,
                          save_path: Optional[str] = None,
                          interval: int = 300):
        """
        动画展示导航过程

        Args:
            space: 认知空间
            trajectory: 实际轨迹 [(x,y), ...]
            space_history: 空间状态历史（可选）
            obstacles_history: 障碍物变化历史（可选）
            save_path: 保存路径
            interval: 帧间隔
        """
        self._check_matplotlib()

        fig, ax = self.plt.subplots(figsize=(8, 6))

        # 获取背景场
        fields = space.get_visualization_fields()
        bg_field = list(fields.values())[0] if fields else np.ones((space.width, space.height))

        im = ax.imshow(bg_field.T, origin='lower', cmap='viridis', alpha=0.6)

        # 轨迹线
        line, = ax.plot([], [], 'w-', linewidth=2, alpha=0.8, label='path')
        position = ax.scatter([], [], c='red', s=100, marker='o', zorder=5)
        trail = ax.scatter([], [], c='yellow', s=30, alpha=0.5)

        ax.set_xlim(0, space.width)
        ax.set_ylim(0, space.height)
        ax.set_title('Navigation Animation')

        # 障碍物绘制
        obstacle_patches = []

        def update(frame_idx):
            # 更新位置
            x, y = trajectory[frame_idx]
            position.set_offsets([[x, y]])

            # 更新轨迹线
            xs = [p[0] for p in trajectory[:frame_idx+1]]
            ys = [p[1] for p in trajectory[:frame_idx+1]]
            line.set_data(xs, ys)

            # 更新轨迹点
            if frame_idx > 0:
                trail.set_offsets(trajectory[:frame_idx])

            # 更新障碍物
            if obstacles_history and frame_idx < len(obstacles_history):
                for patch in obstacle_patches:
                    patch.remove()
                obstacle_patches.clear()

                for (ox, oy) in obstacles_history[frame_idx]:
                    rect = self.patches.Rectangle((ox-0.4, oy-0.4), 0.8, 0.8,
                                                  linewidth=1, edgecolor='r',
                                                  facecolor='red', alpha=0.7)
                    ax.add_patch(rect)
                    obstacle_patches.append(rect)

            return line, position, trail

        frames = len(trajectory)
        anim = self.FuncAnimation(fig, update, frames=frames,
                                 interval=interval, blit=False, repeat=True)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            writer = self.PillowWriter(fps=1000//interval)
            anim.save(save_path, writer=writer)
            print(f"Navigation animation saved to {save_path}")

        self.plt.show()
        return anim

    def compare_path_evolution(self, comparisons: List[Dict],
                              save_path: Optional[str] = None):
        """
        对比不同算法/空间的路径演变

        Args:
            comparisons: [
                {
                    'name': 'Algorithm A',
                    'trajectory': [(x,y),...],
                    'fields': field_data,
                    'color': 'red'
                },
                ...
            ]
        """
        self._check_matplotlib()

        n = len(comparisons)
        fig, axes = self.plt.subplots(1, n, figsize=(6*n, 5))
        if n == 1:
            axes = [axes]

        for ax, comp in zip(axes, comparisons):
            traj = comp['trajectory']
            color = comp.get('color', 'blue')
            name = comp['name']

            # 绘制背景场
            if 'fields' in comp:
                bg = list(comp['fields'].values())[0]
                ax.imshow(bg.T, origin='lower', cmap='viridis', alpha=0.5)

            # 绘制路径
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            ax.plot(xs, ys, color=color, linewidth=2, label=name)
            ax.scatter([xs[0]], [ys[0]], c='green', s=100, marker='o', label='start')
            ax.scatter([xs[-1]], [ys[-1]], c='red', s=100, marker='*', label='goal')

            ax.set_title(name)
            ax.legend()

        self.plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')

        self.plt.show()

    def create_path_smoothness_analysis(self, path: List[Tuple[int, int]],
                                       space,
                                       save_path: Optional[str] = None):
        """
        分析路径平滑度
        """
        self._check_matplotlib()

        if len(path) < 3:
            print("Path too short for analysis")
            return

        # 计算曲率
        curvatures = []
        for i in range(1, len(path) - 1):
            p0 = np.array(path[i-1])
            p1 = np.array(path[i])
            p2 = np.array(path[i+1])

            # 转向角
            v1 = p0 - p1
            v2 = p2 - p1

            angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            curvatures.append(abs(np.sin(angle)))

        fig, (ax1, ax2) = self.plt.subplots(1, 2, figsize=(12, 4))

        # 路径图
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax1.plot(xs, ys, 'b-', linewidth=2)
        ax1.scatter(xs, ys, c=range(len(xs)), cmap='viridis', s=20)
        ax1.set_title('Path with Step Index')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        # 曲率图
        ax2.plot(range(1, len(path)-1), curvatures, 'r-', linewidth=2)
        ax2.set_title('Path Curvature')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Curvature')
        ax2.axhline(y=np.mean(curvatures), color='g', linestyle='--',
                   label=f'Mean: {np.mean(curvatures):.3f}')
        ax2.legend()

        self.plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')

        self.plt.show()

        return {
            'mean_curvature': np.mean(curvatures),
            'max_curvature': np.max(curvatures),
            'total_turning': np.sum(curvatures)
        }
