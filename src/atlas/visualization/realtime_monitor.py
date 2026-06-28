"""
ATLAS Visualization: Real-time Monitor
实时监控器

用于实时监控空间状态和导航过程
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from collections import deque


class RealtimeMonitor:
    """
    实时空间监控器

    可以：
    - 实时绘制空间场变化
    - 记录导航指标
    - 检测异常（如卡死、循环）
    """

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.space_history = deque(maxlen=max_history)
        self.metric_history = deque(maxlen=max_history)
        self.event_log = deque(maxlen=max_history)

        # 实时显示
        self._has_display = False
        self.fig = None
        self.axes = None

        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            self._has_display = True
        except ImportError:
            self.plt = None

    def log_space_state(self, space, timestamp: Optional[float] = None,
                       info: Optional[Dict] = None):
        """
        记录空间状态

        Args:
            space: 认知空间
            timestamp: 时间戳（默认当前时间）
            info: 附加信息
        """
        if timestamp is None:
            timestamp = time.time()

        state = {
            'timestamp': timestamp,
            'fields': space.get_visualization_fields(),
            'statistics': space.get_statistics(),
            'info': info or {}
        }

        self.space_history.append(state)

    def log_metrics(self, metrics: Dict[str, float],
                   timestamp: Optional[float] = None):
        """
        记录性能指标
        """
        if timestamp is None:
            timestamp = time.time()

        self.metric_history.append({
            'timestamp': timestamp,
            **metrics
        })

    def log_event(self, event_type: str, message: str,
                 data: Optional[Dict] = None):
        """
        记录事件
        """
        self.event_log.append({
            'timestamp': time.time(),
            'type': event_type,
            'message': message,
            'data': data or {}
        })

    def detect_loops(self, recent_positions: List[Tuple[int, int]],
                    window: int = 20) -> bool:
        """
        检测是否陷入循环

        Returns:
            True if loop detected
        """
        if len(recent_positions) < window * 2:
            return False

        recent = set(recent_positions[-window:])
        earlier = set(recent_positions[-window*2:-window])

        # 如果最近位置大量重复之前的位置
        overlap = len(recent & earlier)
        return overlap > window * 0.5

    def analyze_convergence(self, metric_name: str,
                           window: int = 50) -> Dict[str, float]:
        """
        分析指标收敛性
        """
        if len(self.metric_history) < window:
            return {'status': 'insufficient_data'}

        recent = list(self.metric_history)[-window:]
        values = [m.get(metric_name, 0) for m in recent]

        # 计算变化趋势
        first_half = np.mean(values[:window//2])
        second_half = np.mean(values[window//2:])
        trend = second_half - first_half

        # 方差
        variance = np.var(values)

        return {
            'status': 'converged' if abs(trend) < 0.01 * abs(first_half + 1e-6) else 'diverging',
            'trend': trend,
            'variance': variance,
            'mean': np.mean(values),
            'current': values[-1] if values else 0
        }

    def plot_metrics(self, metric_names: Optional[List[str]] = None,
                    save_path: Optional[str] = None,
                    show: bool = True):
        """
        绘制指标历史
        """
        if not self.plt:
            raise ImportError("matplotlib required")

        if len(self.metric_history) == 0:
            print("No metrics to plot")
            return

        # 获取所有指标名称
        if metric_names is None:
            sample = self.metric_history[0]
            metric_names = [k for k in sample.keys() if k != 'timestamp']

        n_metrics = len(metric_names)
        if n_metrics == 0:
            return

        fig, axes = self.plt.subplots(n_metrics, 1, figsize=(10, 3*n_metrics))
        if n_metrics == 1:
            axes = [axes]

        timestamps = [m['timestamp'] - self.metric_history[0]['timestamp']
                     for m in self.metric_history]

        for ax, name in zip(axes, metric_names):
            values = [m.get(name, 0) for m in self.metric_history]
            ax.plot(timestamps, values, linewidth=2)
            ax.set_ylabel(name)
            ax.set_xlabel('Time (s)')
            ax.grid(True, alpha=0.3)

        self.plt.tight_layout()

        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            self.plt.show()
        else:
            self.plt.close()

    def plot_field_evolution(self, field_name: str,
                            positions: Optional[List[Tuple[int, int]]] = None,
                            save_path: Optional[str] = None,
                            show: bool = True):
        """
        绘制特定场位置的时间演变
        """
        if not self.plt:
            raise ImportError("matplotlib required")

        if len(self.space_history) == 0:
            print("No space history to plot")
            return

        # 如果没有指定位置，采样几个代表性位置
        if positions is None:
            # 使用场中心附近
            sample_fields = self.space_history[0]['fields']
            if field_name in sample_fields:
                w, h = sample_fields[field_name].shape
                positions = [
                    (w//4, h//4),
                    (w//2, h//2),
                    (3*w//4, 3*h//4),
                ]
            else:
                print(f"Field {field_name} not found")
                return

        fig, ax = self.plt.subplots(figsize=(10, 5))

        timestamps = [s['timestamp'] - self.space_history[0]['timestamp']
                     for s in self.space_history]

        for pos in positions:
            values = []
            for state in self.space_history:
                fields = state['fields']
                if field_name in fields:
                    x, y = pos
                    field = fields[field_name]
                    if 0 <= x < field.shape[0] and 0 <= y < field.shape[1]:
                        values.append(field[x, y])
                    else:
                        values.append(np.nan)
                else:
                    values.append(np.nan)

            ax.plot(timestamps, values, label=f'Pos {pos}', linewidth=2)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel(field_name)
        ax.set_title(f'{field_name} Evolution at Different Positions')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            self.plt.show()
        else:
            self.plt.close()

    def generate_report(self, output_path: str):
        """
        生成监控报告
        """
        from pathlib import Path

        report_lines = [
            "# ATLAS Navigation Report",
            "",
            f"## Summary",
            f"- Total events: {len(self.event_log)}",
            f"- Space snapshots: {len(self.space_history)}",
            f"- Metric records: {len(self.metric_history)}",
            "",
            "## Events",
        ]

        for event in list(self.event_log)[-50:]:  # 最近50个事件
            time_str = time.strftime('%H:%M:%S', time.localtime(event['timestamp']))
            report_lines.append(f"- [{time_str}] {event['type']}: {event['message']}")

        report_lines.extend([
            "",
            "## Statistics",
        ])

        if self.metric_history:
            last_metrics = self.metric_history[-1]
            for key, value in last_metrics.items():
                if key != 'timestamp':
                    report_lines.append(f"- {key}: {value}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"Report saved to {output_path}")
