"""
ATLAS Spaces: Temporal Space
时序空间

支持时间的认知空间，具有：
- 场的时序演化
- 预测能力
- 周期性模式检测
- 记忆形成
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Callable
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..core.space import CognitiveSpace, register_space


@dataclass
class TemporalSnapshot:
    """时序快照"""
    timestamp: float
    fields: Dict[str, np.ndarray]
    position: Optional[Tuple[int, int]]
    observation: Optional[Dict]


class CircularBuffer:
    """
    循环缓冲区 - 存储历史空间状态
    """

    def __init__(self, maxlen: int = 1000):
        self.maxlen = maxlen
        self.buffer = deque(maxlen=maxlen)
        self.timestamps = deque(maxlen=maxlen)

    def append(self, snapshot: TemporalSnapshot):
        """添加快照"""
        self.buffer.append(snapshot)
        self.timestamps.append(snapshot.timestamp)

    def get_recent(self, n: int) -> List[TemporalSnapshot]:
        """获取最近的n个快照"""
        return list(self.buffer)[-n:]

    def get_time_range(self, start_time: float, end_time: float) -> List[TemporalSnapshot]:
        """获取时间范围内的快照"""
        result = []
        for snapshot in self.buffer:
            if start_time <= snapshot.timestamp <= end_time:
                result.append(snapshot)
        return result

    def get_field_history(self, field_name: str, position: Tuple[int, int]) -> Tuple[List[float], List[float]]:
        """
        获取特定位置某个字段的历史值

        Returns:
            (timestamps, values)
        """
        timestamps = []
        values = []

        for snapshot in self.buffer:
            if field_name in snapshot.fields:
                x, y = position
                field = snapshot.fields[field_name]
                if 0 <= x < field.shape[0] and 0 <= y < field.shape[1]:
                    timestamps.append(snapshot.timestamp)
                    values.append(field[x, y])

        return timestamps, values

    def __len__(self):
        return len(self.buffer)


class FieldPredictor:
    """
    场演化预测器

    使用高斯过程或简单趋势外推预测场的未来状态
    """

    def __init__(self, prediction_horizon: int = 10):
        self.prediction_horizon = prediction_horizon
        self.has_sklearn = False

        # 尝试导入sklearn
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
            self.GaussianProcessRegressor = GaussianProcessRegressor
            self.RBF = RBF
            self.WhiteKernel = WhiteKernel
            self.has_sklearn = True
        except ImportError:
            pass

    def predict_trend(self, timestamps: List[float], values: List[float],
                     future_timestamps: List[float]) -> List[float]:
        """
        简单线性趋势预测（无sklearn时的备选）
        """
        if len(timestamps) < 2:
            # 不足两个点，返回最后值
            return [values[-1] if values else 0.0] * len(future_timestamps)

        # 线性回归
        x = np.array(timestamps)
        y = np.array(values)

        # 最小二乘
        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

        # 预测
        future = np.array(future_timestamps)
        predictions = slope * future + intercept

        # 限制范围 [0, 1] 假设归一化场
        return np.clip(predictions, 0, 1).tolist()

    def predict_gp(self, timestamps: List[float], values: List[float],
                  future_timestamps: List[float]) -> Tuple[List[float], List[float]]:
        """
        高斯过程预测（如果有sklearn）

        Returns:
            (predictions, uncertainties)
        """
        if not self.has_sklearn or len(timestamps) < 3:
            predictions = self.predict_trend(timestamps, values, future_timestamps)
            return predictions, [0.1] * len(future_timestamps)  # 默认不确定性

        # 准备数据
        X = np.array(timestamps).reshape(-1, 1)
        y = np.array(values)
        X_future = np.array(future_timestamps).reshape(-1, 1)

        # 核函数：RBF + 白噪声
        kernel = self.RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0)) + \
                 self.WhiteKernel(noise_level=0.1)

        gp = self.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3)
        gp.fit(X, y)

        y_pred, sigma = gp.predict(X_future, return_std=True)

        return y_pred.tolist(), sigma.tolist()

    def predict_field_evolution(self, history: CircularBuffer,
                               field_name: str,
                               prediction_time: float) -> Optional[np.ndarray]:
        """
        预测整个场的未来状态

        Args:
            history: 历史缓冲区
            field_name: 要预测的字段名
            prediction_time: 预测目标时间

        Returns:
            预测的场数组
        """
        if len(history) == 0:
            return None

        # 获取最新的场作为模板
        latest = list(history.buffer)[-1]
        if field_name not in latest.fields:
            return None

        template = latest.fields[field_name]
        predicted_field = np.zeros_like(template)

        # 对每个位置分别预测
        for x in range(template.shape[0]):
            for y in range(template.shape[1]):
                timestamps, values = history.get_field_history(field_name, (x, y))

                if len(timestamps) >= 2:
                    # 预测
                    predictions, _ = self.predict_gp(timestamps, values, [prediction_time])
                    predicted_field[x, y] = predictions[0]
                else:
                    # 数据不足，用当前值
                    predicted_field[x, y] = template[x, y]

        return predicted_field


class PeriodicityDetector:
    """
    周期性模式检测器

    检测场的周期性变化模式（如障碍物按规律移动）
    """

    def __init__(self, min_period: int = 5, max_period: int = 100):
        self.min_period = min_period
        self.max_period = max_period

    def detect_period_fft(self, signal: List[float]) -> Optional[int]:
        """
        使用FFT检测周期
        """
        if len(signal) < self.max_period * 2:
            return None

        # 去趋势
        signal = np.array(signal)
        signal = signal - np.mean(signal)

        # FFT
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal))

        # 找主频（排除直流分量）
        magnitude = np.abs(fft)
        magnitude[0] = 0  # 排除直流

        # 限制频率范围
        valid_idx = np.where((freqs > 1/self.max_period) & (freqs < 1/self.min_period))[0]

        if len(valid_idx) == 0:
            return None

        peak_idx = valid_idx[np.argmax(magnitude[valid_idx])]
        period = int(1 / abs(freqs[peak_idx]))

        return period

    def detect_period_autocorr(self, signal: List[float]) -> Optional[int]:
        """
        使用自相关检测周期
        """
        if len(signal) < self.max_period * 2:
            return None

        signal = np.array(signal)
        signal = signal - np.mean(signal)

        # 自相关
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # 归一化

        # 找第一个显著峰值（排除0滞后）
        for lag in range(self.min_period, min(self.max_period, len(autocorr))):
            if autocorr[lag] > 0.5:  # 阈值
                # 确认是峰值
                if autocorr[lag] > autocorr[lag-1] and autocorr[lag] > autocorr[lag+1]:
                    return lag

        return None

    def analyze_field_periodicity(self, history: CircularBuffer,
                                 field_name: str,
                                 sample_positions: List[Tuple[int, int]]) -> Dict:
        """
        分析场的周期性

        Returns:
            {
                'has_periodicity': bool,
                'period': int or None,
                'confidence': float,
                'position_periods': {pos: period}
            }
        """
        periods = []

        for pos in sample_positions:
            timestamps, values = history.get_field_history(field_name, pos)
            if len(values) >= self.max_period * 2:
                period = self.detect_period_autocorr(values)
                if period:
                    periods.append(period)

        if not periods:
            return {
                'has_periodicity': False,
                'period': None,
                'confidence': 0.0,
                'position_periods': {}
            }

        # 统计最常见的周期
        from collections import Counter
        period_counts = Counter(periods)
        most_common = period_counts.most_common(1)[0]

        confidence = most_common[1] / len(periods)

        return {
            'has_periodicity': confidence > 0.5,
            'period': most_common[0],
            'confidence': confidence,
            'position_periods': dict(periods)
        }


@register_space("temporal")
class TemporalSpace(CognitiveSpace):
    """
    时序认知空间

    具有时间感知能力的认知空间，支持：
    - 历史状态记录
    - 场演化预测
    - 周期性模式检测
    - 长期记忆形成

    Example:
        space = TemporalSpace(
            width=40, height=20,
            history_length=100,
            prediction_horizon=10
        )

        # 随着时间更新
        for t in range(100):
            space.update_from_observation(
                position,
                observation,
                timestamp=t
            )

        # 预测未来状态
        future_field = space.predict_field('uncertainty', t+10)
    """

    def __init__(self, width: int, height: int,
                 base_space_type: str = "ricci",
                 base_space_params: Optional[Dict] = None,
                 history_length: int = 100,
                 prediction_horizon: int = 10,
                 enable_periodicity_detection: bool = True,
                 **kwargs):
        super().__init__(width, height, name="temporal")

        # 创建基础空间
        from ..core.registry import create_space
        base_params = base_space_params or {}
        self.base_space = create_space(base_space_type, width, height, **base_params)

        # 时序组件
        self.history = CircularBuffer(maxlen=history_length)
        self.predictor = FieldPredictor(prediction_horizon=prediction_horizon)
        self.periodicity_detector = PeriodicityDetector() if enable_periodicity_detection else None

        # 配置
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.current_time = 0.0

        # 周期性发现
        self.detected_periods = {}

        # 预测缓存
        self._prediction_cache = {}

    def compute_distance(self, pos1: Tuple[int, int],
                        pos2: Tuple[int, int]) -> float:
        """委托给基础空间"""
        return self.base_space.compute_distance(pos1, pos2)

    def get_heuristic(self, pos: Tuple[int, int],
                     goal: Tuple[int, int]) -> float:
        """委托给基础空间"""
        return self.base_space.get_heuristic(pos, goal)

    def update_from_observation(self, position: Tuple[int, int],
                                observation: Dict[str, Any],
                                timestamp: Optional[float] = None) -> None:
        """
        根据观测更新空间（时序版本）

        Args:
            position: 观测位置
            observation: 观测数据，可以包含:
                - 'timestamp': 时间戳（覆盖参数）
                - 其他标准观测字段
            timestamp: 时间戳（可选，优先使用observation中的）
        """
        # 更新时间
        if 'timestamp' in observation:
            self.current_time = observation['timestamp']
        elif timestamp is not None:
            self.current_time = timestamp
        else:
            self.current_time += 1.0

        # 更新基础空间
        self.base_space.update_from_observation(position, observation)

        # 记录快照
        snapshot = TemporalSnapshot(
            timestamp=self.current_time,
            fields=self.base_space.get_visualization_fields(),
            position=position,
            observation=observation
        )
        self.history.append(snapshot)

        # 清除过期的预测缓存
        self._prediction_cache = {
            k: v for k, v in self._prediction_cache.items()
            if k > self.current_time
        }

        # 周期性检测（定期执行）
        if self.periodicity_detector and len(self.history) % 20 == 0:
            self._update_periodicity_analysis()

    def _update_periodicity_analysis(self):
        """更新周期性分析"""
        if len(self.history) < 50:
            return

        # 采样位置
        sample_positions = [
            (self.width // 4, self.height // 4),
            (self.width // 2, self.height // 2),
            (3 * self.width // 4, 3 * self.height // 4),
        ]

        for field_name in self.base_space.get_visualization_fields().keys():
            result = self.periodicity_detector.analyze_field_periodicity(
                self.history, field_name, sample_positions
            )
            if result['has_periodicity']:
                self.detected_periods[field_name] = result

    def predict_field(self, field_name: str,
                     future_time: float) -> Optional[np.ndarray]:
        """
        预测未来时刻的场状态

        Args:
            field_name: 字段名
            future_time: 未来时间戳

        Returns:
            预测的场数组
        """
        cache_key = (field_name, future_time)
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]

        prediction = self.predictor.predict_field_evolution(
            self.history, field_name, future_time
        )

        if prediction is not None:
            self._prediction_cache[cache_key] = prediction

        return prediction

    def get_field_history(self, field_name: str,
                         position: Tuple[int, int]) -> Tuple[List[float], List[float]]:
        """
        获取特定位置的历史数据

        Returns:
            (timestamps, values)
        """
        return self.history.get_field_history(field_name, position)

    def get_temporal_statistics(self) -> Dict[str, Any]:
        """获取时序统计信息"""
        stats = {
            'history_length': len(self.history),
            'current_time': self.current_time,
            'detected_periods': self.detected_periods,
        }

        # 变化率统计
        if len(self.history) >= 2:
            recent = list(self.history.buffer)[-10:]
            changes = []
            for i in range(1, len(recent)):
                for field_name in recent[i].fields:
                    if field_name in recent[i-1].fields:
                        diff = np.abs(
                            recent[i].fields[field_name] - recent[i-1].fields[field_name]
                        ).mean()
                        changes.append(diff)

            if changes:
                stats['average_field_change_rate'] = np.mean(changes)

        return stats

    def predict_obstacle_trajectory(self, obstacle_id: str,
                                    future_times: List[float]) -> List[Optional[Tuple[int, int]]]:
        """
        预测障碍物的未来轨迹

        基于历史观测预测动态障碍物的位置
        """
        # 从历史中提取该障碍物的轨迹
        trajectory = []
        for snapshot in self.history.buffer:
            if snapshot.observation and 'dynamic_obstacles' in snapshot.observation:
                obstacles = snapshot.observation['dynamic_obstacles']
                if obstacle_id in obstacles:
                    trajectory.append((snapshot.timestamp, obstacles[obstacle_id]))

        if len(trajectory) < 2:
            return [None] * len(future_times)

        # 分离时间戳和位置
        timestamps = [t for t, _ in trajectory]
        xs = [pos[0] for _, pos in trajectory]
        ys = [pos[1] for _, pos in trajectory]

        # 分别预测x和y
        pred_x, _ = self.predictor.predict_gp(timestamps, xs, future_times)
        pred_y, _ = self.predictor.predict_gp(timestamps, ys, future_times)

        return [(int(x), int(y)) for x, y in zip(pred_x, pred_y)]

    def get_visualization_fields(self) -> Dict[str, np.ndarray]:
        """返回基础空间的场 + 预测误差场"""
        fields = self.base_space.get_visualization_fields()

        # 如果有预测，添加预测误差
        if len(self.history) >= 2:
            last_time = list(self.history.timestamps)[-1]
            for field_name in list(fields.keys()):
                pred = self.predict_field(field_name, last_time)
                if pred is not None:
                    actual = fields[field_name]
                    error = np.abs(pred - actual)
                    fields[f'{field_name}_prediction_error'] = error

        return fields

    def get_statistics(self) -> Dict[str, Any]:
        """合并基础统计和时序统计"""
        stats = self.base_space.get_statistics()
        stats.update(self.get_temporal_statistics())
        return stats


@register_space("predictive_ricci")
class PredictiveRicciSpace(TemporalSpace):
    """
    预测性Ricci空间

    专门用于动态环境的TemporalSpace，预测曲率演化
    """

    def __init__(self, width: int, height: int,
                 curvature_decay: float = 0.95,
                 **kwargs):
        super().__init__(
            width, height,
            base_space_type="ricci",
            **kwargs
        )
        self.name = "predictive_ricci"
        self.curvature_decay = curvature_decay

    def predict_future_curvature(self, position: Tuple[int, int],
                                 steps_ahead: int = 5) -> float:
        """
        预测未来特定位置的曲率
        """
        timestamps, values = self.get_field_history('curvature', position)

        if len(timestamps) < 2:
            # 返回当前值
            fields = self.base_space.get_visualization_fields()
            if 'curvature' in fields:
                x, y = position
                return fields['curvature'][x, y]
            return 0.0

        # 预测
        future_time = self.current_time + steps_ahead
        predictions, uncertainties = self.predictor.predict_gp(
            timestamps, values, [future_time]
        )

        return predictions[0] if predictions else 0.0

    def compute_distance_with_prediction(self, pos1: Tuple[int, int],
                                        pos2: Tuple[int, int],
                                        steps_ahead: int = 0) -> float:
        """
        计算距离（可选使用预测的未来状态）
        """
        if steps_ahead == 0:
            return self.compute_distance(pos1, pos2)

        # 使用预测的曲率计算距离
        # 这里简化处理，实际应该基于预测场重新计算
        base_dist = self.compute_distance(pos1, pos2)

        # 添加基于预测的调整
        future_curvature = self.predict_future_curvature(
            ((pos1[0] + pos2[0]) // 2, (pos1[1] + pos2[1]) // 2),
            steps_ahead
        )

        # 曲率越高，距离感越长
        adjustment = 1.0 + 0.1 * abs(future_curvature)
        return base_dist * adjustment
