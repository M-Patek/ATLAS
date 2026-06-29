"""
演示: TemporalSpace 时序空间

展示时间维度、场演化预测、周期性检�?"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from src.core import GeodesicSolver
from src.core.registry import create_space
from src.spaces.temporal import TemporalSpace, CircularBuffer, FieldPredictor, PeriodicityDetector

print("=" * 70)
print("ATLAS: TemporalSpace Demo")
print("=" * 70)
print()

# ============================================================================
# 1. Circular Buffer Demo
# ============================================================================

print("=" * 70)
print("1. Circular Buffer: History Management")
print("=" * 70)
print()

buffer = CircularBuffer(maxlen=5)

# 添加数据
for i in range(10):
    from src.spaces.temporal import TemporalSnapshot
    snapshot = TemporalSnapshot(
        timestamp=float(i),
        fields={'field1': np.random.rand(10, 10)},
        position=(i, i),
        observation={}
    )
    buffer.append(snapshot)

print(f"Added 10 snapshots, buffer size: {len(buffer)} (max=5)")

# 获取最近的
recent = buffer.get_recent(3)
print(f"Recent 3 timestamps: {[s.timestamp for s in recent]}")

# 获取历史
position = (5, 5)
for i in range(10):
    snapshot = TemporalSnapshot(
        timestamp=float(i),
        fields={'test_field': np.full((10, 10), i * 0.1)},
        position=position,
        observation={}
    )
    buffer.append(snapshot)

timestamps, values = buffer.get_field_history('test_field', position)
print(f"Field history at {position}:")
print(f"  Timestamps: {timestamps}")
print(f"  Values: {values}")
print()

# ============================================================================
# 2. Field Prediction Demo
# ============================================================================

print("=" * 70)
print("2. Field Predictor: Evolution Prediction")
print("=" * 70)
print()

predictor = FieldPredictor(prediction_horizon=5)

# 生成趋势数据
original_timestamps = list(range(20))
original_values = [0.1 * t + np.random.normal(0, 0.1) for t in original_timestamps]

# 预测未来
future_timestamps = [20, 21, 22, 23, 24]
predictions, uncertainties = predictor.predict_gp(
    original_timestamps, original_values, future_timestamps
)

print("Linear trend with noise:")
print(f"  Historical: t={original_timestamps[-3:]}, v={[round(v, 2) for v in original_values[-3:]]}")
print(f"  Predicted:  t={future_timestamps}, v={[round(v, 2) for v in predictions]}")
print(f"  Uncertainty: {[round(u, 3) for u in uncertainties]}")
print()

# ============================================================================
# 3. Periodicity Detection Demo
# ============================================================================

print("=" * 70)
print("3. Periodicity Detector: Pattern Recognition")
print("=" * 70)
print()

detector = PeriodicityDetector(min_period=3, max_period=20)

# 生成周期信号
period = 7
periodic_signal = [np.sin(2 * np.pi * t / period) + np.random.normal(0, 0.1)
                   for t in range(50)]

detected_period = detector.detect_period_autocorr(periodic_signal)
print(f"Generated signal with period {period}")
print(f"Detected period: {detected_period}")

# 非周期信�?random_signal = np.random.randn(50).tolist()
detected_random = detector.detect_period_autocorr(random_signal)
print(f"Random signal detected period: {detected_random}")
print()

# ============================================================================
# 4. TemporalSpace: Dynamic Environment Simulation
# ============================================================================

print("=" * 70)
print("4. TemporalSpace: Dynamic Environment Test")
print("=" * 70)
print()

width, height = 30, 20

# 创建时序空间
space = TemporalSpace(
    width=width, height=height,
    base_space_type="ricci",
    base_space_params={"curvature_scale": 1.5},
    history_length=50,
    prediction_horizon=5,
    enable_periodicity_detection=True
)

print(f"Created TemporalSpace {width}x{height} with Ricci base")
print(f"History buffer size: {space.history_length}")
print()

# 模拟动态环�?start_pos = (5, 10)
goal_pos = (25, 10)

print("Simulating environment over 30 time steps...")
print()

# 模拟移动障碍�?obstacle_positions = []
for t in range(30):
    # 障碍物做正弦移动
    obs_x = 15 + int(5 * np.sin(2 * np.pi * t / 10))
    obs_y = 10 + int(3 * np.cos(2 * np.pi * t / 10))
    obstacle_positions.append((obs_x, obs_y))

# 更新空间
for t in range(30):
    obs_pos = obstacle_positions[t]

    # 构造观�?    observation = {
        'timestamp': float(t),
        'obstacles': [obs_pos],
        'goal_position': goal_pos,
        'dynamic_obstacles': {'obs1': obs_pos}
    }

    space.update_from_observation(start_pos, observation)

    if t % 10 == 0:
        print(f"  t={t}: Obstacle at {obs_pos}, History size: {len(space.history)}")

print()

# 分析历史
temporal_stats = space.get_temporal_statistics()
print("Temporal Statistics:")
print(f"  History length: {temporal_stats['history_length']}")
print(f"  Current time: {temporal_stats['current_time']}")
print(f"  Average field change rate: {temporal_stats.get('average_field_change_rate', 'N/A')}")
print()

# 预测障碍物轨�?future_times = [30, 32, 34, 36, 38]
predicted_trajectory = space.predict_obstacle_trajectory('obs1', future_times)
print("Predicted obstacle trajectory:")
actual_future = [obstacle_positions[min(t, 29)] for t in future_times]
for t, pred, actual in zip(future_times, predicted_trajectory, actual_future):
    pred_str = str(pred) if pred else "N/A"
    print(f"  t={t}: predicted={pred_str}, actual={actual}")
print()

# ============================================================================
# 5. Field Prediction
# ============================================================================

print("=" * 70)
print("5. Field Evolution Prediction")
print("=" * 70)
print()

# 预测未来场状�?future_time = 35
predicted_uncertainty = space.predict_field('uncertainty', future_time)

if predicted_uncertainty is not None:
    print(f"Predicted uncertainty field at t={future_time}:")
    print(f"  Shape: {predicted_uncertainty.shape}")
    print(f"  Mean: {predicted_uncertainty.mean():.3f}")
    print(f"  Std: {predicted_uncertainty.std():.3f}")
else:
    print("Prediction not available")

# 获取历史对比
position = (15, 10)
timestamps, values = space.get_field_history('uncertainty', position)
print(f"\nUncertainty history at {position}:")
print(f"  Timestamps (last 5): {timestamps[-5:]}")
print(f"  Values (last 5): {[round(v, 3) for v in values[-5:]]}")
print()

# ============================================================================
# 6. Planning with Temporal Awareness
# ============================================================================

print("=" * 70)
print("6. Path Planning in Temporal Space")
print("=" * 70)
print()

# 在当前状态规�?solver = GeodesicSolver(space)

# 添加静态障碍物
static_obstacles = set()
for t_obs in obstacle_positions[-5:]:  # 最近的障碍物位�?    static_obstacles.add(t_obs)

result = solver.solve(start_pos, goal_pos, static_obstacles)

print(f"Planning with temporal awareness:")
print(f"  Success: {result.success}")
print(f"  Steps: {len(result.path) if result.path else 0}")
print(f"  Cost: {result.cost:.2f}")
print(f"  Time: {result.time_ms:.2f}ms")

if result.path:
    print(f"  Path: {result.path[:3]}...{result.path[-3:]}")
print()

# ============================================================================
# 7. Predictive Ricci Space
# ============================================================================

print("=" * 70)
print("7. PredictiveRicciSpace")
print("=" * 70)
print()

from src.spaces.temporal import PredictiveRicciSpace

pred_space = PredictiveRicciSpace(
    width=width, height=height,
    curvature_decay=0.95,
    history_length=30
)

# 同样的更�?for t in range(20):
    obs_pos = obstacle_positions[t]
    observation = {
        'timestamp': float(t),
        'obstacles': [obs_pos],
        'goal_position': goal_pos,
    }
    pred_space.update_from_observation(start_pos, observation)

# 预测曲率
midpoint = (15, 10)
current_curvature = pred_space.predict_future_curvature(midpoint, steps_ahead=0)
future_curvature = pred_space.predict_future_curvature(midpoint, steps_ahead=5)

print(f"Curvature prediction at {midpoint}:")
print(f"  Current: {current_curvature:.3f}")
print(f"  5 steps ahead: {future_curvature:.3f}")
print()

# ============================================================================
# 8. Comparison with Static Space
# ============================================================================

print("=" * 70)
print("8. Comparison: Temporal vs Static Space")
print("=" * 70)
print()

# 静态空�?static_space = create_space("ricci", width, height, curvature_scale=1.5)
static_space.update_from_observation(start_pos, {
    'obstacles': [obstacle_positions[-1]],
    'goal_position': goal_pos
})

solver_static = GeodesicSolver(static_space)
result_static = solver_static.solve(start_pos, goal_pos, static_obstacles)

print(f"{'Space Type':<20} {'Steps':<8} {'Cost':<10} {'Time(ms)':<10}")
print("-" * 50)
print(f"{'Static Ricci':<20} {len(result_static.path) if result_static.path else 0:<8} "
      f"{result_static.cost:<10.2f} {result_static.time_ms:<10.2f}")
print(f"{'Temporal Ricci':<20} {len(result.path) if result.path else 0:<8} "
      f"{result.cost:<10.2f} {result.time_ms:<10.2f}")
print()

print("Advantage of TemporalSpace:")
print("  - Learns from historical patterns")
print("  - Predicts future obstacle positions")
print("  - Adapts to periodic changes")
print()

print("=" * 70)
print("TemporalSpace Demo completed")
print("=" * 70)
