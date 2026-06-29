"""
Test: TemporalSpace
测试时序空间功能
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from src.core import GeodesicSolver
from src.spaces.temporal import (
    TemporalSpace, CircularBuffer, FieldPredictor, PeriodicityDetector
)


def test_circular_buffer():
    """测试循环缓冲�?""
    print("Test 1: CircularBuffer")

    buffer = CircularBuffer(maxlen=5)

    # 添加数据
    for i in range(10):
        from src.spaces.temporal import TemporalSnapshot
        snapshot = TemporalSnapshot(
            timestamp=float(i),
            fields={'test': np.ones((10, 10)) * i},
            position=(i, i),
            observation={}
        )
        buffer.append(snapshot)

    assert len(buffer) == 5, "Buffer should only keep 5 items"

    recent = buffer.get_recent(3)
    assert len(recent) == 3, "Should get 3 recent items"

    timestamps, values = buffer.get_field_history('test', (5, 5))
    assert len(timestamps) == 5, "Should get 5 timestamps"

    print("  [OK] CircularBuffer works correctly")


def test_field_predictor():
    """测试场预测器"""
    print("Test 2: FieldPredictor")

    predictor = FieldPredictor()

    # 测试趋势预测
    timestamps = list(range(10))
    values = [0.1 * t for t in timestamps]  # 线性趋�?
    future_times = [10, 11, 12]
    predictions = predictor.predict_trend(timestamps, values, future_times)

    assert len(predictions) == 3, "Should get 3 predictions"
    assert predictions[0] > 0.9, "Should predict increasing trend"

    # 测试GP预测（如果有sklearn�?    predictions_gp, uncertainties = predictor.predict_gp(timestamps, values, future_times)
    assert len(predictions_gp) == 3, "GP should also return 3 predictions"
    assert len(uncertainties) == 3, "GP should return uncertainties"

    print("  [OK] FieldPredictor works correctly")


def test_periodicity_detector():
    """测试周期性检测器"""
    print("Test 3: PeriodicityDetector")

    detector = PeriodicityDetector(min_period=3, max_period=20)

    # 生成周期信号
    period = 7
    signal = [np.sin(2 * np.pi * t / period) for t in range(50)]

    detected = detector.detect_period_autocorr(signal)
    assert detected is not None, "Should detect period"
    assert abs(detected - period) <= 1, f"Detected period {detected} should be close to {period}"

    # 随机信号不应该有周期
    random_signal = np.random.randn(50).tolist()
    detected_random = detector.detect_period_autocorr(random_signal)
    assert detected_random is None, "Random signal should not have period"

    print("  [OK] PeriodicityDetector works correctly")


def test_temporal_space():
    """测试TemporalSpace"""
    print("Test 4: TemporalSpace")

    space = TemporalSpace(
        width=20, height=15,
        base_space_type="ricci",
        history_length=30
    )

    # 更新空间
    for t in range(20):
        space.update_from_observation(
            (5, 7),
            {
                'timestamp': float(t),
                'obstacles': [(10, 7)] if t < 10 else [(12, 7)],
                'goal_position': (18, 7)
            }
        )

    assert len(space.history) == 20, "Should have 20 snapshots"
    assert space.current_time == 19.0, "Current time should be 19"

    # 测试历史查询
    timestamps, values = space.get_field_history('uncertainty', (10, 7))
    assert len(timestamps) == 20, "Should have 20 history points"

    # 测试规划
    solver = GeodesicSolver(space)
    result = solver.solve((5, 7), (18, 7), {(12, 7)})

    assert result.success, "Planning should succeed"
    assert len(result.path) > 0, "Should have path"

    # 测试统计
    stats = space.get_temporal_statistics()
    assert 'history_length' in stats, "Should have history_length stat"
    assert stats['history_length'] == 20, "History length should be 20"

    print("  [OK] TemporalSpace works correctly")


def test_predictive_ricci():
    """测试PredictiveRicciSpace"""
    print("Test 5: PredictiveRicciSpace")

    from src.spaces.temporal import PredictiveRicciSpace

    space = PredictiveRicciSpace(width=20, height=15, history_length=20)

    # 模拟移动障碍�?    for t in range(15):
        obs_x = 10 + int(2 * np.sin(2 * np.pi * t / 8))
        space.update_from_observation(
            (5, 7),
            {
                'timestamp': float(t),
                'obstacles': [(obs_x, 7)],
                'goal_position': (18, 7)
            }
        )

    # 测试预测
    current = space.predict_future_curvature((10, 7), steps_ahead=0)
    future = space.predict_future_curvature((10, 7), steps_ahead=3)

    print(f"  Current curvature: {current:.3f}, Future: {future:.3f}")
    print("  [OK] PredictiveRicciSpace works correctly")


def test_comparison():
    """对比Temporal和静态空�?""
    print("Test 6: Temporal vs Static Comparison")

    from src.core.registry import create_space

    width, height = 30, 20
    start = (5, 10)
    goal = (25, 10)

    # 静态空�?    static = create_space("ricci", width, height)
    static.update_from_observation(start, {'goal_position': goal})

    # 时序空间
    temporal = TemporalSpace(width, height, base_space_type="ricci")
    for t in range(10):
        temporal.update_from_observation(start, {
            'timestamp': float(t),
            'goal_position': goal
        })

    # 两者都应该能规�?    solver_s = GeodesicSolver(static)
    solver_t = GeodesicSolver(temporal)

    result_s = solver_s.solve(start, goal)
    result_t = solver_t.solve(start, goal)

    assert result_s.success and result_t.success, "Both should succeed"

    print(f"  Static: {len(result_s.path)} steps, Temporal: {len(result_t.path)} steps")
    print("  [OK] Both spaces can plan successfully")


def main():
    print()
    print("=" * 70)
    print("ATLAS: TemporalSpace Test Suite")
    print("=" * 70)
    print()

    try:
        test_circular_buffer()
        test_field_predictor()
        test_periodicity_detector()
        test_temporal_space()
        test_predictive_ricci()
        test_comparison()
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print()
    print("=" * 70)
    print("All TemporalSpace tests passed!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
