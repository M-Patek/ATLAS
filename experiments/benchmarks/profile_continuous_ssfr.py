"""
Performance Profiler for Continuous SSFR

分析连续SSFR的性能瓶颈
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import numpy as np
import math
from typing import Dict, List, Tuple

from src.spaces.continuous import (
    ContinuousField, ContinuousRicciSpace,
    ContinuousFisherSpace, ContinuousWassersteinSpace,
)
from src.spaces.continuous_ssfr import ContinuousSSFR
from src.kitchen import create_demo_kitchen


def profile_continuous_field():
    """分析ContinuousField性能"""
    print("=" * 70)
    print("PROFILE: ContinuousField")
    print("=" * 70)

    field = ContinuousField(default_value=0.0)

    # 1. 添加采样点性能
    n_samples = 1000
    start = time.time()
    for i in range(n_samples):
        x = np.random.random() * 10
        y = np.random.random() * 10
        field.add_sample((x, y), np.random.random())
    add_time = time.time() - start
    print(f"Add {n_samples} samples: {add_time*1000:.2f}ms ({add_time/n_samples*1000:.3f}ms/sample)")

    # 2. 查询性能
    n_queries = 1000
    start = time.time()
    for i in range(n_queries):
        x = np.random.random() * 10
        y = np.random.random() * 10
        field.query((x, y))
    query_time = time.time() - start
    print(f"Query {n_queries} times: {query_time*1000:.2f}ms ({query_time/n_queries*1000:.3f}ms/query)")

    # 3. 批量查询性能
    positions = [(np.random.random()*10, np.random.random()*10) for _ in range(n_queries)]
    start = time.time()
    field.query_batch(positions)
    batch_time = time.time() - start
    print(f"Batch query {n_queries}: {batch_time*1000:.2f}ms ({batch_time/n_queries*1000:.3f}ms/query)")

    # 4. 缓存效果
    # 重复查询同一位置
    pos = (5.0, 5.0)
    start = time.time()
    for _ in range(1000):
        field.query(pos)
    cached_time = time.time() - start
    print(f"Cached query 1000x: {cached_time*1000:.2f}ms ({cached_time/1000*1000:.3f}ms/query)")

    return {
        'add_time_ms': add_time * 1000,
        'query_time_ms': query_time * 1000,
        'batch_time_ms': batch_time * 1000,
        'cached_time_ms': cached_time * 1000,
    }


def profile_continuous_space():
    """分析ContinuousSpace性能"""
    print("\n" + "=" * 70)
    print("PROFILE: ContinuousSpace")
    print("=" * 70)

    space = ContinuousRicciSpace()

    # 预填充一些数�?    for _ in range(100):
        x = np.random.random() * 10
        y = np.random.random() * 10
        space.update_from_observation((x, y), {
            'obstacles': [(x + 0.5, y + 0.5)],
            'goal_position': (5.0, 5.0),
        })

    # 1. 距离计算
    n = 100
    start = time.time()
    for _ in range(n):
        p1 = (np.random.random()*10, np.random.random()*10)
        p2 = (np.random.random()*10, np.random.random()*10)
        space.compute_distance(p1, p2)
    dist_time = time.time() - start
    print(f"Distance compute {n}x: {dist_time*1000:.2f}ms ({dist_time/n*1000:.3f}ms/call)")

    # 2. 预测
    n = 100
    start = time.time()
    for _ in range(n):
        pos = (np.random.random()*10, np.random.random()*10)
        space.predict_next_state(pos, {
            'goal_position': (5.0, 5.0),
            'obstacles': [],
            'step_size': 0.5,
        })
    pred_time = time.time() - start
    print(f"Predict {n}x: {pred_time*1000:.2f}ms ({pred_time/n*1000:.3f}ms/call)")

    # 3. 更新
    n = 100
    start = time.time()
    for _ in range(n):
        pos = (np.random.random()*10, np.random.random()*10)
        space.update_from_observation(pos, {
            'obstacles': [(pos[0] + 0.5, pos[1] + 0.5)],
            'goal_position': (5.0, 5.0),
        })
    update_time = time.time() - start
    print(f"Update {n}x: {update_time*1000:.2f}ms ({update_time/n*1000:.3f}ms/call)")

    return {
        'dist_time_ms': dist_time * 1000,
        'pred_time_ms': pred_time * 1000,
        'update_time_ms': update_time * 1000,
    }


def profile_continuous_ssfr():
    """分析ContinuousSSFR性能"""
    print("\n" + "=" * 70)
    print("PROFILE: ContinuousSSFR")
    print("=" * 70)

    ssfr = ContinuousSSFR(space_names=['ricci', 'fisher'])

    # 1. 感知
    n = 50
    start = time.time()
    for i in range(n):
        observation = {
            'position': (float(i)*0.1, float(i)*0.1),
            'goal_position': (5.0, 5.0),
            'obstacles': [(2.0, 2.0)],
            'uncertainty': 0.3,
        }
        ssfr.perceive((float(i)*0.1, float(i)*0.1), observation)
    perceive_time = time.time() - start
    print(f"Perceive {n}x: {perceive_time*1000:.2f}ms ({perceive_time/n*1000:.3f}ms/call)")

    # 2. 竞争
    n = 50
    start = time.time()
    for i in range(n):
        observation = {
            'position': (float(i)*0.1, float(i)*0.1),
            'goal_position': (5.0, 5.0),
            'obstacles': [],
        }
        actual = {'position': (float(i)*0.1 + 0.05, float(i)*0.1)}
        ssfr.compete(observation, actual)
    compete_time = time.time() - start
    print(f"Compete {n}x: {compete_time*1000:.2f}ms ({compete_time/n*1000:.3f}ms/call)")

    # 3. 完整步骤
    n = 50
    start = time.time()
    for i in range(n):
        observation = {
            'position': (float(i)*0.1, float(i)*0.1),
            'goal_position': (5.0, 5.0),
            'obstacles': [],
        }
        actual = {'position': (float(i)*0.1 + 0.05, float(i)*0.1)}
        ssfr.step((float(i)*0.1, float(i)*0.1), observation, actual)
    step_time = time.time() - start
    print(f"Full step {n}x: {step_time*1000:.2f}ms ({step_time/n*1000:.3f}ms/call)")

    stats = ssfr.get_statistics()
    print(f"\nPool size: {stats['pool_stats']['num_structures']}")
    print(f"Reuse rate: {stats.get('reuse_rate', 0):.3f}")

    return {
        'perceive_time_ms': perceive_time * 1000,
        'compete_time_ms': compete_time * 1000,
        'step_time_ms': step_time * 1000,
    }


def profile_kitchen_integration():
    """分析厨房集成性能"""
    print("\n" + "=" * 70)
    print("PROFILE: Kitchen Integration")
    print("=" * 70)

    from test_continuous_ssfr import (
        ContinuousPhysicalSSFR, ContinuousSSFRTaskPlanner
    )

    kitchen = create_demo_kitchen()
    robot_id = list(kitchen.robots.keys())[0]

    physical_ssfr = ContinuousPhysicalSSFR(kitchen)
    planner = ContinuousSSFRTaskPlanner(physical_ssfr)

    planner.assign_task(robot_id, 'make_coffee')

    # 运行并计�?    n = 100
    start = time.time()
    for _ in range(n):
        kitchen.step()
        planner.step(robot_id)
    total_time = time.time() - start

    print(f"Kitchen + SSFR {n} steps: {total_time*1000:.2f}ms")
    print(f"Per step: {total_time/n*1000:.3f}ms")

    # 分析各组件时�?    stats = physical_ssfr.get_statistics()
    print(f"\nSSFR perceive time: {stats.get('avg_perceive_time_ms', 0):.3f}ms")
    print(f"SSFR compete time: {stats.get('avg_compete_time_ms', 0):.3f}ms")
    print(f"Pool size: {stats['pool_stats']['num_structures']}")

    return {
        'total_time_ms': total_time * 1000,
        'per_step_ms': total_time / n * 1000,
    }


def run_all_profiles():
    """运行所有性能分析"""
    print("=" * 70)
    print("Continuous SSFR Performance Profile")
    print("=" * 70)

    results = {}
    results['field'] = profile_continuous_field()
    results['space'] = profile_continuous_space()
    results['ssfr'] = profile_continuous_ssfr()
    results['kitchen'] = profile_kitchen_integration()

    # 汇�?    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n--- ContinuousField ---")
    print(f"  Add sample: {results['field']['add_time_ms']/1000:.3f}ms/1000")
    print(f"  Query: {results['field']['query_time_ms']/1000:.3f}ms/1000")
    print(f"  Batch query: {results['field']['batch_time_ms']/1000:.3f}ms/1000")
    print(f"  Cached query: {results['field']['cached_time_ms']/1000:.3f}ms/1000")

    print("\n--- ContinuousSpace ---")
    print(f"  Distance: {results['space']['dist_time_ms']/100:.3f}ms/100")
    print(f"  Predict: {results['space']['pred_time_ms']/100:.3f}ms/100")
    print(f"  Update: {results['space']['update_time_ms']/100:.3f}ms/100")

    print("\n--- ContinuousSSFR ---")
    print(f"  Perceive: {results['ssfr']['perceive_time_ms']/50:.3f}ms/50")
    print(f"  Compete: {results['ssfr']['compete_time_ms']/50:.3f}ms/50")
    print(f"  Full step: {results['ssfr']['step_time_ms']/50:.3f}ms/50")

    print("\n--- Kitchen Integration ---")
    print(f"  Total: {results['kitchen']['total_time_ms']:.2f}ms/100")
    print(f"  Per step: {results['kitchen']['per_step_ms']:.3f}ms")

    # 识别瓶颈
    print("\n--- Bottlenecks ---")
    all_times = [
        ('Field query', results['field']['query_time_ms'] / 1000),
        ('Space predict', results['space']['pred_time_ms'] / 100),
        ('SSFR perceive', results['ssfr']['perceive_time_ms'] / 50),
        ('SSFR compete', results['ssfr']['compete_time_ms'] / 50),
    ]
    all_times.sort(key=lambda x: x[1], reverse=True)

    for name, t in all_times:
        print(f"  {name}: {t:.3f}ms")

    return results


if __name__ == "__main__":
    results = run_all_profiles()
