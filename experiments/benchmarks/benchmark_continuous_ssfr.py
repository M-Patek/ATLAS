"""
Performance Comparison: Original vs Optimized Continuous SSFR

对比原始实现和优化实现的性能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import numpy as np
import math

from src.spaces.continuous import ContinuousField, ContinuousRicciSpace
from src.spaces.continuous_ssfr import ContinuousSSFR
from src.spaces.continuous_optimized import (
    OptimizedContinuousField,
    OptimizedContinuousRicciSpace,
    OptimizedContinuousSSFR,
)


def benchmark_field():
    """对比ContinuousField性能"""
    print("=" * 70)
    print("BENCHMARK: ContinuousField")
    print("=" * 70)

    # 原始
    original = ContinuousField(default_value=0.0)
    # 优化
    optimized = OptimizedContinuousField(default_value=0.0)

    # 添加采样�?    n = 1000
    positions = [(np.random.random()*10, np.random.random()*10) for _ in range(n)]

    # 添加性能
    start = time.time()
    for pos in positions:
        original.add_sample(pos, np.random.random())
    orig_add = time.time() - start

    start = time.time()
    for pos in positions:
        optimized.add_sample(pos, np.random.random())
    opt_add = time.time() - start

    print(f"\nAdd {n} samples:")
    print(f"  Original: {orig_add*1000:.2f}ms ({orig_add/n*1000:.3f}ms/sample)")
    print(f"  Optimized: {opt_add*1000:.2f}ms ({opt_add/n*1000:.3f}ms/sample)")
    print(f"  Speedup: {orig_add/opt_add:.2f}x")

    # 查询性能
    n_query = 1000
    query_positions = [(np.random.random()*10, np.random.random()*10) for _ in range(n_query)]

    start = time.time()
    for pos in query_positions:
        original.query(pos)
    orig_query = time.time() - start

    start = time.time()
    for pos in query_positions:
        optimized.query(pos)
    opt_query = time.time() - start

    print(f"\nQuery {n_query} times:")
    print(f"  Original: {orig_query*1000:.2f}ms ({orig_query/n_query*1000:.3f}ms/query)")
    print(f"  Optimized: {opt_query*1000:.2f}ms ({opt_query/n_query*1000:.3f}ms/query)")
    print(f"  Speedup: {orig_query/opt_query:.2f}x")

    # 批量查询
    start = time.time()
    original.query_batch(query_positions)
    orig_batch = time.time() - start

    start = time.time()
    optimized.query_batch(query_positions)
    opt_batch = time.time() - start

    print(f"\nBatch query {n_query}:")
    print(f"  Original: {orig_batch*1000:.2f}ms")
    print(f"  Optimized: {opt_batch*1000:.2f}ms")
    print(f"  Speedup: {orig_batch/opt_batch:.2f}x")

    return {
        'add_speedup': orig_add / opt_add,
        'query_speedup': orig_query / opt_query,
        'batch_speedup': orig_batch / opt_batch,
    }


def benchmark_space():
    """对比ContinuousSpace性能"""
    print("\n" + "=" * 70)
    print("BENCHMARK: ContinuousSpace")
    print("=" * 70)

    # 原始
    original = ContinuousRicciSpace()
    # 优化
    optimized = OptimizedContinuousRicciSpace()

    # 预填充数�?    for _ in range(50):
        pos = (np.random.random()*10, np.random.random()*10)
        obs = {
            'obstacles': [(pos[0]+0.5, pos[1]+0.5)],
            'goal_position': (5.0, 5.0),
        }
        original.update_from_observation(pos, obs)
        optimized.update_from_observation(pos, obs)

    # 距离计算
    n = 100
    start = time.time()
    for _ in range(n):
        p1 = (np.random.random()*10, np.random.random()*10)
        p2 = (np.random.random()*10, np.random.random()*10)
        original.compute_distance(p1, p2)
    orig_dist = time.time() - start

    start = time.time()
    for _ in range(n):
        p1 = (np.random.random()*10, np.random.random()*10)
        p2 = (np.random.random()*10, np.random.random()*10)
        optimized.compute_distance(p1, p2)
    opt_dist = time.time() - start

    print(f"\nDistance compute {n}x:")
    print(f"  Original: {orig_dist*1000:.2f}ms ({orig_dist/n*1000:.3f}ms/call)")
    print(f"  Optimized: {opt_dist*1000:.2f}ms ({opt_dist/n*1000:.3f}ms/call)")
    print(f"  Speedup: {orig_dist/opt_dist:.2f}x")

    # 预测
    n = 100
    start = time.time()
    for _ in range(n):
        pos = (np.random.random()*10, np.random.random()*10)
        original.predict_next_state(pos, {
            'goal_position': (5.0, 5.0),
            'obstacles': [],
            'step_size': 0.5,
        })
    orig_pred = time.time() - start

    start = time.time()
    for _ in range(n):
        pos = (np.random.random()*10, np.random.random()*10)
        optimized.predict_next_state(pos, {
            'goal_position': (5.0, 5.0),
            'obstacles': [],
            'step_size': 0.5,
        })
    opt_pred = time.time() - start

    print(f"\nPredict {n}x:")
    print(f"  Original: {orig_pred*1000:.2f}ms ({orig_pred/n*1000:.3f}ms/call)")
    print(f"  Optimized: {opt_pred*1000:.2f}ms ({opt_pred/n*1000:.3f}ms/call)")
    print(f"  Speedup: {orig_pred/opt_pred:.2f}x")

    # 更新
    n = 100
    start = time.time()
    for _ in range(n):
        pos = (np.random.random()*10, np.random.random()*10)
        original.update_from_observation(pos, {
            'obstacles': [(pos[0]+0.5, pos[1]+0.5)],
            'goal_position': (5.0, 5.0),
        })
    orig_update = time.time() - start

    start = time.time()
    for _ in range(n):
        pos = (np.random.random()*10, np.random.random()*10)
        optimized.update_from_observation(pos, {
            'obstacles': [(pos[0]+0.5, pos[1]+0.5)],
            'goal_position': (5.0, 5.0),
        })
    opt_update = time.time() - start

    print(f"\nUpdate {n}x:")
    print(f"  Original: {orig_update*1000:.2f}ms ({orig_update/n*1000:.3f}ms/call)")
    print(f"  Optimized: {opt_update*1000:.2f}ms ({opt_update/n*1000:.3f}ms/call)")
    print(f"  Speedup: {orig_update/opt_update:.2f}x")

    return {
        'dist_speedup': orig_dist / opt_dist,
        'pred_speedup': orig_pred / opt_pred,
        'update_speedup': orig_update / opt_update,
    }


def benchmark_ssfr():
    """对比ContinuousSSFR性能"""
    print("\n" + "=" * 70)
    print("BENCHMARK: ContinuousSSFR")
    print("=" * 70)

    # 原始
    original = ContinuousSSFR(space_names=['ricci'])
    # 优化
    optimized = OptimizedContinuousSSFR(space_names=['ricci'])

    # 感知
    n = 50
    start = time.time()
    for i in range(n):
        observation = {
            'position': (float(i)*0.1, float(i)*0.1),
            'goal_position': (5.0, 5.0),
            'obstacles': [(2.0, 2.0)],
            'uncertainty': 0.3,
        }
        original.perceive((float(i)*0.1, float(i)*0.1), observation)
    orig_perceive = time.time() - start

    start = time.time()
    for i in range(n):
        observation = {
            'position': (float(i)*0.1, float(i)*0.1),
            'goal_position': (5.0, 5.0),
            'obstacles': [(2.0, 2.0)],
            'uncertainty': 0.3,
        }
        optimized.perceive((float(i)*0.1, float(i)*0.1), observation)
    opt_perceive = time.time() - start

    print(f"\nPerceive {n}x:")
    print(f"  Original: {orig_perceive*1000:.2f}ms ({orig_perceive/n*1000:.3f}ms/call)")
    print(f"  Optimized: {opt_perceive*1000:.2f}ms ({opt_perceive/n*1000:.3f}ms/call)")
    print(f"  Speedup: {orig_perceive/opt_perceive:.2f}x")

    # 竞争
    n = 50
    start = time.time()
    for i in range(n):
        observation = {
            'position': (float(i)*0.1, float(i)*0.1),
            'goal_position': (5.0, 5.0),
            'obstacles': [],
        }
        actual = {'position': (float(i)*0.1 + 0.05, float(i)*0.1)}
        original.compete(observation, actual)
    orig_compete = time.time() - start

    start = time.time()
    for i in range(n):
        observation = {
            'position': (float(i)*0.1, float(i)*0.1),
            'goal_position': (5.0, 5.0),
            'obstacles': [],
        }
        actual = {'position': (float(i)*0.1 + 0.05, float(i)*0.1)}
        optimized.compete(observation, actual)
    opt_compete = time.time() - start

    print(f"\nCompete {n}x:")
    print(f"  Original: {orig_compete*1000:.2f}ms ({orig_compete/n*1000:.3f}ms/call)")
    print(f"  Optimized: {opt_compete*1000:.2f}ms ({opt_compete/n*1000:.3f}ms/call)")
    print(f"  Speedup: {orig_compete/opt_compete:.2f}x")

    return {
        'perceive_speedup': orig_perceive / opt_perceive,
        'compete_speedup': orig_compete / opt_compete,
    }


def run_all_benchmarks():
    """运行所有对比测�?""
    print("=" * 70)
    print("Continuous SSFR Performance Comparison")
    print("Original vs Optimized")
    print("=" * 70)

    results = {}
    results['field'] = benchmark_field()
    results['space'] = benchmark_space()
    results['ssfr'] = benchmark_ssfr()

    # 汇�?    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n--- ContinuousField ---")
    print(f"  Add speedup: {results['field']['add_speedup']:.2f}x")
    print(f"  Query speedup: {results['field']['query_speedup']:.2f}x")
    print(f"  Batch speedup: {results['field']['batch_speedup']:.2f}x")

    print("\n--- ContinuousSpace ---")
    print(f"  Distance speedup: {results['space']['dist_speedup']:.2f}x")
    print(f"  Predict speedup: {results['space']['pred_speedup']:.2f}x")
    print(f"  Update speedup: {results['space']['update_speedup']:.2f}x")

    print("\n--- ContinuousSSFR ---")
    print(f"  Perceive speedup: {results['ssfr']['perceive_speedup']:.2f}x")
    print(f"  Compete speedup: {results['ssfr']['compete_speedup']:.2f}x")

    # 平均加速比
    all_speedups = [
        results['field']['add_speedup'],
        results['field']['query_speedup'],
        results['field']['batch_speedup'],
        results['space']['dist_speedup'],
        results['space']['pred_speedup'],
        results['space']['update_speedup'],
        results['ssfr']['perceive_speedup'],
        results['ssfr']['compete_speedup'],
    ]
    avg_speedup = sum(all_speedups) / len(all_speedups)
    print(f"\nAverage speedup: {avg_speedup:.2f}x")

    return results


if __name__ == "__main__":
    results = run_all_benchmarks()
