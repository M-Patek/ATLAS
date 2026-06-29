#!/usr/bin/env python3
"""
ATLAS Performance Benchmark Suite
性能基准测试套件

运行：python experiments/benchmarks/perf_benchmark.py
"""
import sys
import time
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, 'src')

from src.core.registry import create_space
from src.core.solver import GeodesicSolver
from src.core.optimized_solvers import JPSGeodesicSolver
from src.spaces.continuous import ContinuousRicciSpace


class PerformanceBenchmark:
    """性能基准测试"""

    def __init__(self):
        self.results = {}
        self.timestamp = datetime.now().isoformat()

    def run_all(self):
        """运行所有基准测�?""
        print("=" * 70)
        print("ATLAS Performance Benchmark")
        print(f"Timestamp: {self.timestamp}")
        print("=" * 70)

        self.benchmark_discrete_spaces()
        self.benchmark_continuous_spaces()
        self.benchmark_solvers()
        self.benchmark_path_planning()

        self.save_results()
        self.print_summary()

    def benchmark_discrete_spaces(self):
        """离散空间基准"""
        print("\n" + "-" * 70)
        print("Discrete Spaces - compute_distance")
        print("-" * 70)

        spaces = ['euclidean', 'ricci', 'fisher', 'conformal', 'wasserstein']
        results = {}

        for name in spaces:
            space = create_space(name, 50, 50)

            # Warmup
            for _ in range(10):
                space.compute_distance((10, 10), (40, 40))

            # Benchmark
            times = []
            for _ in range(1000):
                start = time.perf_counter()
                space.compute_distance((10, 10), (40, 40))
                times.append((time.perf_counter() - start) * 1000)

            avg_time = np.mean(times)
            std_time = np.std(times)
            results[name] = {'mean_ms': avg_time, 'std_ms': std_time}

            print(f"  {name:12s}: {avg_time:.4f}ms ± {std_time:.4f}ms")

        self.results['discrete_spaces'] = results

    def benchmark_continuous_spaces(self):
        """连续空间基准"""
        print("\n" + "-" * 70)
        print("Continuous Spaces")
        print("-" * 70)

        results = {}

        # Field query
        space = ContinuousRicciSpace()
        obs = {'obstacles': [(float(i), 50.0) for i in range(40, 60)]}

        for i in range(0, 50, 5):
            space.update_from_observation((float(i), float(i)), obs)

        times = []
        for _ in range(1000):
            start = time.perf_counter()
            val = space.uncertainty_field.query((25.0, 25.0))
            times.append((time.perf_counter() - start) * 1000)

        results['field_query'] = {'mean_ms': np.mean(times), 'std_ms': np.std(times)}
        print(f"  Field query : {np.mean(times):.4f}ms ± {np.std(times):.4f}ms")

        # Continuous distance
        times = []
        for _ in range(100):
            start = time.perf_counter()
            d = space.compute_distance((10.0, 10.0), (40.0, 40.0))
            times.append((time.perf_counter() - start) * 1000)

        results['continuous_distance'] = {'mean_ms': np.mean(times), 'std_ms': np.std(times)}
        print(f"  Distance    : {np.mean(times):.4f}ms ± {np.std(times):.4f}ms")

        self.results['continuous_spaces'] = results

    def benchmark_solvers(self):
        """求解器基�?""
        print("\n" + "-" * 70)
        print("Solvers")
        print("-" * 70)

        space = create_space('ricci', 50, 50)
        obs = {'obstacles': [(25, i) for i in range(20, 30)]}
        for i in range(0, 40, 10):
            space.update_from_observation((i, i), obs)

        solvers = [
            ('Standard A*', GeodesicSolver(space)),
            ('JPS Cached', JPSGeodesicSolver(space)),
        ]

        results = {}
        for name, solver in solvers:
            times = []
            nodes_list = []

            for _ in range(5):
                start = time.perf_counter()
                result = solver.solve((5, 5), (45, 45), set(obs['obstacles']))
                times.append((time.perf_counter() - start) * 1000)
                nodes_list.append(result.nodes_expanded)

            results[name] = {
                'mean_ms': np.mean(times),
                'std_ms': np.std(times),
                'nodes': np.mean(nodes_list)
            }

            print(f"  {name:15s}: {np.mean(times):6.1f}ms, nodes: {np.mean(nodes_list):.0f}")

        self.results['solvers'] = results

    def benchmark_path_planning(self):
        """路径规划场景基准"""
        print("\n" + "-" * 70)
        print("Path Planning Scenarios")
        print("-" * 70)

        scenarios = [
            ('Open Field', 50, 50, set(), (5, 5), (45, 45)),
            ('Narrow Gap', 50, 50, {(25, i) for i in range(20)} | {(25, i) for i in range(30, 50)}, (5, 25), (45, 25)),
            ('Maze-like', 50, 50, {(i, 25) for i in range(10, 40, 5)}, (5, 5), (45, 45)),
        ]

        results = {}
        for name, width, height, obstacles, start, goal in scenarios:
            space = create_space('ricci', width, height)
            solver = GeodesicSolver(space)

            times = []
            for _ in range(5):
                start_time = time.perf_counter()
                result = solver.solve(start, goal, obstacles)
                times.append((time.perf_counter() - start_time) * 1000)

            results[name] = {
                'mean_ms': np.mean(times),
                'std_ms': np.std(times),
                'success': result.success if 'result' in dir() else False
            }

            print(f"  {name:15s}: {np.mean(times):6.1f}ms")

        self.results['scenarios'] = results

    def save_results(self):
        """保存结果"""
        output = {
            'timestamp': self.timestamp,
            'results': self.results
        }

        filename = f"benchmark_results_{self.timestamp[:10]}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to: {filename}")

    def print_summary(self):
        """打印摘要"""
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)

        # 计算相对于基线的改进
        if 'discrete_spaces' in self.results:
            euclidean_time = self.results['discrete_spaces']['euclidean']['mean_ms']
            ricci_time = self.results['discrete_spaces']['ricci']['mean_ms']
            overhead = (ricci_time / euclidean_time - 1) * 100
            print(f"Ricci overhead vs Euclidean: {overhead:.0f}%")

        if 'solvers' in self.results:
            standard_time = self.results['solvers']['Standard A*']['mean_ms']
            jps_time = self.results['solvers']['JPS Cached']['mean_ms']
            improvement = (standard_time - jps_time) / standard_time * 100
            print(f"JPS improvement over A*: {improvement:.0f}%")


if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    benchmark.run_all()
