#!/usr/bin/env python3
"""
ATLAS Performance Profiler
全面性能分析脚本
"""
import sys
import time
import cProfile
import pstats
import io
import numpy as np
from functools import wraps

sys.path.insert(0, 'src')

from src.core.registry import create_space
from src.core.solver import GeodesicSolver
from src.spaces.continuous import ContinuousRicciSpace


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        return result, elapsed
    return wrapper


class PerformanceAnalyzer:
    def __init__(self):
        self.results = {}

    def run_profile(self, func, *args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)
        return result, s.getvalue()

    def test_discrete_ricci(self):
        print("=" * 60)
        print("Test 1: Discrete Ricci Space Performance")
        print("=" * 60)

        space = create_space('ricci', 100, 100)
        solver = GeodesicSolver(space)

        # Test 1a: Update performance
        obs = {
            'obstacles': [(i, 50) for i in range(40, 60)],
            'goal_position': (90, 90)
        }

        times = []
        for i in range(0, 50, 5):
            start = time.perf_counter()
            space.update_from_observation((i, i), obs)
            times.append((time.perf_counter() - start) * 1000)

        print(f"Update time (avg of {len(times)}): {np.mean(times):.3f}ms ± {np.std(times):.3f}ms")

        # Test 1b: Distance computation
        times = []
        for _ in range(100):
            start = time.perf_counter()
            d = space.compute_distance((10, 10), (90, 90))
            times.append((time.perf_counter() - start) * 1000)

        print(f"Distance compute (avg of 100): {np.mean(times):.3f}ms ± {np.std(times):.3f}ms")

        # Test 1c: Path planning
        times = []
        nodes_list = []
        for _ in range(5):
            space = create_space('ricci', 100, 100)  # Fresh space
            for i in range(0, 50, 10):
                space.update_from_observation((i, i), obs)

            solver = GeodesicSolver(space)
            start = time.perf_counter()
            result = solver.solve((10, 10), (90, 90), set(obs['obstacles']))
            times.append((time.perf_counter() - start) * 1000)
            nodes_list.append(result.nodes_expanded)

        print(f"Path planning (avg of 5): {np.mean(times):.1f}ms ± {np.std(times):.1f}ms")
        print(f"Nodes expanded (avg): {np.mean(nodes_list):.0f}")
        print(f"Time per node: {np.mean(times)/np.mean(nodes_list)*1000:.2f}us")

        return np.mean(times)

    def test_continuous_ricci(self):
        print("\n" + "=" * 60)
        print("Test 2: Continuous Ricci Space Performance")
        print("=" * 60)

        space = ContinuousRicciSpace(curvature_scale=2.0)

        # Test 2a: Field query performance
        obs = {
            'obstacles': [(float(i), 50.0) for i in range(40, 60)],
            'goal_position': (90.0, 90.0)
        }

        # Add some samples first
        for i in range(0, 50, 5):
            space.update_from_observation((float(i), float(i)), obs)

        print(f"Uncertainty samples: {space.uncertainty_field.num_samples()}")

        # Test query performance
        times = []
        for _ in range(100):
            start = time.perf_counter()
            val = space.uncertainty_field.query((25.0, 25.0))
            times.append((time.perf_counter() - start) * 1000)

        print(f"Field query (avg of 100): {np.mean(times):.3f}ms ± {np.std(times):.3f}ms")

        # Test 2b: Distance computation (continuous)
        times = []
        for _ in range(50):
            start = time.perf_counter()
            d = space.compute_distance((10.0, 10.0), (90.0, 90.0))
            times.append((time.perf_counter() - start) * 1000)

        print(f"Continuous distance (avg of 50): {np.mean(times):.3f}ms ± {np.std(times):.3f}ms")

        return np.mean(times)

    def test_space_comparison(self):
        print("\n" + "=" * 60)
        print("Test 3: Space Type Comparison (compute_distance)")
        print("=" * 60)

        spaces_config = [
            ('euclidean', {}),
            ('ricci', {}),
            ('fisher', {}),
            ('conformal', {}),
            ('wasserstein', {}),
        ]

        results = {}
        for name, kwargs in spaces_config:
            try:
                space = create_space(name, 50, 50, **kwargs)

                # Warmup
                for _ in range(10):
                    space.compute_distance((10, 10), (40, 40))

                # Measure
                times = []
                for _ in range(100):
                    start = time.perf_counter()
                    space.compute_distance((10, 10), (40, 40))
                    times.append((time.perf_counter() - start) * 1000)

                avg_time = np.mean(times)
                results[name] = avg_time
                print(f"{name:15s}: {avg_time:.3f}ms ± {np.std(times):.3f}ms")

            except Exception as e:
                print(f"{name:15s}: Error - {e}")

        return results

    def profile_solver_deep(self):
        print("\n" + "=" * 60)
        print("Test 4: Deep Solver Profiling")
        print("=" * 60)

        space = create_space('ricci', 50, 50)
        obs = {'obstacles': [(25, i) for i in range(20, 30)]}
        for i in range(0, 40, 10):
            space.update_from_observation((i, i), obs)

        def solve_test():
            solver = GeodesicSolver(space)
            result = solver.solve((5, 5), (45, 45), set(obs['obstacles']))
            return result

        result, profile_output = self.run_profile(solve_test)
        print(f"Solver expanded: {result.nodes_expanded} nodes")
        print("\nTop 20 by cumulative time:")
        print(profile_output[:2000])


if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()

    # Run tests
    analyzer.test_discrete_ricci()
    analyzer.test_continuous_ricci()
    analyzer.test_space_comparison()
    analyzer.profile_solver_deep()

    print("\n" + "=" * 60)
    print("Performance Analysis Complete")
    print("=" * 60)
