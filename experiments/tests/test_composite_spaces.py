"""
Test: Composite Spaces
测试复合空间功能
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from atlas.core import GeodesicSolver
from atlas.core.registry import create_space
from atlas.spaces.composite import (
    ProductSpace, HierarchicalSpace, MixedSpace,
    create_exploration_navigation_balance,
    create_adaptive_exploration_space
)


def test_product_space():
    """测试乘积空间"""
    print("=" * 70)
    print("Test 1: Product Space (Ricci + Conformal)")
    print("=" * 70)

    width, height = 30, 20

    # 创建子空间
    ricci = create_space("ricci", width, height, curvature_scale=1.5)
    conformal = create_space("conformal", width, height)

    # 创建乘积空间
    product = ProductSpace(width, height, [
        ("ricci", ricci, 0.6),
        ("conformal", conformal, 0.4)
    ])

    # 设置场景
    start = (5, 10)
    goal = (25, 10)
    obstacles = {(15, y) for y in range(5, 16) if y != 10}

    # 更新空间
    product.update_from_observation(start, {
        'obstacles': list(obstacles),
        'goal_position': goal
    })

    # 规划
    solver = GeodesicSolver(product)
    result = solver.solve(start, goal, obstacles)

    print(f"Weights: Ricci=0.6, Conformal=0.4")
    print(f"Success: {result.success}")
    print(f"Steps: {len(result.path) if result.path else 0}")
    print(f"Cost: {result.cost:.2f}")

    # 对比单独空间
    ricci.update_from_observation(start, {
        'obstacles': list(obstacles),
        'goal_position': goal
    })
    conformal.update_from_observation(start, {
        'obstacles': list(obstacles),
        'goal_position': goal
    })

    solver_r = GeodesicSolver(ricci)
    solver_c = GeodesicSolver(conformal)

    result_r = solver_r.solve(start, goal, obstacles)
    result_c = solver_c.solve(start, goal, obstacles)

    print(f"\nComparison:")
    print(f"  Ricci alone:     {len(result_r.path) if result_r.path else 0} steps, cost={result_r.cost:.2f}")
    print(f"  Conformal alone: {len(result_c.path) if result_c.path else 0} steps, cost={result_c.cost:.2f}")
    print(f"  Product:         {len(result.path) if result.path else 0} steps, cost={result.cost:.2f}")

    # 测试动态权重调整
    print(f"\nAdjusting weights to Ricci=0.3, Conformal=0.7...")
    product.adjust_weights({"ricci": 0.3, "conformal": 0.7})

    result2 = solver.solve(start, goal, obstacles)
    print(f"New cost: {result2.cost:.2f}")

    print()


def test_hierarchical_space():
    """测试层次空间"""
    print("=" * 70)
    print("Test 2: Hierarchical Space")
    print("=" * 70)

    # 创建大小空间
    global_space = create_space("ricci", 10, 10, curvature_scale=1.0)  # 粗粒度
    local_space = create_space("conformal", 50, 50)  # 细粒度

    hierarchical = HierarchicalSpace(
        width=50, height=50,
        global_space=global_space,
        local_space=local_space,
        transition_threshold=15.0
    )

    # 场景
    start = (5, 25)
    goal = (45, 25)

    hierarchical.update_from_observation(start, {'goal_position': goal})

    # 测试距离计算
    short_dist = hierarchical.compute_distance((10, 25), (12, 25))  # 短距离，用局部
    long_dist = hierarchical.compute_distance(start, goal)  # 长距离，用全局

    print(f"Short distance (local):  {short_dist:.2f}")
    print(f"Long distance (global):  {long_dist:.2f}")

    # 规划
    solver = GeodesicSolver(hierarchical)
    result = solver.solve(start, goal)

    print(f"Success: {result.success}")
    print(f"Steps: {len(result.path) if result.path else 0}")

    # 层次规划
    global_path, local_segments = hierarchical.plan_hierarchically(start, goal)
    print(f"Global path waypoints: {len(global_path)}")
    print(f"Local segments: {len(local_segments)} points")

    print()


def test_mixed_space():
    """测试混合空间"""
    print("=" * 70)
    print("Test 3: Mixed Space (Context Switching)")
    print("=" * 70)

    width, height = 30, 20

    # 创建子空间
    ricci = create_space("ricci", width, height, curvature_scale=2.0)
    euclidean = create_space("euclidean", width, height)

    # 混合空间：高uncertainty时用Ricci，否则用Euclidean
    mixed = MixedSpace(width, height, [
        (ricci, lambda ctx: ctx.get('uncertainty', 0) > 0.5),
        (euclidean, lambda ctx: True),  # 默认
    ])

    start = (5, 10)
    goal = (25, 10)

    # 测试1: 低uncertainty（应该选Euclidean）
    mixed.update_from_observation(start, {
        'uncertainty': 0.3,
        'goal_position': goal
    })

    active = mixed.get_active_space_name()
    print(f"Low uncertainty (0.3): Active space = {active}")

    dist1 = mixed.compute_distance(start, goal)
    print(f"Distance: {dist1:.2f}")

    # 测试2: 高uncertainty（应该选Ricci）
    mixed.update_from_observation(start, {
        'uncertainty': 0.8,
        'goal_position': goal
    })

    active = mixed.get_active_space_name()
    print(f"High uncertainty (0.8): Active space = {active}")

    dist2 = mixed.compute_distance(start, goal)
    print(f"Distance: {dist2:.2f}")

    # 规划
    solver = GeodesicSolver(mixed)
    result = solver.solve(start, goal)

    print(f"Success: {result.success}")
    print(f"Steps: {len(result.path) if result.path else 0}")

    print()


def test_helper_functions():
    """测试辅助函数"""
    print("=" * 70)
    print("Test 4: Helper Functions")
    print("=" * 70)

    # 创建探索-导航平衡空间
    space = create_exploration_navigation_balance(
        width=30, height=20,
        exploration_weight=0.7
    )

    print(f"Created exploration-navigation balance space")
    print(f"Name: {space.name}")
    print(f"Sub-spaces: {[name for name, _, _ in space.normalized_weights]}")
    print(f"Weights: {[w for _, _, w in space.normalized_weights]}")

    # 测试规划
    start = (5, 10)
    goal = (25, 10)

    space.update_from_observation(start, {'goal_position': goal})

    solver = GeodesicSolver(space)
    result = solver.solve(start, goal)

    print(f"Planning result: {result.success}, {len(result.path) if result.path else 0} steps")

    # 创建自适应探索空间
    adaptive = create_adaptive_exploration_space(
        width=30, height=20,
        high_uncertainty_threshold=0.6
    )

    print(f"\nCreated adaptive exploration space")
    print(f"Name: {adaptive.name}")
    print(f"Blending: {adaptive.blending}")

    print()


def test_comparison():
    """对比不同组合策略"""
    print("=" * 70)
    print("Test 5: Composition Strategy Comparison")
    print("=" * 70)

    width, height = 40, 20
    start = (5, 10)
    goal = (35, 10)
    obstacles = {(20, y) for y in range(5, 16) if y != 10}

    results = []

    for composition in ["euclidean", "manhattan"]:
        ricci = create_space("ricci", width, height, curvature_scale=1.5)
        conformal = create_space("conformal", width, height)

        product = ProductSpace(width, height, [
            ("ricci", ricci, 0.5),
            ("conformal", conformal, 0.5)
        ], composition=composition)

        product.update_from_observation(start, {
            'obstacles': list(obstacles),
            'goal_position': goal
        })

        solver = GeodesicSolver(product)
        result = solver.solve(start, goal, obstacles)

        results.append({
            'composition': composition,
            'steps': len(result.path) if result.path else 0,
            'cost': result.cost,
            'time': result.time_ms
        })

    print(f"{'Composition':<15} {'Steps':<10} {'Cost':<12} {'Time(ms)':<12}")
    print("-" * 50)
    for r in results:
        print(f"{r['composition']:<15} {r['steps']:<10} {r['cost']:<12.2f} {r['time']:<12.2f}")

    print()


def main():
    print()
    print("=" * 70)
    print("ATLAS: Composite Spaces Test Suite")
    print("=" * 70)
    print()

    try:
        test_product_space()
        test_hierarchical_space()
        test_mixed_space()
        test_helper_functions()
        test_comparison()
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 70)
    print("All composite space tests completed")
    print("=" * 70)


if __name__ == "__main__":
    main()
