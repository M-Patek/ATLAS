"""
演示: 空间组合系统

展示 ProductSpace, HierarchicalSpace, MixedSpace 的实际应用
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from atlas.core import GeodesicSolver
from atlas.core.registry import create_space
from atlas.spaces.composite import (
    ProductSpace, HierarchicalSpace, MixedSpace,
    create_exploration_navigation_balance
)

print("=" * 70)
print("ATLAS: Space Composition Demo")
print("=" * 70)
print()

# ============================================================================
# 场景设置
# ============================================================================

print("Setting up scenario...")
width, height = 50, 30

# 复杂迷宫场景
start = (5, 15)
goal = (45, 15)
obstacles = set()

# 垂直墙（有缺口）
for y in range(5, 26):
    if abs(y - 15) > 2:  # 在 y=15 附近留缺口
        obstacles.add((25, y))

# 随机障碍物块
np.random.seed(42)
for _ in range(30):
    x = np.random.randint(10, 40)
    y = np.random.randint(5, 25)
    if (x, y) not in [(25, y) for y in range(12, 19)]:
        obstacles.add((x, y))

print(f"Environment: {width}x{height}")
print(f"Start: {start}, Goal: {goal}")
print(f"Obstacles: {len(obstacles)}")
print()

# ============================================================================
# 1. Product Space Demo
# ============================================================================

print("=" * 70)
print("1. Product Space: Exploration vs Navigation Balance")
print("=" * 70)
print()

# 创建不同权重的乘积空间
weight_configs = [
    ("Pure Explore", 1.0, 0.0),
    ("Explore-heavy", 0.7, 0.3),
    ("Balanced", 0.5, 0.5),
    ("Navigate-heavy", 0.3, 0.7),
    ("Pure Navigate", 0.0, 1.0),
]

product_results = []

for name, explore_w, nav_w in weight_configs:
    ricci = create_space("ricci", width, height, curvature_scale=1.5)
    conformal = create_space("conformal", width, height)

    product = ProductSpace(width, height, [
        ("explore", ricci, explore_w),
        ("navigate", conformal, nav_w)
    ])

    # 初始化
    product.update_from_observation(start, {
        'obstacles': list(obstacles),
        'goal_position': goal
    })

    # 规划
    solver = GeodesicSolver(product)
    result = solver.solve(start, goal, obstacles)

    product_results.append({
        'name': name,
        'weights': (explore_w, nav_w),
        'steps': len(result.path) if result.path else 0,
        'cost': result.cost,
        'time': result.time_ms
    })

print(f"{'Config':<20} {'Weights':<15} {'Steps':<8} {'Cost':<10} {'Time(ms)':<10}")
print("-" * 70)
for r in product_results:
    w_str = f"{r['weights'][0]:.1f}/{r['weights'][1]:.1f}"
    print(f"{r['name']:<20} {w_str:<15} {r['steps']:<8} {r['cost']:<10.2f} {r['time']:<10.2f}")

print()
print("Observation: As exploration weight increases, paths tend to")
print("            explore curvature-rich regions (information hotspots)")
print()

# ============================================================================
# 2. Hierarchical Space Demo
# ============================================================================

print("=" * 70)
print("2. Hierarchical Space: Multi-Scale Planning")
print("=" * 70)
print()

# 创建层次空间
scale_factor = 5  # 全局空间比局部粗5倍
global_width = width // scale_factor
global_height = height // scale_factor

global_space = create_space("ricci", global_width, global_height, curvature_scale=0.5)
local_space = create_space("conformal", width, height)

hierarchical = HierarchicalSpace(
    width=width, height=height,
    global_space=global_space,
    local_space=local_space,
    transition_threshold=10.0
)

# 初始化
hierarchical.update_from_observation(start, {
    'obstacles': list(obstacles),
    'goal_position': goal
})

print(f"Global space: {global_width}x{global_height} (coarse)")
print(f"Local space:  {width}x{height} (fine)")
print(f"Scale factor: {scale_factor}x")
print()

# 层次规划
print("Performing hierarchical planning...")
global_path, local_segments = hierarchical.plan_hierarchically(start, goal)

print(f"Global path waypoints: {len(global_path)}")
print(f"Global path: {global_path[:3]}...{global_path[-3:]}")
print(f"Local detailed segments: {len(local_segments)} points")

# 使用求解器
solver_h = GeodesicSolver(hierarchical)
result_h = solver_h.solve(start, goal, obstacles)

print(f"Solver result: {result_h.success}, {len(result_h.path) if result_h.path else 0} steps")
print(f"Planning time: {result_h.time_ms:.2f}ms")
print()
print("Advantage: Long distances use coarse global space (fast),")
print("           short distances use fine local space (precise)")
print()

# ============================================================================
# 3. Mixed Space Demo
# ============================================================================

print("=" * 70)
print("3. Mixed Space: Context-Aware Switching")
print("=" * 70)
print()

# 创建自适应混合空间
ricci = create_space("ricci", width, height, curvature_scale=2.0)
euclidean = create_space("euclidean", width, height)

# 基于uncertainty动态切换
mixed = MixedSpace(width, height, [
    (ricci, lambda ctx: ctx.get('uncertainty', 0) > 0.6),
    (euclidean, lambda ctx: True),  # 默认
])

print("MixedSpace configuration:")
print("  - uncertainty > 0.6: use Ricci (exploration)")
print("  - uncertainty <= 0.6: use Euclidean (efficiency)")
print()

# 模拟自适应导航
print("Simulating adaptive navigation...")

# 起点（高uncertainty区域）
position = start
uncertainty = 0.8  # 初始高不确定性

path_taken = [position]
active_spaces = []

for step in range(50):  # 最大50步
    if position == goal:
        break

    # 根据当前状态获取激活空间
    context = {
        'position': position,
        'uncertainty': uncertainty,
        'goal_position': goal
    }

    active = mixed.get_active_space_name()
    active_spaces.append(active)

    # 规划下一步
    mixed.update_from_observation(position, context)

    solver_m = GeodesicSolver(mixed)
    result_m = solver_m.solve(position, goal, obstacles)

    if not result_m.success or not result_m.path:
        break

    # 移动（只走1-2步）
    next_pos = result_m.path[min(2, len(result_m.path)-1)]
    position = next_pos
    path_taken.append(position)

    # 模拟：越靠近目标，uncertainty越低
    dist_to_goal = np.sqrt(
        (position[0] - goal[0])**2 + (position[1] - goal[1])**2
    )
    max_dist = np.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
    uncertainty = 0.3 + 0.5 * (dist_to_goal / max_dist)

print(f"Path length: {len(path_taken)} steps")
print(f"Space switches:")

# 统计空间使用
from collections import Counter
space_counts = Counter(active_spaces)
for space_name, count in space_counts.items():
    pct = 100 * count / len(active_spaces) if active_spaces else 0
    print(f"  - {space_name}: {count} steps ({pct:.1f}%)")

print()
print("Advantage: Automatically switches space type based on context")
print()

# ============================================================================
# 4. Comparative Analysis
# ============================================================================

print("=" * 70)
print("4. Comparative Summary")
print("=" * 70)
print()

# 对比所有方法
comparison = []

# 单独空间
for space_name in ["euclidean", "ricci", "conformal"]:
    space = create_space(space_name, width, height)
    space.update_from_observation(start, {
        'obstacles': list(obstacles),
        'goal_position': goal
    })
    solver = GeodesicSolver(space)
    result = solver.solve(start, goal, obstacles)
    comparison.append({
        'method': space_name,
        'steps': len(result.path) if result.path else 0,
        'cost': result.cost,
        'time': result.time_ms
    })

# 组合空间
balance = create_exploration_navigation_balance(width, height, 0.5)
balance.update_from_observation(start, {
    'obstacles': list(obstacles),
    'goal_position': goal
})
solver_b = GeodesicSolver(balance)
result_b = solver_b.solve(start, goal, obstacles)
comparison.append({
    'method': 'product(0.5/0.5)',
    'steps': len(result_b.path) if result_b.path else 0,
    'cost': result_b.cost,
    'time': result_b.time_ms
})

comparison.append({
    'method': 'hierarchical',
    'steps': len(result_h.path) if result_h.path else 0,
    'cost': result_h.cost,
    'time': result_h.time_ms
})

print(f"{'Method':<20} {'Steps':<8} {'Cost':<10} {'Time(ms)':<10}")
print("-" * 50)
for c in comparison:
    print(f"{c['method']:<20} {c['steps']:<8} {c['cost']:<10.2f} {c['time']:<10.2f}")

print()
print("=" * 70)
print("Demo completed")
print("=" * 70)
