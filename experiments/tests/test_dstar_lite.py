"""
Test: D* Lite Incremental Planning
测试 D* Lite 增量规划能力

场景: 动态障碍物环境
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
from atlas.core import AdaptiveNavigator
from atlas.core.registry import create_space


def create_maze_with_moving_obstacle():
    """创建带有移动障碍物的场景"""
    return {
        'start': (5, 10),
        'goal': (35, 10),
        # 初始障碍物（墙）
        'initial_obstacles': {(20, y) for y in range(3, 17) if y != 10},
        # 移动障碍物轨迹
        'moving_obstacle': [
            (25, 8),   # 时间步1: 在路径上出现
            (26, 9),   # 时间步2: 移动
            (27, 10),  # 时间步3: 阻挡原路径
            (28, 10),  # 时间步4: 继续
        ]
    }


def test_static_comparison():
    """对比: 静态环境中的 A* vs D* Lite"""
    print("=" * 70)
    print("Test 1: 静态环境对比")
    print("=" * 70)

    scenario = create_maze_with_moving_obstacle()

    # 创建 Ricci 空间
    space = create_space("ricci", width=40, height=20, curvature_scale=1.0)

    # 应用初始观测
    space.update_from_observation(
        (5, 10),
        {'obstacles': list(scenario['initial_obstacles'])}
    )

    # GeodesicSolver (A*)
    from atlas.core import GeodesicSolver
    solver_a = GeodesicSolver(space)
    result_a = solver_a.solve(
        scenario['start'], scenario['goal'], scenario['initial_obstacles']
    )

    # D* Lite
    navigator = AdaptiveNavigator(space)
    success = navigator.initialize(
        scenario['start'], scenario['goal'], scenario['initial_obstacles']
    )

    print(f"A*       : Success={result_a.success}, Steps={len(result_a.path) if result_a.path else 0}, Time={result_a.time_ms:.2f}ms")
    print(f"D* Lite  : Success={success}, Path length={len(navigator.get_path())}")
    print()


def test_dynamic_replanning():
    """测试: 动态重新规划能力"""
    print("=" * 70)
    print("Test 2: 动态障碍物重新规划")
    print("=" * 70)

    scenario = create_maze_with_moving_obstacle()

    # 创建空间
    space = create_space("ricci", width=40, height=20, curvature_scale=1.0)

    # 初始化导航器
    navigator = AdaptiveNavigator(space, replan_threshold=2.0)
    success = navigator.initialize(
        scenario['start'], scenario['goal'], scenario['initial_obstacles']
    )

    if not success:
        print("Initial planning failed!")
        return

    print(f"Initial path: {len(navigator.get_path())} steps")
    print(f"Path: {navigator.get_path()[:5]}...{navigator.get_path()[-3:]}")
    print()

    # 模拟运行
    current_pos = scenario['start']
    step = 0
    max_steps = 50

    path_history = [current_pos]

    while current_pos != scenario['goal'] and step < max_steps:
        step += 1

        # 模拟障碍物移动
        new_obs = {}
        if step <= len(scenario['moving_obstacle']):
            moving_obs = scenario['moving_obstacle'][step - 1]
            new_obs = {'obstacles': [moving_obs]}
            print(f"Step {step}: Moving obstacle appears at {moving_obs}")

        # 导航步进
        next_pos = navigator.step(current_pos, new_obs)

        if next_pos is None:
            print(f"Step {step}: Goal reached!")
            break

        current_pos = next_pos
        path_history.append(current_pos)

        # 每5步显示状态
        if step % 5 == 0 or new_obs:
            print(f"  Position: {current_pos}, Path devs: {navigator.stats['path_deviations']}")

    print()
    print(f"Total steps: {step}")
    print(f"Replan count: {navigator.stats['replan_count']}")
    print(f"Path deviations: {navigator.stats['path_deviations']}")
    print(f"Final position: {current_pos}")
    print()


def test_euclidean_vs_ricci_dynamic():
    """对比: 不同空间在动态环境中的适应性"""
    print("=" * 70)
    print("Test 3: 空间类型对比 (动态环境)")
    print("=" * 70)

    scenario = create_maze_with_moving_obstacle()

    results = []

    for space_name in ["euclidean", "ricci", "conformal"]:
        try:
            space = create_space(space_name, width=40, height=20)

            navigator = AdaptiveNavigator(space)
            success = navigator.initialize(
                scenario['start'], scenario['goal'], scenario['initial_obstacles']
            )

            if not success:
                continue

            # 模拟10步
            current_pos = scenario['start']
            for step in range(1, 11):
                new_obs = {}
                if step == 5:  # 第5步添加新障碍物
                    new_obs = {'obstacles': [(15, 10)]}

                next_pos = navigator.step(current_pos, new_obs)
                if next_pos is None:
                    break
                current_pos = next_pos

            results.append({
                'space': space_name,
                'steps': 10,
                'replan': navigator.stats['replan_count'],
                'final_pos': current_pos
            })

            print(f"{space_name:12s}: replans={navigator.stats['replan_count']}, "
                  f"final_pos={current_pos}")

        except Exception as e:
            print(f"{space_name:12s}: Error - {e}")

    print()


def main():
    print()
    print("=" * 70)
    print("ATLAS: D* Lite Incremental Planning Tests")
    print("=" * 70)
    print()

    try:
        test_static_comparison()
        test_dynamic_replanning()
        test_euclidean_vs_ricci_dynamic()
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 70)
    print("Tests completed")
    print("=" * 70)


if __name__ == "__main__":
    main()
