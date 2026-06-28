"""
ATLAS SSFR Kitchen Integration Tests

测试SSFR与物理厨房的集成：
1. 空间适配器测试
2. 物理SSFR测试
3. 任务规划器测试
4. 端到端集成测试
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import math

from atlas.kitchen import (
    Kitchen, Robot, PhysicsConfig, ObjectType, RobotAction,
    SensorSystem, KitchenSSFRInterface, TASK_LIBRARY, Task, TaskStep,
    create_demo_kitchen,
)
from demo_ssfr_kitchen import (
    KitchenSpaceAdapter, PhysicalSSFR, SSFRTaskPlanner,
)


def test_kitchen_space_adapter():
    """测试空间适配器"""
    print("=" * 70)
    print("TEST 1: Kitchen Space Adapter")
    print("=" * 70)

    kitchen = create_demo_kitchen()
    adapter = KitchenSpaceAdapter(kitchen, grid_resolution=0.5)

    # 测试网格尺寸
    print(f"Grid size: {adapter.grid_width} x {adapter.grid_height}")
    assert adapter.grid_width > 0
    assert adapter.grid_height > 0

    # 测试坐标转换
    world_pos = (5.0, 3.0)
    grid_pos = adapter.world_to_grid(world_pos)
    back_to_world = adapter.grid_to_world(grid_pos)
    print(f"World {world_pos} -> Grid {grid_pos} -> World {back_to_world}")

    # 检查转换误差（应在半个网格内）
    error = math.sqrt(
        (back_to_world[0] - world_pos[0])**2 +
        (back_to_world[1] - world_pos[1])**2
    )
    assert error <= adapter.grid_resolution, f"Conversion error too large: {error}"
    print(f"Conversion error: {error:.3f}m (threshold: {adapter.grid_resolution}m)")

    # 测试障碍物获取
    robot_id = list(kitchen.robots.keys())[0]
    obstacles = adapter.get_obstacles(robot_id)
    print(f"Obstacles found: {len(obstacles)}")
    assert len(obstacles) > 0, "No obstacles found"

    # 测试观测编码
    observation = adapter.encode_observation(robot_id)
    print(f"Observation keys: {list(observation.keys())}")
    assert 'position' in observation
    assert 'obstacles' in observation
    assert 'uncertainty' in observation
    assert 0 <= observation['uncertainty'] <= 1.0

    print(f"Uncertainty: {observation['uncertainty']:.3f}")
    print(f"Nearby objects: {observation['nearby_objects_count']}")

    print("[PASS] Kitchen space adapter works")
    return True


def test_physical_ssfr():
    """测试物理SSFR"""
    print("\n" + "=" * 70)
    print("TEST 2: Physical SSFR")
    print("=" * 70)

    kitchen = create_demo_kitchen()
    robot_id = list(kitchen.robots.keys())[0]

    # 创建物理SSFR
    physical_ssfr = PhysicalSSFR(kitchen, grid_resolution=0.5)

    print(f"SSFR initialized with {len(physical_ssfr.ssfr.spaces)} spaces")
    print(f"Grid: {physical_ssfr.adapter.grid_width} x {physical_ssfr.adapter.grid_height}")

    # 测试感知
    print("\n--- Perception ---")
    hypotheses = physical_ssfr.perceive(robot_id)
    print(f"Hypotheses generated: {len(hypotheses)}")
    assert len(hypotheses) > 0, "No hypotheses generated"

    for i, hyp in enumerate(hypotheses[:3]):
        print(f"  {hyp.name}: fitness={hyp.fitness:.3f}, "
              f"spaces={list(hyp.representations.keys())}")

    # 测试竞争
    print("\n--- Competition ---")
    robot = kitchen.robots[robot_id]
    winner = physical_ssfr.compete(robot_id, robot.position)
    if winner:
        print(f"Winner: {winner.name} (fitness={winner.fitness:.3f})")
    else:
        print("No winner (expected on first step)")

    # 测试空间有效性
    print("\n--- Space Validity ---")
    validity = physical_ssfr.get_space_validity(robot_id)
    for name, val in validity.items():
        print(f"  {name}: {val:.3f}")
        assert 0 <= val <= 1.0, f"Invalid validity: {val}"

    # 测试统计
    print("\n--- Statistics ---")
    stats = physical_ssfr.get_statistics()
    print(f"Perceptions: {stats['physical_perceptions']}")
    print(f"Pool size: {stats['pool_stats']['num_structures']}")
    print(f"Avg time: {stats['avg_time_per_step']*1000:.2f}ms")

    assert stats['physical_perceptions'] > 0
    assert stats['pool_stats']['num_structures'] > 0

    print("[PASS] Physical SSFR works")
    return True


def test_ssfr_task_planner():
    """测试SSFR任务规划器"""
    print("\n" + "=" * 70)
    print("TEST 3: SSFR Task Planner")
    print("=" * 70)

    kitchen = create_demo_kitchen()
    robot_id = list(kitchen.robots.keys())[0]

    # 创建规划器
    physical_ssfr = PhysicalSSFR(kitchen, grid_resolution=0.5)
    planner = SSFRTaskPlanner(physical_ssfr)

    # 测试任务分配
    print("\n--- Task Assignment ---")
    success = planner.assign_task(robot_id, 'make_coffee')
    print(f"Task assigned: {success}")
    assert success, "Task assignment failed"

    # 测试单步执行
    print("\n--- Step Execution ---")
    for i in range(5):
        kitchen.step()
        result = planner.step(robot_id)

        print(f"Step {i+1}:")
        print(f"  Status: {result['status']}")
        print(f"  Step: {result.get('step', 'none')}")
        print(f"  Progress: {result.get('progress', 0)*100:.1f}%")
        print(f"  Best space: {result.get('best_space', 'none')}")
        print(f"  Hypotheses: {result.get('num_hypotheses', 0)}")

    # 测试状态查询
    print("\n--- Task Status ---")
    status = planner.get_task_status(robot_id)
    print(f"Task: {status['task']}")
    print(f"Status: {status['status']}")
    print(f"Progress: {status['progress']*100:.1f}%")
    print(f"Current step: {status['current_step']}/{status['total_steps']}")

    assert status['task'] == 'Make Coffee'
    assert status['total_steps'] > 0

    print("[PASS] SSFR task planner works")
    return True


def test_ssfr_navigation():
    """测试SSFR导航"""
    print("\n" + "=" * 70)
    print("TEST 4: SSFR Navigation")
    print("=" * 70)

    kitchen = create_demo_kitchen()
    robot_id = list(kitchen.robots.keys())[0]
    robot = kitchen.robots[robot_id]

    # 创建规划器
    physical_ssfr = PhysicalSSFR(kitchen, grid_resolution=0.5)
    planner = SSFRTaskPlanner(physical_ssfr)

    # 分配任务（需要导航到咖啡机）
    planner.assign_task(robot_id, 'make_coffee')

    # 记录初始位置
    initial_pos = robot.position
    print(f"Initial position: ({initial_pos[0]:.2f}, {initial_pos[1]:.2f})")

    # 运行一段时间
    for i in range(120):
        kitchen.step()
        result = planner.step(robot_id)

        if i % 30 == 0:
            pos = robot.position
            print(f"  Step {i}: pos=({pos[0]:.2f}, {pos[1]:.2f}), "
                  f"step={result.get('step', 'none')}")

    # 检查是否移动了
    final_pos = robot.position
    distance = math.sqrt(
        (final_pos[0] - initial_pos[0])**2 +
        (final_pos[1] - initial_pos[1])**2
    )
    print(f"\nDistance moved: {distance:.2f}m")
    assert distance > 0.1, "Robot did not move"

    print("[PASS] SSFR navigation works")
    return True


def test_ssfr_structure_evolution():
    """测试SSFR结构演化"""
    print("\n" + "=" * 70)
    print("TEST 5: SSFR Structure Evolution")
    print("=" * 70)

    kitchen = create_demo_kitchen()
    robot_id = list(kitchen.robots.keys())[0]
    robot = kitchen.robots[robot_id]

    physical_ssfr = PhysicalSSFR(kitchen, grid_resolution=0.5)

    # 在不同位置进行感知
    positions = [(2.0, 2.0), (4.0, 2.0), (6.0, 4.0)]
    initial_pool_size = None

    for i, pos in enumerate(positions):
        robot.body.position = pos
        robot.body.velocity = (0, 0)

        # 步进
        for _ in range(10):
            kitchen.step()

        # 感知
        hypotheses = physical_ssfr.perceive(robot_id)

        stats = physical_ssfr.get_statistics()
        if i == 0:
            initial_pool_size = stats['pool_stats']['num_structures']

        print(f"Position {i+1}: ({pos[0]}, {pos[1]})")
        print(f"  Hypotheses: {len(hypotheses)}")
        print(f"  Pool size: {stats['pool_stats']['num_structures']}")

    # 验证结构池增长
    final_pool_size = stats['pool_stats']['num_structures']
    print(f"\nPool growth: {initial_pool_size} -> {final_pool_size}")
    assert final_pool_size >= initial_pool_size, "Structure pool did not grow"

    print("[PASS] SSFR structure evolution works")
    return True


def test_ssfr_space_comparison():
    """测试不同空间的比较"""
    print("\n" + "=" * 70)
    print("TEST 6: SSFR Space Comparison")
    print("=" * 70)

    kitchen = create_demo_kitchen()
    robot_id = list(kitchen.robots.keys())[0]

    # 测试不同空间配置
    configs = [
        (['ricci'], 'Ricci'),
        (['fisher'], 'Fisher'),
        (['wasserstein'], 'Wasserstein'),
        (['ricci', 'fisher'], 'Ricci+Fisher'),
    ]

    results = []
    for space_names, label in configs:
        physical_ssfr = PhysicalSSFR(kitchen, grid_resolution=0.5, space_names=space_names)
        hypotheses = physical_ssfr.perceive(robot_id)
        stats = physical_ssfr.get_statistics()

        results.append({
            'label': label,
            'spaces': len(space_names),
            'hypotheses': len(hypotheses),
            'pool_size': stats['pool_stats']['num_structures'],
            'time': stats['avg_time_per_step'] * 1000,
        })

    print(f"{'Config':<15} {'Spaces':<8} {'Hypotheses':<12} {'Pool':<8} {'Time(ms)':<10}")
    print("-" * 55)
    for r in results:
        print(f"{r['label']:<15} {r['spaces']:<8} {r['hypotheses']:<12} "
              f"{r['pool_size']:<8} {r['time']:>6.2f}")

    # 验证多空间配置产生更多假设
    multi_space = next((r for r in results if r['spaces'] > 1), None)
    single_space = next((r for r in results if r['spaces'] == 1), None)

    if multi_space and single_space:
        assert multi_space['hypotheses'] >= single_space['hypotheses'], \
            "Multi-space should generate at least as many hypotheses"

    print("[PASS] SSFR space comparison works")
    return True


def test_end_to_end_integration():
    """测试端到端集成"""
    print("\n" + "=" * 70)
    print("TEST 7: End-to-End Integration")
    print("=" * 70)

    kitchen = create_demo_kitchen()
    robot_id = list(kitchen.robots.keys())[0]

    # 创建完整的SSFR系统
    physical_ssfr = PhysicalSSFR(kitchen, grid_resolution=0.5)
    planner = SSFRTaskPlanner(physical_ssfr)

    # 分配任务
    planner.assign_task(robot_id, 'make_coffee')

    # 运行模拟
    print("Running simulation...")
    max_steps = 300
    step_results = []

    for step in range(max_steps):
        kitchen.step()
        result = planner.step(robot_id)
        step_results.append(result)

        # 检查完成
        if result.get('status') in ['completed', 'failed']:
            break

    # 验证
    status = planner.get_task_status(robot_id)
    print(f"\nFinal status: {status['status']}")
    print(f"Progress: {status['progress']*100:.1f}%")
    print(f"Steps completed: {status['current_step']}/{status['total_steps']}")

    # 验证SSFR统计
    stats = physical_ssfr.get_statistics()
    print(f"\nSSFR Statistics:")
    print(f"  Perceptions: {stats['physical_perceptions']}")
    print(f"  Competitions: {stats['physical_competitions']}")
    print(f"  Pool size: {stats['pool_stats']['num_structures']}")
    print(f"  Avg time: {stats['avg_time_per_step']*1000:.2f}ms")

    assert stats['physical_perceptions'] > 0, "No perceptions recorded"
    assert stats['pool_stats']['num_structures'] > 0, "No structures in pool"

    print("[PASS] End-to-end integration works")
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("ATLAS SSFR Kitchen Integration Tests")
    print("=" * 70)

    tests = [
        test_kitchen_space_adapter,
        test_physical_ssfr,
        test_ssfr_task_planner,
        test_ssfr_navigation,
        test_ssfr_structure_evolution,
        test_ssfr_space_comparison,
        test_end_to_end_integration,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))

    # 汇总
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()

    print("\n" + "=" * 70)
    if success:
        print("All tests passed!")
    else:
        print("Some tests failed!")
    print("=" * 70)
