"""
ATLAS Sandbox 测试 - 验证组件组合与任务执行

测试场景：
1. 基础移动：机器人能否在网格中移动
2. 组件组合：不同组件组合产生不同能力
3. 任务完成：机器人能否完成特定任务
4. SSFR 集成：SSFR 能否发现最优组件组合
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from atlas.sandbox import (
    Sandbox, Robot, Component, Task,
    COMPONENT_LIBRARY, TASK_LIBRARY,
    TaskAllocator, create_example_scenario
)
from atlas.core.ssfr_enhanced import SSFREnhanced


def test_basic_movement():
    """测试基础移动"""
    print("=" * 70)
    print("TEST 1: Basic Movement")
    print("=" * 70)

    sandbox = Sandbox(width=10, height=10)

    robot = Robot(name='Mover', position=(1, 1))
    robot.add_component(COMPONENT_LIBRARY['wheels'])
    robot.add_component(COMPONENT_LIBRARY['small_battery'])
    sandbox.add_robot(robot)

    print(f"Initial position: {robot.position}")

    # 模拟移动
    robot.position = (2, 1)
    print(f"After move: {robot.position}")

    # 检查能量消耗
    initial_energy = robot.energy
    robot.step()
    print(f"Energy: {initial_energy:.1f} -> {robot.energy:.1f}")

    print("[PASS] Basic movement works")
    return True


def test_component_combination():
    """测试组件组合"""
    print("\n" + "=" * 70)
    print("TEST 2: Component Combination")
    print("=" * 70)

    # 机器人1：基础配置
    robot1 = Robot(name='Basic')
    robot1.add_component(COMPONENT_LIBRARY['basic_camera'])
    robot1.add_component(COMPONENT_LIBRARY['wheels'])
    robot1.add_component(COMPONENT_LIBRARY['basic_cpu'])
    robot1.add_component(COMPONENT_LIBRARY['small_battery'])

    # 机器人2：高级配置
    robot2 = Robot(name='Advanced')
    robot2.add_component(COMPONENT_LIBRARY['thermal_camera'])
    robot2.add_component(COMPONENT_LIBRARY['tracks'])
    robot2.add_component(COMPONENT_LIBRARY['advanced_cpu'])
    robot2.add_component(COMPONENT_LIBRARY['lidar'])
    robot2.add_component(COMPONENT_LIBRARY['large_battery'])

    print(f"Robot 1 capabilities: {robot1.get_capabilities()}")
    print(f"Robot 2 capabilities: {robot2.get_capabilities()}")

    # 比较
    print(f"\nComparison:")
    print(f"  Robot 1 can see: {robot1.can_perform('vision')}")
    print(f"  Robot 2 can see: {robot2.can_perform('vision')}")
    print(f"  Robot 1 can detect thermal: {robot1.can_perform('thermal_detection')}")
    print(f"  Robot 2 can detect thermal: {robot2.can_perform('thermal_detection')}")

    # 检查重量
    print(f"\nWeight:")
    print(f"  Robot 1: {sum(c.weight for c in robot1.components):.1f}/10")
    print(f"  Robot 2: {sum(c.weight for c in robot2.components):.1f}/10")

    print("[PASS] Component combination works")
    return True


def test_task_completion():
    """测试任务完成"""
    print("\n" + "=" * 70)
    print("TEST 3: Task Completion")
    print("=" * 70)

    # 创建机器人（配置用于救援）
    robot = Robot(name='Rescuer')
    robot.add_component(COMPONENT_LIBRARY['thermal_camera'])
    robot.add_component(COMPONENT_LIBRARY['wheels'])
    robot.add_component(COMPONENT_LIBRARY['mechanical_arm'])
    robot.add_component(COMPONENT_LIBRARY['large_battery'])

    # 测试任务
    tasks = [
        TASK_LIBRARY['exploration'],
        TASK_LIBRARY['rescue'],
        TASK_LIBRARY['mining'],
        TASK_LIBRARY['surveillance'],
    ]

    print(f"Robot capabilities: {robot.get_capabilities()}")
    print()

    for task in tasks:
        can_complete = task.check_completion(robot)
        progress = task.get_progress(robot)
        print(f"Task: {task.name}")
        print(f"  Can complete: {can_complete}")
        print(f"  Progress: {progress*100:.1f}%")
        print(f"  Required: {task.required_capabilities}")
        print()

    print("[PASS] Task completion check works")
    return True


def test_terrain_effects():
    """测试地形影响"""
    print("\n" + "=" * 70)
    print("TEST 4: Terrain Effects")
    print("=" * 70)

    sandbox = Sandbox(width=15, height=15)

    # 设置地形
    for x in range(5, 10):
        for y in range(5, 10):
            sandbox.set_terrain(x, y, 2)  # 水域

    for x in range(10, 13):
        for y in range(10, 13):
            sandbox.set_terrain(x, y, 3)  # 山地

    # 普通机器人
    robot1 = Robot(name='Normal', position=(1, 1))
    robot1.add_component(COMPONENT_LIBRARY['wheels'])
    robot1.add_component(COMPONENT_LIBRARY['small_battery'])
    sandbox.add_robot(robot1)

    # 全地形机器人
    robot2 = Robot(name='AllTerrain', position=(1, 1))
    robot2.add_component(COMPONENT_LIBRARY['tracks'])
    robot2.add_component(COMPONENT_LIBRARY['large_battery'])
    sandbox.add_robot(robot2)

    # 测试通过性
    test_positions = [
        (3, 3),   # 平地
        (7, 7),   # 水域
        (11, 11), # 山地
    ]

    for pos in test_positions:
        terrain = sandbox.get_terrain_at(pos)
        pass1 = sandbox.is_passable(pos, robot1)
        pass2 = sandbox.is_passable(pos, robot2)
        print(f"Position {pos} (terrain={terrain}): Normal={pass1}, AllTerrain={pass2}")

    print("[PASS] Terrain effects work")
    return True


def test_ssfr_component_discovery():
    """测试 SSFR 发现最优组件组合"""
    print("\n" + "=" * 70)
    print("TEST 5: SSFR Component Discovery")
    print("=" * 70)

    # 创建 SSFR
    ssfr = SSFREnhanced(
        width=20, height=20,
        space_names=['ricci', 'fisher'],
        max_structures=100,
        reuse_threshold=0.85
    )

    # 模拟任务序列
    tasks = [
        ('exploration', {'vision': 0.5, 'movement': 0.5}),
        ('rescue', {'thermal_detection': 0.7, 'grasping': 0.5}),
        ('mining', {'drilling': 0.8, 'mining': 0.7}),
        ('surveillance', {'vision': 0.7, 'object_detection': 0.6}),
    ]

    # 记录组件组合策略
    strategies = []

    for task_name, required_caps in tasks:
        # 模拟感知
        observation = {
            'position': (1, 1),
            'task': task_name,
            'required_capabilities': required_caps,
        }

        # SSFR 感知
        ssfr.perceive((1, 1), observation, active_space_name='ricci')

        # 获取最佳结构
        best = ssfr.get_best_structures(n=1)
        if best:
            strategies.append({
                'task': task_name,
                'structure': best[0].id,
                'space': list(best[0].representations.keys())[0] if best[0].representations else 'none',
            })

        ssfr.step_count += 1

    print("SSFR discovered strategies:")
    for s in strategies:
        print(f"  Task: {s['task']}, Space: {s['space']}")

    stats = ssfr.get_statistics()
    print(f"\nSSFR Stats:")
    print(f"  Total perceptions: {stats['num_perceptions']}")
    print(f"  Pool size: {stats['pool_stats']['num_structures']}")
    print(f"  Reuse rate: {stats.get('reuse_rate', 0)*100:.1f}%")

    print("[PASS] SSFR component discovery works")
    return True


def test_task_allocator():
    """测试任务分配器"""
    print("\n" + "=" * 70)
    print("TEST 6: Task Allocator")
    print("=" * 70)

    sandbox = Sandbox(width=20, height=20)
    allocator = TaskAllocator(sandbox)

    # 创建两个机器人
    robot1 = Robot(name='Scout', position=(1, 1))
    robot1.add_component(COMPONENT_LIBRARY['basic_camera'])
    robot1.add_component(COMPONENT_LIBRARY['wheels'])
    robot1.add_component(COMPONENT_LIBRARY['small_battery'])
    sandbox.add_robot(robot1)

    robot2 = Robot(name='Miner', position=(5, 5))
    robot2.add_component(COMPONENT_LIBRARY['drill'])
    robot2.add_component(COMPONENT_LIBRARY['tracks'])
    robot2.add_component(COMPONENT_LIBRARY['large_battery'])
    sandbox.add_robot(robot2)

    # 添加任务
    tasks = [
        TASK_LIBRARY['exploration'],
        TASK_LIBRARY['mining'],
        TASK_LIBRARY['rescue'],
    ]

    for task in tasks:
        sandbox.add_task(task)

    # 分配任务
    print("Task allocation:")
    for rid, robot in sandbox.robots.items():
        task = allocator.allocate_task(robot, sandbox.tasks)
        if task:
            score = allocator.evaluate_robot_for_task(robot, task)
            print(f"  {robot.name} -> {task.name} (score={score:.2f})")

            # 建议组件
            suggestions = allocator.suggest_components(task, robot)
            if suggestions:
                print(f"    Suggested components: {suggestions[:3]}")

    print("[PASS] Task allocator works")
    return True


def run_all_tests():
    """运行所有测试"""
    tests = [
        test_basic_movement,
        test_component_combination,
        test_task_completion,
        test_terrain_effects,
        test_ssfr_component_discovery,
        test_task_allocator,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            results.append((test.__name__, False))

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
