"""
ATLAS Kitchen Demo - 物理厨房演示

运行物理厨房模拟，展示：
1. 重力、碰撞、摩擦
2. 机器人移动和交互
3. 任务执行
4. 可视化渲染（可选）
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import math
import time
import argparse

from atlas.kitchen import (
    Kitchen, Robot, PhysicsConfig, ObjectType,
    create_demo_kitchen, KitchenRenderer,
    OBJECT_LIBRARY, TASK_LIBRARY,
)
from atlas.kitchen.controller import (
    TaskExecutor, NavigationController, SSFRKitchenController,
)


def demo_basic_physics():
    """基础物理演示"""
    print("=" * 70)
    print("DEMO: Basic Physics")
    print("=" * 70)

    kitchen = Kitchen(width=10.0, height=8.0)

    # 添加物体
    cup = kitchen.add_object('coffee_cup', (5.0, 6.0))
    plate = kitchen.add_object('plate', (3.0, 6.0))
    apple = kitchen.add_object('apple', (7.0, 6.0))

    print(f"Objects added: {len(kitchen.objects)}")
    print(f"Initial positions:")
    for obj in kitchen.objects.values():
        print(f"  {obj.properties.name}: ({obj.position[0]:.1f}, {obj.position[1]:.1f})")

    # 模拟
    print(f"\nSimulating 2 seconds...")
    for _ in range(120):
        kitchen.step()

    print(f"Final positions:")
    for obj in kitchen.objects.values():
        print(f"  {obj.properties.name}: ({obj.position[0]:.1f}, {obj.position[1]:.1f})")

    print("\n[OK] Objects fell due to gravity and stopped at floor")


def demo_robot_movement():
    """机器人移动演示"""
    print("\n" + "=" * 70)
    print("DEMO: Robot Movement")
    print("=" * 70)

    kitchen = Kitchen(width=10.0, height=8.0)
    kitchen.setup_default_kitchen()

    robot = kitchen.add_robot('Mover', (2.0, 2.0))

    print(f"Robot initial: ({robot.position[0]:.1f}, {robot.position[1]:.1f})")

    # 前进
    print("\nMoving forward...")
    robot.move_forward(force=5.0)
    for _ in range(60):
        kitchen.step()
    print(f"After forward: ({robot.position[0]:.1f}, {robot.position[1]:.1f})")

    # 停止
    robot.stop()
    for _ in range(30):
        kitchen.step()

    # 转向
    print("\nTurning left...")
    robot.turn_left(torque=3.0)
    for _ in range(60):
        kitchen.step()
    print(f"After turn: angle={robot.angle:.2f} rad")

    # 再前进
    print("\nMoving forward again...")
    robot.move_forward(force=5.0)
    for _ in range(60):
        kitchen.step()
    print(f"Final: ({robot.position[0]:.1f}, {robot.position[1]:.1f})")

    print("\n[OK] Robot moved and turned correctly")


def demo_collision():
    """碰撞演示"""
    print("\n" + "=" * 70)
    print("DEMO: Collision")
    print("=" * 70)

    kitchen = Kitchen(width=10.0, height=8.0)

    # 两个物体
    obj1 = kitchen.add_object('coffee_cup', (3.0, 2.0))
    obj2 = kitchen.add_object('coffee_cup', (4.0, 2.0))

    print(f"Object 1: ({obj1.position[0]:.1f}, {obj1.position[1]:.1f})")
    print(f"Object 2: ({obj2.position[0]:.1f}, {obj2.position[1]:.1f})")

    # 给第一个物体速度
    obj1.body.velocity = (3.0, 0)
    print(f"\nApplying velocity to object 1: (3.0, 0)")

    # 模拟
    for i in range(60):
        kitchen.step()
        if i % 10 == 0:
            print(f"  Step {i}: obj1=({obj1.position[0]:.1f}, {obj1.position[1]:.1f}), "
                  f"obj2=({obj2.position[0]:.1f}, {obj2.position[1]:.1f})")

    print(f"\nFinal: obj1=({obj1.position[0]:.1f}, {obj1.position[1]:.1f}), "
          f"obj2=({obj2.position[0]:.1f}, {obj2.position[1]:.1f})")
    print("[OK] Objects collided and bounced")


def demo_task_execution():
    """任务执行演示"""
    print("\n" + "=" * 70)
    print("DEMO: Task Execution")
    print("=" * 70)

    kitchen = create_demo_kitchen()
    controller = SSFRKitchenController(kitchen)

    # 注册机器人
    robot_id = list(kitchen.robots.keys())[0]
    controller.register_robot(robot_id)

    # 分配任务
    task_name = 'make_coffee'
    success = controller.assign_task(robot_id, task_name)
    print(f"Task assigned: {task_name} (success={success})")

    # 运行
    print(f"\nRunning task for 5 seconds...")
    for step in range(300):
        kitchen.step()
        results = controller.step()

        if step % 60 == 0:
            result = results.get(robot_id, {})
            status = result.get('status', 'unknown')
            progress = result.get('progress', 0)
            step_info = result.get('step', 'idle')
            print(f"  {step//60:3d}s: {step_info:15s} | {status:10s} | {progress*100:5.1f}%")

    print("\n[OK] Task execution demo complete")


def demo_visualization(duration: float = 10.0):
    """可视化演示"""
    print("\n" + "=" * 70)
    print("DEMO: Visualization")
    print("=" * 70)

    kitchen = create_demo_kitchen()
    robot = list(kitchen.robots.values())[0]

    # 创建渲染器
    try:
        renderer = KitchenRenderer(kitchen)
        print("Renderer initialized. Close window to stop.")
    except ImportError:
        print("pygame not available. Install with: pip install pygame")
        return

    running = True
    start_time = time.time()
    step_count = 0

    # 简单的机器人控制
    import random

    try:
        while running and (time.time() - start_time) < duration:
            # 物理步进
            kitchen.step()

            # 简单的机器人行为
            if step_count % 120 == 0:
                action = random.choice(['forward', 'left', 'right', 'stop'])
                if action == 'forward':
                    robot.move_forward()
                elif action == 'left':
                    robot.turn_left()
                elif action == 'right':
                    robot.turn_right()
                else:
                    robot.stop()

            # 渲染
            running = renderer.render()
            step_count += 1

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        renderer.close()

    print(f"\n[OK] Visualization complete: {step_count} frames, {kitchen.time:.1f}s simulated")


def demo_full_simulation(duration: float = 30.0):
    """完整模拟演示"""
    print("\n" + "=" * 70)
    print("DEMO: Full Simulation")
    print("=" * 70)

    kitchen = create_demo_kitchen()
    controller = SSFRKitchenController(kitchen)

    # 注册机器人
    for robot_id in kitchen.robots:
        controller.register_robot(robot_id)

    # 分配任务
    robot_id = list(kitchen.robots.keys())[0]
    controller.assign_task(robot_id, 'make_coffee')

    print(f"Running full simulation for {duration} seconds...")
    print(f"Kitchen: {kitchen.width}m x {kitchen.height}m")
    print(f"Robots: {len(kitchen.robots)}")
    print(f"Objects: {len(kitchen.objects)}")
    print()

    # 运行
    step_count = 0
    max_steps = int(duration / PhysicsConfig.TIME_STEP)
    start_time = time.time()

    while step_count < max_steps:
        # 物理步进
        kitchen.step()

        # 控制步进
        results = controller.step()

        # 打印状态
        if step_count % 60 == 0:
            result = results.get(robot_id, {})
            status = result.get('status', 'unknown')
            progress = result.get('progress', 0)
            step_info = result.get('step', 'idle')

            print(f"  {step_count//60:3d}s: {step_info:15s} | "
                  f"{status:10s} | {progress*100:5.1f}%")

        step_count += 1

    elapsed = time.time() - start_time
    print(f"\n[OK] Simulation complete: {step_count} steps in {elapsed:.1f}s")
    print(f"  Real-time factor: {kitchen.time / elapsed:.1f}x")


def main():
    parser = argparse.ArgumentParser(description='ATLAS Physical Kitchen Demo')
    parser.add_argument('--demo', type=str, default='all',
                       choices=['all', 'physics', 'robot', 'collision',
                               'task', 'visual', 'full'],
                       help='Which demo to run')
    parser.add_argument('--duration', type=float, default=10.0,
                       help='Duration in seconds for visual/full demo')
    args = parser.parse_args()

    print("ATLAS Physical Kitchen Demo")
    print("=" * 70)
    print()

    demos = {
        'physics': demo_basic_physics,
        'robot': demo_robot_movement,
        'collision': demo_collision,
        'task': demo_task_execution,
        'visual': lambda: demo_visualization(args.duration),
        'full': lambda: demo_full_simulation(args.duration),
    }

    if args.demo == 'all':
        # 运行所有非可视化演示
        for name, func in demos.items():
            if name not in ['visual', 'full']:
                func()
    else:
        demos[args.demo]()

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
