"""
ATLAS Physical Kitchen 测试

验证物理厨房沙盒的所有基础设施：
1. 物理环境（重力、碰撞、摩擦）
2. 机器人（移动、转向、机械臂）
3. 物体系统（添加、移除、交互）
4. 传感器（射线投射）
5. 任务系统（动作序列）
6. SSFR 集成接口
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import math
import time

from atlas.kitchen import (
    Kitchen, Robot, PhysicsEntity, ObjectProperties,
    ObjectType, MaterialType, RobotAction,
    PhysicsConfig, OBJECT_LIBRARY, TASK_LIBRARY,
    SensorSystem, KitchenSSFRInterface,
    create_demo_kitchen, run_simulation,
)


def test_physics_environment():
    """测试物理环境"""
    print("=" * 70)
    print("TEST 1: Physics Environment")
    print("=" * 70)

    kitchen = Kitchen(width=10.0, height=8.0)

    # 检查重力
    print(f"Gravity: {kitchen.space.gravity}")
    assert kitchen.space.gravity == PhysicsConfig.GRAVITY, "Gravity not set correctly"

    # 检查墙壁
    print(f"Static bodies: {len(kitchen.space.static_body.shapes)}")

    # 添加物体并测试重力
    cup = kitchen.add_object('coffee_cup', (5.0, 6.0))
    assert cup is not None, "Failed to add cup"
    print(f"Cup added at: {cup.position}")

    initial_y = cup.position[1]

    # 模拟1秒
    for _ in range(60):
        kitchen.step()

    final_y = cup.position[1]
    print(f"Cup fell from {initial_y:.2f} to {final_y:.2f}")
    assert final_y < initial_y, "Object did not fall (gravity not working)"

    # 检查碰撞（物体应该停在地板上）
    assert final_y > 0.1, "Object fell through floor"

    print("[PASS] Physics environment works")
    return True


def test_robot_movement():
    """测试机器人移动"""
    print("\n" + "=" * 70)
    print("TEST 2: Robot Movement")
    print("=" * 70)

    kitchen = Kitchen(width=10.0, height=8.0)
    kitchen.setup_default_kitchen()

    robot = kitchen.add_robot('TestBot', (5.0, 2.0))
    print(f"Robot initial position: {robot.position}")
    print(f"Robot initial angle: {robot.angle:.2f} rad")

    # 测试前进
    initial_pos = robot.position
    robot.move_forward(force=5.0)
    for _ in range(30):
        kitchen.step()

    new_pos = robot.position
    distance = math.sqrt((new_pos[0] - initial_pos[0])**2 +
                         (new_pos[1] - initial_pos[1])**2)
    print(f"Moved distance: {distance:.2f}m")
    assert distance > 0.1, "Robot did not move forward"

    # 测试转向
    initial_angle = robot.angle
    robot.turn_left(torque=2.0)
    for _ in range(30):
        kitchen.step()

    angle_diff = abs(robot.angle - initial_angle)
    print(f"Turned angle: {angle_diff:.2f} rad")
    assert angle_diff > 0.1, "Robot did not turn"

    # 测试停止
    robot.stop()
    vel_before = math.sqrt(robot.body.velocity.x**2 + robot.body.velocity.y**2)
    for _ in range(10):
        kitchen.step()
    vel_after = math.sqrt(robot.body.velocity.x**2 + robot.body.velocity.y**2)
    print(f"Velocity before stop: {vel_before:.2f}, after: {vel_after:.2f}")

    # 测试最大速度限制
    robot.move_forward(force=20.0)
    for _ in range(60):
        kitchen.step()
    speed = math.sqrt(robot.body.velocity.x**2 + robot.body.velocity.y**2)
    print(f"Max speed: {speed:.2f} m/s (limit: {PhysicsConfig.ROBOT_MAX_SPEED})")
    assert speed <= PhysicsConfig.ROBOT_MAX_SPEED * 1.1, "Robot exceeded max speed"

    print("[PASS] Robot movement works")
    return True


def test_collision():
    """测试碰撞"""
    print("\n" + "=" * 70)
    print("TEST 3: Collision Detection")
    print("=" * 70)

    kitchen = Kitchen(width=10.0, height=8.0)

    # 添加两个物体
    obj1 = kitchen.add_object('coffee_cup', (4.0, 2.0))
    obj2 = kitchen.add_object('coffee_cup', (4.5, 2.0))

    initial_dist = math.sqrt((obj1.position[0] - obj2.position[0])**2 +
                            (obj1.position[1] - obj2.position[1])**2)
    print(f"Initial distance: {initial_dist:.2f}m")

    # 给第一个物体一个朝向第二个物体的速度
    obj1.body.velocity = (2.0, 0)

    # 模拟
    for _ in range(60):
        kitchen.step()

    final_dist = math.sqrt((obj1.position[0] - obj2.position[0])**2 +
                           (obj1.position[1] - obj2.position[1])**2)
    print(f"Final distance: {final_dist:.2f}m")

    # 碰撞后距离应该不等于初始距离（发生了碰撞）
    assert final_dist != initial_dist, "Objects did not collide (distance unchanged)"
    # 物体应该还在厨房内
    assert 0 < obj1.position[0] < 10, "Object 1 went out of bounds"
    assert 0 < obj2.position[0] < 10, "Object 2 went out of bounds"

    print("[PASS] Collision detection works")
    return True


def test_friction():
    """测试摩擦"""
    print("\n" + "=" * 70)
    print("TEST 4: Friction")
    print("=" * 70)

    kitchen = Kitchen(width=10.0, height=8.0)

    # 添加物体并给一个初始速度
    obj = kitchen.add_object('plate', (5.0, 2.0))
    obj.body.velocity = (5.0, 0)

    initial_speed = math.sqrt(obj.body.velocity.x**2 + obj.body.velocity.y**2)
    print(f"Initial speed: {initial_speed:.2f} m/s")

    # 模拟
    for _ in range(120):
        kitchen.step()

    final_speed = math.sqrt(obj.body.velocity.x**2 + obj.body.velocity.y**2)
    print(f"Final speed: {final_speed:.2f} m/s")

    # 摩擦应该使物体减速
    assert final_speed < initial_speed, "Friction not working"

    print("[PASS] Friction works")
    return True


def test_sensor_system():
    """测试传感器系统"""
    print("\n" + "=" * 70)
    print("TEST 5: Sensor System")
    print("=" * 70)

    kitchen = Kitchen(width=10.0, height=8.0)
    kitchen.setup_default_kitchen()

    robot = kitchen.add_robot('SensorBot', (5.0, 2.0))
    sensor = SensorSystem(robot, kitchen.space)

    # 射线投射
    results = sensor.raycast()
    print(f"Raycast detected {len(results)} objects")
    assert len(results) > 0, "Sensor did not detect any objects"

    # 附近物体
    all_entities = list(kitchen.entities.values())
    nearby = sensor.get_nearby_objects(all_entities, max_distance=5.0)
    print(f"Nearby objects: {len(nearby)}")
    assert len(nearby) > 0, "Sensor did not find nearby objects"

    for obj, dist in nearby[:5]:
        print(f"  - {obj.properties.name} at {dist:.2f}m")

    print("[PASS] Sensor system works")
    return True


def test_task_system():
    """测试任务系统"""
    print("\n" + "=" * 70)
    print("TEST 6: Task System")
    print("=" * 70)

    task = TASK_LIBRARY['make_coffee']
    print(f"Task: {task.name}")
    print(f"Description: {task.description}")
    print(f"Steps: {len(task.steps)}")

    # 检查步骤
    for i, step in enumerate(task.steps):
        print(f"  Step {i+1}: {step.action} -> {step.target}")

    # 测试进度
    assert task.get_progress() == 0.0, "Initial progress should be 0"

    # 完成步骤
    for _ in range(len(task.steps)):
        task.complete_step()

    assert task.completed, "Task should be completed"
    assert task.get_progress() == 1.0, "Final progress should be 1.0"

    print(f"Progress: {task.get_progress()*100:.0f}%")

    print("[PASS] Task system works")
    return True


def test_ssfr_interface():
    """测试 SSFR 集成接口"""
    print("\n" + "=" * 70)
    print("TEST 7: SSFR Interface")
    print("=" * 70)

    kitchen = Kitchen(width=10.0, height=8.0)
    kitchen.setup_default_kitchen()
    robot = kitchen.add_robot('SSFRBot', (5.0, 2.0))

    interface = KitchenSSFRInterface(kitchen)

    # 获取观察
    obs = interface.get_observation(robot.entity.id)
    print(f"Observation keys: {list(obs.keys())}")
    assert 'position' in obs, "Observation missing position"
    assert 'nearby_objects' in obs, "Observation missing nearby_objects"

    print(f"Robot position: {obs['position']}")
    print(f"Nearby objects: {len(obs['nearby_objects'])}")

    for obj in obs['nearby_objects'][:3]:
        print(f"  - {obj['name']} ({obj['type']}) at {obj['distance']:.2f}m")

    # 编码状态
    state = interface.encode_state_for_ssfr()
    print(f"State keys: {list(state.keys())}")
    assert 'robots' in state, "State missing robots"
    assert 'objects' in state, "State missing objects"

    print("[PASS] SSFR interface works")
    return True


def test_kitchen_layout():
    """测试厨房布局"""
    print("\n" + "=" * 70)
    print("TEST 8: Kitchen Layout")
    print("=" * 70)

    kitchen = create_demo_kitchen()

    print(f"Kitchen size: {kitchen.width}m x {kitchen.height}m")
    print(f"Robots: {len(kitchen.robots)}")
    print(f"Objects: {len(kitchen.objects)}")

    # 检查家具
    furniture = [o for o in kitchen.objects.values()
                 if o.properties.obj_type == ObjectType.FURNITURE]
    appliances = [o for o in kitchen.objects.values()
                  if o.properties.obj_type == ObjectType.APPLIANCE]
    items = [o for o in kitchen.objects.values()
             if o.properties.obj_type in [ObjectType.CUP, ObjectType.PLATE,
                                         ObjectType.INGREDIENT, ObjectType.UTENSIL]]

    print(f"Furniture: {len(furniture)}")
    print(f"Appliances: {len(appliances)}")
    print(f"Items: {len(items)}")

    for obj in kitchen.objects.values():
        print(f"  - {obj.properties.name} at ({obj.position[0]:.1f}, {obj.position[1]:.1f})")

    assert len(kitchen.robots) > 0, "No robots in kitchen"
    assert len(kitchen.objects) > 0, "No objects in kitchen"

    print("[PASS] Kitchen layout works")
    return True


def test_mass_and_inertia():
    """测试质量和惯性"""
    print("\n" + "=" * 70)
    print("TEST 9: Mass and Inertia")
    print("=" * 70)

    kitchen = Kitchen(width=10.0, height=8.0)

    # 不同质量的物体
    light_obj = kitchen.add_object('spoon', (3.0, 2.0))
    heavy_obj = kitchen.add_object('coffee_machine', (7.0, 2.0))

    print(f"Light object mass: {light_obj.body.mass:.2f} kg")
    print(f"Heavy object mass: {heavy_obj.body.mass:.2f} kg")

    # 给相同力，观察加速度差异
    force = 5.0
    light_obj.body.apply_force_at_local_point((force, 0), (0, 0))
    heavy_obj.body.apply_force_at_local_point((force, 0), (0, 0))

    for _ in range(10):
        kitchen.step()

    light_speed = math.sqrt(light_obj.body.velocity.x**2 + light_obj.body.velocity.y**2)
    heavy_speed = math.sqrt(heavy_obj.body.velocity.x**2 + heavy_obj.body.velocity.y**2)

    print(f"Light object speed: {light_speed:.2f} m/s")
    print(f"Heavy object speed: {heavy_speed:.2f} m/s")

    # 轻物体应该加速更快
    assert light_speed > heavy_speed, "Lighter object should accelerate faster"

    print("[PASS] Mass and inertia work correctly")
    return True


def test_robot_arm():
    """测试机械臂"""
    print("\n" + "=" * 70)
    print("TEST 10: Robot Arm")
    print("=" * 70)

    kitchen = Kitchen(width=10.0, height=8.0)
    robot = kitchen.add_robot('ArmBot', (5.0, 2.0))

    # 测试机械臂位置
    arm = robot.arm
    print(f"Arm initial angle: {arm.arm_angle:.2f}")
    print(f"Arm initial extension: {arm.arm_extension:.2f}")

    grip_pos = arm.get_grip_position()
    print(f"Grip position: ({grip_pos[0]:.2f}, {grip_pos[1]:.2f})")

    # 设置角度
    arm.set_arm_angle(math.pi / 4)
    print(f"Arm angle after set: {arm.arm_angle:.2f}")
    assert abs(arm.arm_angle - math.pi/4) < 0.01, "Arm angle not set correctly"

    # 设置伸展
    arm.set_extension(0.8)
    print(f"Arm extension after set: {arm.arm_extension:.2f}")
    assert abs(arm.arm_extension - 0.8) < 0.01, "Arm extension not set correctly"

    # 检查限制
    arm.set_arm_angle(10.0)  # 超出范围
    assert arm.arm_angle <= math.pi/2, "Arm angle not clamped"

    arm.set_extension(2.0)  # 超出范围
    assert arm.arm_extension <= 1.0, "Arm extension not clamped"

    print("[PASS] Robot arm works")
    return True


def run_all_tests():
    """运行所有测试"""
    tests = [
        test_physics_environment,
        test_robot_movement,
        test_collision,
        test_friction,
        test_sensor_system,
        test_task_system,
        test_ssfr_interface,
        test_kitchen_layout,
        test_mass_and_inertia,
        test_robot_arm,
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
    print("ATLAS Physical Kitchen - Infrastructure Tests")
    print("=" * 70)
    print()

    success = run_all_tests()

    print("\n" + "=" * 70)
    if success:
        print("All tests passed!")
    else:
        print("Some tests failed!")
    print("=" * 70)
