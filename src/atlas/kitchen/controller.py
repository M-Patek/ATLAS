"""
ATLAS Kitchen Controller - 厨房任务控制器

将 SSFR 与物理厨房集成，实现：
1. 感知编码（物理状态 → SSFR 观察）
2. 动作解码（SSFR 输出 → 物理动作）
3. 任务分解（高层任务 → 动作序列）
4. 闭环控制（感知 → 决策 → 执行 → 感知）
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from atlas.kitchen import (
    Kitchen, Robot, PhysicsConfig,
    ObjectType, RobotAction,
    SensorSystem, KitchenSSFRInterface,
    TASK_LIBRARY, Task, TaskStep,
)


# ============================================================================
# 1. 状态编码器
# ============================================================================

class StateEncoder:
    """将物理状态编码为 SSFR 可用的向量"""

    def __init__(self, max_objects: int = 20):
        self.max_objects = max_objects
        # 特征维度: [x, y, vx, vy, angle, type_id, mass]
        self.feature_dim = 7

    def encode(self, kitchen: Kitchen, robot_id: str) -> np.ndarray:
        """编码当前状态"""
        if robot_id not in kitchen.robots:
            return np.zeros(self.feature_dim * self.max_objects)

        robot = kitchen.robots[robot_id]
        sensor = SensorSystem(robot, kitchen.space)

        # 获取附近物体
        all_entities = list(kitchen.entities.values())
        nearby = sensor.get_nearby_objects(all_entities, max_distance=5.0)

        features = []

        # 编码机器人自身状态
        robot_pos = robot.position
        robot_vel = (robot.body.velocity.x, robot.body.velocity.y)
        features.extend([
            robot_pos[0] / 10.0,  # 归一化位置
            robot_pos[1] / 10.0,
            robot_vel[0] / 5.0,   # 归一化速度
            robot_vel[1] / 5.0,
            robot.angle / (2 * math.pi),  # 归一化角度
            0.0,  # 类型: 机器人
            robot.body.mass / 10.0,  # 归一化质量
        ])

        # 编码附近物体
        for obj, dist in nearby[:self.max_objects - 1]:
            obj_pos = obj.position
            obj_vel = (obj.body.velocity.x, obj.body.velocity.y)
            type_id = self._encode_type(obj.properties.obj_type)

            features.extend([
                obj_pos[0] / 10.0,
                obj_pos[1] / 10.0,
                obj_vel[0] / 5.0,
                obj_vel[1] / 5.0,
                obj.angle / (2 * math.pi),
                type_id,
                obj.body.mass / 10.0,
            ])

        # 填充剩余位置
        while len(features) < self.feature_dim * self.max_objects:
            features.extend([0.0] * self.feature_dim)

        return np.array(features[:self.feature_dim * self.max_objects], dtype=np.float32)

    def _encode_type(self, obj_type: ObjectType) -> float:
        """编码物体类型为数值"""
        type_map = {
            ObjectType.ROBOT: 0.0,
            ObjectType.CUP: 1.0,
            ObjectType.PLATE: 2.0,
            ObjectType.BOWL: 3.0,
            ObjectType.UTENSIL: 4.0,
            ObjectType.INGREDIENT: 5.0,
            ObjectType.APPLIANCE: 6.0,
            ObjectType.CONTAINER: 7.0,
            ObjectType.FURNITURE: 8.0,
        }
        return type_map.get(obj_type, 9.0) / 10.0


# ============================================================================
# 2. 动作解码器
# ============================================================================

class ActionDecoder:
    """将 SSFR 输出解码为物理动作"""

    def __init__(self):
        self.action_map = {
            0: RobotAction.MOVE_FORWARD,
            1: RobotAction.MOVE_BACKWARD,
            2: RobotAction.TURN_LEFT,
            3: RobotAction.TURN_RIGHT,
            4: RobotAction.STOP,
            5: RobotAction.GRIP,
            6: RobotAction.RELEASE,
            7: RobotAction.ARM_UP,
            8: RobotAction.ARM_DOWN,
            9: RobotAction.ARM_EXTEND,
            10: RobotAction.ARM_RETRACT,
        }

    def decode(self, action_id: int, robot: Robot, kitchen: Kitchen,
               target_position: Optional[Tuple[float, float]] = None) -> bool:
        """执行动作"""
        action = self.action_map.get(action_id, RobotAction.STOP)

        if action == RobotAction.MOVE_FORWARD:
            robot.move_forward()
            return True

        elif action == RobotAction.MOVE_BACKWARD:
            robot.move_backward()
            return True

        elif action == RobotAction.TURN_LEFT:
            robot.turn_left()
            return True

        elif action == RobotAction.TURN_RIGHT:
            robot.turn_right()
            return True

        elif action == RobotAction.STOP:
            robot.stop()
            return True

        elif action == RobotAction.GRIP:
            return self._execute_grip(robot, kitchen)

        elif action == RobotAction.RELEASE:
            return robot.arm.release()

        elif action == RobotAction.ARM_UP:
            robot.arm.set_arm_angle(robot.arm.arm_angle + 0.2)
            return True

        elif action == RobotAction.ARM_DOWN:
            robot.arm.set_arm_angle(robot.arm.arm_angle - 0.2)
            return True

        elif action == RobotAction.ARM_EXTEND:
            robot.arm.set_extension(robot.arm.arm_extension + 0.1)
            return True

        elif action == RobotAction.ARM_RETRACT:
            robot.arm.set_extension(robot.arm.arm_extension - 0.1)
            return True

        return False

    def _execute_grip(self, robot: Robot, kitchen: Kitchen) -> bool:
        """执行抓取动作"""
        if robot.arm.is_gripping:
            return False

        # 获取抓取点
        grip_pos = robot.arm.get_grip_position()

        # 查找最近的物体
        all_entities = list(kitchen.entities.values())
        sensor = SensorSystem(robot, kitchen.space)
        nearby = sensor.get_nearby_objects(all_entities, max_distance=0.5)

        if not nearby:
            return False

        # 找到最近的可抓取物体
        for obj, dist in nearby:
            if obj.properties.grabbable and obj.body != robot.body:
                # 创建关节约束
                joint = pymunk.PinJoint(
                    robot.body, obj.body,
                    (0, 0), (0, 0)
                )
                joint.max_force = PhysicsConfig.GRIP_STRENGTH
                kitchen.space.add(joint)

                robot.arm.is_gripping = True
                robot.arm.gripped_object = obj
                robot.arm.grip_joint = joint
                return True

        return False


# ============================================================================
# 3. 导航控制器
# ============================================================================

class NavigationController:
    """导航控制器 - 简单的目标追踪"""

    def __init__(self, robot: Robot):
        self.robot = robot
        self.target: Optional[Tuple[float, float]] = None
        self.arrival_threshold = 0.3  # 到达阈值 (m)

    def set_target(self, position: Tuple[float, float]):
        """设置目标位置"""
        self.target = position

    def update(self, dt: float) -> bool:
        """更新导航，返回是否到达"""
        if self.target is None:
            self.robot.stop()
            return False

        robot_pos = self.robot.position
        target = self.target

        # 计算距离和方向
        dx = target[0] - robot_pos[0]
        dy = target[1] - robot_pos[1]
        distance = math.sqrt(dx**2 + dy**2)

        # 检查是否到达
        if distance < self.arrival_threshold:
            self.robot.stop()
            self.target = None
            return True

        # 计算目标角度
        target_angle = math.atan2(dy, dx)

        # 计算角度差
        angle_diff = target_angle - self.robot.angle
        # 归一化到 [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # 转向控制
        if abs(angle_diff) > 0.2:  # 角度差阈值
            if angle_diff > 0:
                self.robot.turn_left(torque=2.0)
            else:
                self.robot.turn_right(torque=2.0)
            # 前进力较小，主要转向
            self.robot.move_forward(force=2.0)
        else:
            # 角度对齐，全速前进
            self.robot.move_forward(force=8.0)

        return False


# ============================================================================
# 4. 任务执行器
# ============================================================================

class TaskExecutor:
    """任务执行器 - 执行动作序列"""

    def __init__(self, robot: Robot, kitchen: Kitchen):
        self.robot = robot
        self.kitchen = kitchen
        self.nav = NavigationController(robot)
        self.action_decoder = ActionDecoder()
        self.state_encoder = StateEncoder()

        # 当前任务
        self.current_task: Optional[Task] = None
        self.task_history: List[Dict[str, Any]] = []

    def assign_task(self, task: Task) -> bool:
        """分配任务"""
        self.current_task = task
        task.current_step = 0
        task.completed = False
        task.failed = False
        task.elapsed_time = 0.0
        return True

    def step(self, dt: float) -> Dict[str, Any]:
        """执行一步任务"""
        if self.current_task is None:
            return {'status': 'idle'}

        if self.current_task.completed:
            return {'status': 'completed', 'task': self.current_task.name}

        if self.current_task.failed:
            return {'status': 'failed', 'task': self.current_task.name}

        # 更新时间
        self.current_task.elapsed_time += dt
        if self.current_task.elapsed_time > self.current_task.time_limit:
            self.current_task.failed = True
            return {'status': 'timeout', 'task': self.current_task.name}

        # 获取当前步骤
        step = self.current_task.get_current_step()
        if step is None:
            self.current_task.completed = True
            return {'status': 'completed', 'task': self.current_task.name}

        # 执行步骤
        result = self._execute_step(step, dt)

        # 记录历史
        self.task_history.append({
            'time': self.kitchen.time,
            'step': step.action,
            'target': step.target,
            'result': result,
        })

        return {
            'status': 'running',
            'task': self.current_task.name,
            'step': step.action,
            'target': step.target,
            'progress': self.current_task.get_progress(),
            'result': result,
        }

    def _execute_step(self, step: TaskStep, dt: float) -> str:
        """执行单个步骤"""
        action = step.action
        target = step.target

        if action == 'move_to':
            return self._execute_move_to(target)

        elif action == 'grab':
            return self._execute_grab(target)

        elif action == 'place':
            return self._execute_place(target)

        elif action == 'press_button':
            return self._execute_press_button(target)

        elif action == 'wait':
            return self._execute_wait()

        else:
            return f'unknown_action: {action}'

    def _execute_move_to(self, target_name: str) -> str:
        """执行移动到目标"""
        # 查找目标位置
        target_pos = None

        # 在物体中查找
        for obj in self.kitchen.objects.values():
            if obj.properties.name.lower() == target_name.lower():
                target_pos = obj.position
                break

        # 在预定义位置中查找
        if target_pos is None:
            target_pos = self._get_target_position(target_name)

        if target_pos is None:
            return f'target_not_found: {target_name}'

        # 设置导航目标
        self.nav.set_target(target_pos)

        # 更新导航
        arrived = self.nav.update(PhysicsConfig.TIME_STEP)

        if arrived:
            # 完成步骤
            self.current_task.complete_step()
            return 'arrived'

        return 'moving'

    def _execute_grab(self, target_name: str) -> str:
        """执行抓取"""
        # 尝试抓取
        success = self.action_decoder.decode(5, self.robot, self.kitchen)  # GRIP = 5

        if success:
            self.current_task.complete_step()
            return 'grabbed'

        return 'grabbing'

    def _execute_place(self, target_name: str) -> str:
        """执行放置"""
        # 释放
        success = self.action_decoder.decode(6, self.robot, self.kitchen)  # RELEASE = 6

        if success:
            self.current_task.complete_step()
            return 'placed'

        return 'releasing'

    def _execute_press_button(self, target_name: str) -> str:
        """执行按按钮"""
        # 检查是否在目标附近
        robot_pos = self.robot.position
        target_pos = self._get_target_position(target_name)

        if target_pos:
            dist = math.sqrt((robot_pos[0] - target_pos[0])**2 +
                           (robot_pos[1] - target_pos[1])**2)
            if dist < 0.5:
                self.current_task.complete_step()
                return 'button_pressed'

        return 'moving_to_button'

    def _execute_wait(self) -> str:
        """执行等待"""
        # 简单等待，直接完成
        self.current_task.complete_step()
        return 'waited'

    def _get_target_position(self, target_name: str) -> Optional[Tuple[float, float]]:
        """获取目标位置"""
        # 预定义位置
        positions = {
            'coffee_machine': (1.5, 6.5),
            'microwave': (3.0, 6.5),
            'sink': (8.0, 6.5),
            'stove': (5.0, 6.5),
            'table': (5.0, 3.0),
        }

        # 查找物体
        for obj in self.kitchen.objects.values():
            if target_name.lower() in obj.properties.name.lower():
                return obj.position

        return positions.get(target_name.lower())


# ============================================================================
# 5. SSFR 集成控制器
# ============================================================================

class SSFRKitchenController:
    """SSFR 厨房控制器 - 将 SSFR 与物理厨房集成"""

    def __init__(self, kitchen: Kitchen, ssfr=None):
        self.kitchen = kitchen
        self.ssfr = ssfr
        self.executors: Dict[str, TaskExecutor] = {}
        self.interface = KitchenSSFRInterface(kitchen)
        self.state_encoder = StateEncoder()
        self.action_decoder = ActionDecoder()

    def register_robot(self, robot_id: str):
        """注册机器人到控制器"""
        if robot_id in self.kitchen.robots:
            robot = self.kitchen.robots[robot_id]
            self.executors[robot_id] = TaskExecutor(robot, self.kitchen)

    def assign_task(self, robot_id: str, task_name: str) -> bool:
        """分配任务给机器人"""
        if robot_id not in self.executors:
            return False

        if task_name not in TASK_LIBRARY:
            return False

        task = TASK_LIBRARY[task_name]
        # 深拷贝任务
        import copy
        task_copy = copy.deepcopy(task)
        return self.executors[robot_id].assign_task(task_copy)

    def step(self, dt: float = None) -> Dict[str, Any]:
        """执行一步控制"""
        if dt is None:
            dt = PhysicsConfig.TIME_STEP

        results = {}

        # 执行每个机器人的任务
        for robot_id, executor in self.executors.items():
            result = executor.step(dt)
            results[robot_id] = result

            # 如果有 SSFR，更新感知
            if self.ssfr:
                self._update_ssfr(robot_id)

        return results

    def _update_ssfr(self, robot_id: str):
        """更新 SSFR 感知"""
        if robot_id not in self.kitchen.robots:
            return

        # 获取观察
        obs = self.interface.get_observation(robot_id)

        # 编码状态
        state_vector = self.state_encoder.encode(self.kitchen, robot_id)

        # 更新 SSFR
        robot = self.kitchen.robots[robot_id]
        self.ssfr.perceive(robot.position, {
            'position': robot.position,
            'angle': robot.angle,
            'state_vector': state_vector,
            'nearby_objects': obs.get('nearby_objects', []),
        }, active_space_name='ricci')

    def get_status(self) -> Dict[str, Any]:
        """获取控制器状态"""
        return {
            'robots': {
                rid: {
                    'position': r.position,
                    'task': self.executors.get(rid, None) and
                           self.executors[rid].current_task and
                           self.executors[rid].current_task.name,
                    'progress': self.executors.get(rid, None) and
                               self.executors[rid].current_task and
                               self.executors[rid].current_task.get_progress(),
                }
                for rid, r in self.kitchen.robots.items()
            },
            'kitchen_time': self.kitchen.time,
        }


# ============================================================================
# 6. 主程序
# ============================================================================

def run_task_demo(task_name: str = 'make_coffee', duration: float = 30.0):
    """运行任务演示"""
    print("=" * 70)
    print(f"Task Demo: {task_name}")
    print("=" * 70)

    from atlas.kitchen import create_demo_kitchen

    # 创建厨房
    kitchen = create_demo_kitchen()

    # 创建控制器
    controller = SSFRKitchenController(kitchen)

    # 注册机器人
    robot_id = list(kitchen.robots.keys())[0]
    controller.register_robot(robot_id)

    # 分配任务
    success = controller.assign_task(robot_id, task_name)
    print(f"Task assigned: {success}")

    # 运行模拟
    step_count = 0
    max_steps = int(duration / PhysicsConfig.TIME_STEP)

    while step_count < max_steps:
        # 物理步进
        kitchen.step()

        # 控制步进
        results = controller.step()

        # 检查任务状态
        robot_result = results.get(robot_id, {})
        if robot_result.get('status') in ['completed', 'failed', 'timeout']:
            print(f"\nTask finished: {robot_result['status']}")
            break

        # 打印进度
        if step_count % 60 == 0:  # 每秒打印一次
            progress = robot_result.get('progress', 0)
            step = robot_result.get('step', 'idle')
            print(f"  Step {step_count//60:3d}s: {step:15s} | Progress: {progress*100:5.1f}%")

        step_count += 1

    # 最终结果
    status = controller.get_status()
    print(f"\nFinal status:")
    print(f"  Time: {status['kitchen_time']:.1f}s")
    print(f"  Steps: {step_count}")

    return status


if __name__ == "__main__":
    print("ATLAS Kitchen Controller")
    print("=" * 70)
    print()

    # 运行任务演示
    status = run_task_demo('make_coffee', duration=30.0)
