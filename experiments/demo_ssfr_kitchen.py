"""
ATLAS SSFR Kitchen Integration - SSFR与物理厨房的集成

实现SSFR在物理厨房中的实际应用：
1. 物理状态 → 离散空间表示（适配层）
2. SSFR感知 → 结构发现 → 动作生成
3. 闭环：物理执行 → 观测 → 结构更新

架构:
    PhysicalKitchen → KitchenSpaceAdapter → SSFR → ActionDecoder → PhysicalKitchen
         ↑___________________________________________________________|
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import uuid
import time

from atlas.core import SSFREnhanced, CognitiveSpace
from atlas.core.registry import create_space
from atlas.kitchen import (
    Kitchen, Robot, PhysicsConfig, ObjectType, RobotAction,
    SensorSystem, KitchenSSFRInterface, TASK_LIBRARY, Task, TaskStep,
    create_demo_kitchen,
)


# ============================================================================
# 1. 物理空间适配器 - 将连续物理状态映射到离散认知空间
# ============================================================================

class KitchenSpaceAdapter:
    """厨房空间适配器

    将物理厨房的连续状态映射到SSFR可用的离散网格表示。
    这是物理世界与认知空间之间的桥梁。

    映射策略:
    - 位置: 连续(x,y) → 离散网格单元 (grid_x, grid_y)
    - 障碍物: 物体占据的网格标记为障碍
    - 目标: 任务目标位置映射到网格
    - 不确定性: 基于传感器噪声和物理状态
    """

    def __init__(self, kitchen: Kitchen, grid_resolution: float = 0.5):
        """
        Args:
            kitchen: 物理厨房环境
            grid_resolution: 网格分辨率（米/单元）
        """
        self.kitchen = kitchen
        self.grid_resolution = grid_resolution

        # 网格尺寸
        self.grid_width = int(kitchen.width / grid_resolution) + 1
        self.grid_height = int(kitchen.height / grid_resolution) + 1

        # 缓存
        self._obstacle_cache = None
        self._cache_timestamp = -1

    def world_to_grid(self, position: Tuple[float, float]) -> Tuple[int, int]:
        """世界坐标 → 网格坐标"""
        x, y = position
        grid_x = int(x / self.grid_resolution)
        grid_y = int(y / self.grid_resolution)
        # 限制在网格范围内
        grid_x = max(0, min(grid_x, self.grid_width - 1))
        grid_y = max(0, min(grid_y, self.grid_height - 1))
        return (grid_x, grid_y)

    def grid_to_world(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        """网格坐标 → 世界坐标（中心点）"""
        gx, gy = grid_pos
        x = (gx + 0.5) * self.grid_resolution
        y = (gy + 0.5) * self.grid_resolution
        return (x, y)

    def get_obstacles(self, robot_id: str) -> List[Tuple[int, int]]:
        """获取机器人视角的障碍物（网格坐标）"""
        # 使用缓存避免重复计算
        if self._obstacle_cache is not None and \
           self._cache_timestamp == self.kitchen.step_count:
            return self._obstacle_cache

        obstacles = []
        occupied = set()

        # 标记所有物体占据的网格
        for obj in self.kitchen.objects.values():
            if obj.properties.obj_type == ObjectType.ROBOT:
                continue
            # 物体占据的网格（考虑半径）
            pos = obj.position
            radius = obj.properties.radius
            grid_pos = self.world_to_grid(pos)

            # 标记物体周围的网格为障碍
            grid_radius = int(radius / self.grid_resolution) + 1
            for dx in range(-grid_radius, grid_radius + 1):
                for dy in range(-grid_radius, grid_radius + 1):
                    gx = grid_pos[0] + dx
                    gy = grid_pos[1] + dy
                    if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                        # 检查实际距离
                        world_pos = self.grid_to_world((gx, gy))
                        dist = math.sqrt(
                            (world_pos[0] - pos[0])**2 +
                            (world_pos[1] - pos[1])**2
                        )
                        if dist < radius + self.grid_resolution * 0.5:
                            occupied.add((gx, gy))

        # 标记墙壁为障碍
        for gx in range(self.grid_width):
            occupied.add((gx, 0))  # 底部
            occupied.add((gx, self.grid_height - 1))  # 顶部
        for gy in range(self.grid_height):
            occupied.add((0, gy))  # 左侧
            occupied.add((self.grid_width - 1, gy))  # 右侧

        obstacles = list(occupied)
        self._obstacle_cache = obstacles
        self._cache_timestamp = self.kitchen.step_count
        return obstacles

    def get_goal_position(self, robot_id: str, task: Optional[Task] = None) -> Optional[Tuple[int, int]]:
        """获取当前任务目标的网格位置"""
        if task is None:
            return None

        step = task.get_current_step()
        if step is None:
            return None

        # 查找目标位置
        target_name = step.target
        if not target_name:
            return None

        # 在物体中查找
        for obj in self.kitchen.objects.values():
            if target_name.lower() in obj.properties.name.lower():
                return self.world_to_grid(obj.position)

        # 预定义位置
        predefined = {
            'coffee_machine': (1.5, 6.5),
            'microwave': (3.0, 6.5),
            'sink': (8.0, 6.5),
            'stove': (5.0, 6.5),
            'table': (5.0, 3.0),
        }

        if target_name.lower() in predefined:
            return self.world_to_grid(predefined[target_name.lower()])

        return None

    def encode_observation(self, robot_id: str, task: Optional[Task] = None) -> Dict[str, Any]:
        """编码物理观测为SSFR可用的观测字典"""
        if robot_id not in self.kitchen.robots:
            return {}

        robot = self.kitchen.robots[robot_id]
        grid_pos = self.world_to_grid(robot.position)
        obstacles = self.get_obstacles(robot_id)
        goal = self.get_goal_position(robot_id, task)

        # 获取传感器数据
        sensor = SensorSystem(robot, self.kitchen.space)
        raycast_results = sensor.raycast()

        # 计算不确定性（基于传感器数据）
        # 不确定性来源：
        # 1. 速度越高，不确定性越大（运动模糊）
        # 2. 附近物体越多，不确定性越大（遮挡）
        # 3. 距离目标越远，不确定性越大
        velocity = robot.body.velocity
        speed = math.sqrt(velocity.x**2 + velocity.y**2)
        speed_uncertainty = min(speed / PhysicsConfig.ROBOT_MAX_SPEED, 1.0)

        nearby = sensor.get_nearby_objects(
            list(self.kitchen.entities.values()), max_distance=3.0
        )
        density_uncertainty = min(len(nearby) / 10.0, 1.0)

        goal_uncertainty = 0.0
        if goal:
            goal_world = self.grid_to_world(goal)
            dist_to_goal = math.sqrt(
                (robot.position[0] - goal_world[0])**2 +
                (robot.position[1] - goal_world[1])**2
            )
            goal_uncertainty = min(dist_to_goal / 10.0, 1.0)

        uncertainty = (speed_uncertainty + density_uncertainty + goal_uncertainty) / 3.0

        observation = {
            'position': grid_pos,
            'goal_position': goal,
            'obstacles': obstacles,
            'uncertainty': uncertainty,
            'velocity': (velocity.x, velocity.y),
            'speed': speed,
            'nearby_objects_count': len(nearby),
            'robot_angle': robot.angle,
            'task_step': task.get_current_step().action if task and task.get_current_step() else None,
            'task_target': task.get_current_step().target if task and task.get_current_step() else None,
            # 原始物理状态（用于动作解码）
            'physical_position': robot.position,
            'physical_goal': self.grid_to_world(goal) if goal else None,
            'grid_resolution': self.grid_resolution,
        }

        return observation

    def decode_action(self, action_id: int, robot: Robot,
                     observation: Dict[str, Any]) -> bool:
        """解码SSFR动作到物理动作"""
        action_map = {
            0: RobotAction.MOVE_FORWARD,
            1: RobotAction.MOVE_BACKWARD,
            2: RobotAction.TURN_LEFT,
            3: RobotAction.TURN_RIGHT,
            4: RobotAction.STOP,
            5: RobotAction.GRIP,
            6: RobotAction.RELEASE,
        }

        action = action_map.get(action_id, RobotAction.STOP)

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
            return self._execute_grip(robot)
        elif action == RobotAction.RELEASE:
            return robot.arm.release()

        return False

    def _execute_grip(self, robot: Robot) -> bool:
        """执行抓取"""
        if robot.arm.is_gripping:
            return False

        # 获取抓取点
        grip_pos = robot.arm.get_grip_position()

        # 查找最近的可抓取物体
        all_entities = list(self.kitchen.entities.values())
        sensor = SensorSystem(robot, self.kitchen.space)
        nearby = sensor.get_nearby_objects(all_entities, max_distance=0.5)

        for obj, dist in nearby:
            if obj.properties.grabbable and obj.body != robot.body:
                # 创建物理约束
                import pymunk
                joint = pymunk.PinJoint(
                    robot.body, obj.body,
                    (0, 0), (0, 0)
                )
                joint.max_force = PhysicsConfig.GRIP_STRENGTH
                self.kitchen.space.add(joint)

                robot.arm.is_gripping = True
                robot.arm.gripped_object = obj
                robot.arm.grip_joint = joint
                return True

        return False

    def get_path_in_world(self, grid_path: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """将网格路径转换为世界坐标路径"""
        return [self.grid_to_world(pos) for pos in grid_path]


# ============================================================================
# 2. 物理SSFR - 封装SSFR用于物理厨房
# ============================================================================

class PhysicalSSFR:
    """物理SSFR

    将SSFR封装为适用于物理厨房的形式。
    处理连续/离散转换、物理状态集成、动作解码。
    """

    def __init__(self, kitchen: Kitchen, grid_resolution: float = 0.5,
                 space_names: List[str] = None):
        """
        Args:
            kitchen: 物理厨房
            grid_resolution: 空间离散化分辨率
            space_names: 使用的认知空间列表
        """
        self.kitchen = kitchen
        self.adapter = KitchenSpaceAdapter(kitchen, grid_resolution)

        # 初始化SSFR（使用适配后的网格尺寸）
        space_names = space_names or ['ricci', 'fisher', 'wasserstein']
        self.ssfr = SSFREnhanced(
            width=self.adapter.grid_width,
            height=self.adapter.grid_height,
            space_names=space_names,
            max_structures=50,
            evolution_interval=20,
        )

        # 机器人状态
        self.robot_states: Dict[str, Dict[str, Any]] = {}

        # 结构历史
        self.structure_history: List[Dict[str, Any]] = []

        # 性能统计
        self.stats = {
            'perception_count': 0,
            'competition_count': 0,
            'evolution_count': 0,
            'total_time': 0.0,
        }

    def perceive(self, robot_id: str, task: Optional[Task] = None,
                 active_space: Optional[str] = None) -> List[Any]:
        """感知：从物理状态生成结构假设"""
        start = time.time()

        # 编码观测
        observation = self.adapter.encode_observation(robot_id, task)
        if not observation:
            return []

        grid_pos = observation['position']

        # 调用SSFR感知
        hypotheses = self.ssfr.perceive(
            position=grid_pos,
            observation=observation,
            active_space_name=active_space
        )

        # 记录状态
        self.robot_states[robot_id] = {
            'observation': observation,
            'hypotheses': [h.id for h in hypotheses],
            'timestamp': self.kitchen.time,
        }

        self.stats['perception_count'] += 1
        self.stats['total_time'] += time.time() - start

        return hypotheses

    def compete(self, robot_id: str, actual_position: Tuple[float, float]) -> Optional[Any]:
        """竞争：选择最优结构"""
        if robot_id not in self.robot_states:
            return None

        observation = self.robot_states[robot_id]['observation']
        grid_actual = self.adapter.world_to_grid(actual_position)

        actual = {
            'position': grid_actual,
            'velocity': observation.get('velocity', (0, 0)),
        }

        winner = self.ssfr.compete(observation, actual)
        self.stats['competition_count'] += 1

        return winner

    def get_best_structure(self, robot_id: str) -> Optional[Any]:
        """获取当前最佳结构"""
        best = self.ssfr.get_best_structures(n=1)
        return best[0] if best else None

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        ssfr_stats = self.ssfr.get_statistics()
        return {
            **ssfr_stats,
            'physical_perceptions': self.stats['perception_count'],
            'physical_competitions': self.stats['competition_count'],
            'avg_time_per_step': self.stats['total_time'] / max(1, self.stats['perception_count']),
            'grid_size': (self.adapter.grid_width, self.adapter.grid_height),
            'grid_resolution': self.adapter.grid_resolution,
        }

    def get_space_validity(self, robot_id: str) -> Dict[str, float]:
        """获取各空间在当前场景下的有效性"""
        validities = {}
        for name, space in self.ssfr.spaces.items():
            if robot_id in self.robot_states:
                obs = self.robot_states[robot_id]['observation']
                grid_pos = obs['position']
                validities[name] = space.compute_validity(grid_pos, obs)
            else:
                validities[name] = 0.5
        return validities


# ============================================================================
# 3. SSFR任务规划器 - 使用SSFR进行任务级规划
# ============================================================================

class SSFRTaskPlanner:
    """SSFR任务规划器

    使用SSFR发现的结构来指导任务执行。
    核心思想：SSFR发现的结构反映了环境的"认知地图"，
    规划器利用这个认知地图来做更智能的决策。
    """

    def __init__(self, physical_ssfr: PhysicalSSFR):
        self.physical_ssfr = physical_ssfr
        self.kitchen = physical_ssfr.kitchen
        self.adapter = physical_ssfr.adapter

        # 任务执行状态
        self.active_tasks: Dict[str, Task] = {}
        self.task_progress: Dict[str, Dict[str, Any]] = {}

        # 导航状态
        self.navigation_paths: Dict[str, List[Tuple[float, float]]] = {}
        self.path_index: Dict[str, int] = {}

    def assign_task(self, robot_id: str, task_name: str) -> bool:
        """分配任务"""
        if task_name not in TASK_LIBRARY:
            return False

        if robot_id not in self.kitchen.robots:
            return False

        import copy
        task = copy.deepcopy(TASK_LIBRARY[task_name])
        self.active_tasks[robot_id] = task
        self.task_progress[robot_id] = {
            'start_time': self.kitchen.time,
            'steps_completed': 0,
            'current_step_start': self.kitchen.time,
        }
        self.navigation_paths[robot_id] = []
        self.path_index[robot_id] = 0

        return True

    def step(self, robot_id: str) -> Dict[str, Any]:
        """执行一步规划"""
        if robot_id not in self.active_tasks:
            return {'status': 'no_task'}

        if robot_id not in self.kitchen.robots:
            return {'status': 'robot_not_found'}

        robot = self.kitchen.robots[robot_id]
        task = self.active_tasks[robot_id]

        # 检查任务完成
        if task.completed:
            return {
                'status': 'completed',
                'task': task.name,
                'progress': 1.0,
            }

        if task.failed:
            return {
                'status': 'failed',
                'task': task.name,
            }

        # 检查超时
        elapsed = self.kitchen.time - self.task_progress[robot_id]['start_time']
        if elapsed > task.time_limit:
            task.failed = True
            return {'status': 'timeout'}

        # 获取当前步骤
        step = task.get_current_step()
        if step is None:
            task.completed = True
            return {'status': 'completed'}

        # === SSFR 感知 ===
        # 在每一步都进行SSFR感知，获取当前环境的结构理解
        hypotheses = self.physical_ssfr.perceive(robot_id, task)

        # 获取空间有效性（哪个空间最适合当前场景）
        validity = self.physical_ssfr.get_space_validity(robot_id)
        best_space = max(validity, key=validity.get) if validity else 'ricci'

        # === 执行步骤 ===
        result = self._execute_step(robot_id, step, robot, task, best_space)

        # === SSFR 竞争（用实际结果验证）===
        actual_pos = robot.position
        winner = self.physical_ssfr.compete(robot_id, actual_pos)

        return {
            'status': 'running',
            'task': task.name,
            'step': step.action,
            'target': step.target,
            'progress': task.get_progress(),
            'best_space': best_space,
            'space_validity': validity,
            'num_hypotheses': len(hypotheses),
            'winner_id': winner.id if winner else None,
            'step_result': result,
        }

    def _execute_step(self, robot_id: str, step: TaskStep,
                     robot: Robot, task: Task, best_space: str) -> str:
        """执行单个步骤（使用SSFR指导）"""
        action = step.action
        target = step.target

        if action == 'move_to':
            return self._execute_move_to(robot_id, robot, target, best_space)

        elif action == 'grab':
            return self._execute_grab(robot_id, robot, target)

        elif action == 'place':
            return self._execute_place(robot_id, robot, target)

        elif action == 'press_button':
            return self._execute_press_button(robot_id, robot, target)

        elif action == 'wait':
            return self._execute_wait(robot_id, robot, task)

        return 'unknown_action'

    def _execute_move_to(self, robot_id: str, robot: Robot,
                        target_name: str, best_space: str) -> str:
        """执行移动（SSFR指导）"""
        # 查找目标位置
        target_pos = None
        for obj in self.kitchen.objects.values():
            if target_name.lower() in obj.properties.name.lower():
                target_pos = obj.position
                break

        if target_pos is None:
            # 预定义位置
            predefined = {
                'coffee_machine': (1.5, 6.5),
                'microwave': (3.0, 6.5),
                'sink': (8.0, 6.5),
                'stove': (5.0, 6.5),
                'table': (5.0, 3.0),
            }
            target_pos = predefined.get(target_name.lower())

        if target_pos is None:
            return 'target_not_found'

        # 计算到目标的距离
        robot_pos = robot.position
        dx = target_pos[0] - robot_pos[0]
        dy = target_pos[1] - robot_pos[1]
        distance = math.sqrt(dx**2 + dy**2)

        # 到达阈值
        arrival_threshold = 0.5

        if distance < arrival_threshold:
            robot.stop()
            self.active_tasks[robot_id].complete_step()
            self.navigation_paths[robot_id] = []
            return 'arrived'

        # === SSFR指导的导航 ===
        # 使用SSFR发现的空间结构来指导导航
        # 不同空间提供不同的导航策略

        if best_space == 'ricci':
            # Ricci空间：考虑曲率，沿测地线移动
            result = self._navigate_ricci(robot, robot_pos, target_pos, dx, dy)
        elif best_space == 'fisher':
            # Fisher空间：考虑信息几何，选择信息增益最大的路径
            result = self._navigate_fisher(robot, robot_pos, target_pos, dx, dy)
        elif best_space == 'wasserstein':
            # Wasserstein空间：考虑概率分布的最优传输
            result = self._navigate_wasserstein(robot, robot_pos, target_pos, dx, dy)
        else:
            # 默认：直接导航
            result = self._navigate_direct(robot, robot_pos, target_pos, dx, dy)

        return result

    def _navigate_direct(self, robot: Robot, robot_pos: Tuple[float, float],
                        target_pos: Tuple[float, float],
                        dx: float, dy: float) -> str:
        """直接导航（基础导航）"""
        target_angle = math.atan2(dy, dx)
        angle_diff = target_angle - robot.angle

        # 归一化角度差
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # 转向控制
        if abs(angle_diff) > 0.3:
            if angle_diff > 0:
                robot.turn_left(torque=2.0)
            else:
                robot.turn_right(torque=2.0)
            robot.move_forward(force=1.0)
        else:
            robot.move_forward(force=5.0)

        return 'moving'

    def _navigate_ricci(self, robot: Robot, robot_pos: Tuple[float, float],
                       target_pos: Tuple[float, float],
                       dx: float, dy: float) -> str:
        """Ricci空间导航：考虑曲率，避开高曲率区域"""
        # Ricci空间的特性：曲率高的区域表示"认知困难"
        # 策略：如果前方有高曲率区域，尝试绕行

        # 获取Ricci空间的曲率场
        ricci_space = self.physical_ssfr.ssfr.spaces.get('ricci')
        if ricci_space:
            fields = ricci_space.get_visualization_fields()
            if 'curvature' in fields:
                curvature = fields['curvature']
                grid_pos = self.adapter.world_to_grid(robot_pos)
                gx, gy = grid_pos

                # 检查前方区域的曲率
                if 0 <= gx < curvature.shape[0] and 0 <= gy < curvature.shape[1]:
                    current_curvature = curvature[gx, gy]

                    # 如果当前位置曲率很高，尝试转向避开
                    if current_curvature > 2.0:
                        # 随机选择转向方向
                        import random
                        if random.random() > 0.5:
                            robot.turn_left(torque=3.0)
                        else:
                            robot.turn_right(torque=3.0)
                        robot.move_forward(force=1.0)
                        return 'navigating_ricci_avoid'

        # 否则使用直接导航
        return self._navigate_direct(robot, robot_pos, target_pos, dx, dy)

    def _navigate_fisher(self, robot: Robot, robot_pos: Tuple[float, float],
                        target_pos: Tuple[float, float],
                        dx: float, dy: float) -> str:
        """Fisher空间导航：考虑信息增益"""
        # Fisher空间的特性：信息距离表示"认知差异"
        # 策略：选择能提供最多信息的路径（探索未知区域）

        # 简化为直接导航（Fisher空间的优势在结构发现阶段更明显）
        return self._navigate_direct(robot, robot_pos, target_pos, dx, dy)

    def _navigate_wasserstein(self, robot: Robot, robot_pos: Tuple[float, float],
                             target_pos: Tuple[float, float],
                             dx: float, dy: float) -> str:
        """Wasserstein空间导航：考虑最优传输"""
        # Wasserstein空间的特性：最优传输距离
        # 策略：选择"能量消耗最小"的路径

        # 简化为直接导航
        return self._navigate_direct(robot, robot_pos, target_pos, dx, dy)

    def _execute_grab(self, robot_id: str, robot: Robot, target_name: str) -> str:
        """执行抓取"""
        # 检查是否已抓取
        if robot.arm.is_gripping:
            self.active_tasks[robot_id].complete_step()
            return 'already_gripping'

        # 尝试抓取
        success = self.adapter._execute_grip(robot)

        if success:
            self.active_tasks[robot_id].complete_step()
            return 'grabbed'

        # 如果抓取失败，可能是距离太远，继续移动
        return 'approaching_for_grab'

    def _execute_place(self, robot_id: str, robot: Robot, target_name: str) -> str:
        """执行放置"""
        if not robot.arm.is_gripping:
            self.active_tasks[robot_id].complete_step()
            return 'nothing_to_place'

        success = robot.arm.release()

        if success:
            self.active_tasks[robot_id].complete_step()
            return 'placed'

        return 'releasing'

    def _execute_press_button(self, robot_id: str, robot: Robot,
                             target_name: str) -> str:
        """执行按按钮"""
        # 检查是否在目标附近
        robot_pos = robot.position
        target_pos = None

        for obj in self.kitchen.objects.values():
            if target_name.lower() in obj.properties.name.lower():
                target_pos = obj.position
                break

        if target_pos:
            dist = math.sqrt(
                (robot_pos[0] - target_pos[0])**2 +
                (robot_pos[1] - target_pos[1])**2
            )
            if dist < 0.8:
                robot.stop()
                self.active_tasks[robot_id].complete_step()
                return 'button_pressed'

        # 需要靠近
        return 'approaching_button'

    def _execute_wait(self, robot_id: str, robot: Robot, task: Task) -> str:
        """执行等待"""
        robot.stop()

        # 检查等待时间
        step = task.get_current_step()
        if step and step.target:
            # 如果有目标，检查目标状态
            for obj in self.kitchen.objects.values():
                if step.target.lower() in obj.properties.name.lower():
                    # 模拟状态变化（如咖啡机完成）
                    step_time = self.kitchen.time - self.task_progress[robot_id]['current_step_start']
                    if step_time > 3.0:  # 等待3秒
                        self.active_tasks[robot_id].complete_step()
                        return 'wait_complete'
                    return 'waiting'

        # 默认等待
        step_time = self.kitchen.time - self.task_progress[robot_id]['current_step_start']
        if step_time > 2.0:
            self.active_tasks[robot_id].complete_step()
            return 'wait_complete'

        return 'waiting'

    def get_task_status(self, robot_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        if robot_id not in self.active_tasks:
            return {'status': 'no_task'}

        task = self.active_tasks[robot_id]
        return {
            'task': task.name,
            'status': 'completed' if task.completed else 'failed' if task.failed else 'running',
            'progress': task.get_progress(),
            'current_step': task.current_step,
            'total_steps': len(task.steps),
            'elapsed_time': self.kitchen.time - self.task_progress[robot_id]['start_time'],
        }


# ============================================================================
# 4. 完整演示
# ============================================================================

def demo_ssfr_perception():
    """演示SSFR感知"""
    print("=" * 70)
    print("DEMO: SSFR Perception in Physical Kitchen")
    print("=" * 70)

    # 创建厨房
    kitchen = create_demo_kitchen()
    robot_id = list(kitchen.robots.keys())[0]

    # 创建物理SSFR
    physical_ssfr = PhysicalSSFR(kitchen, grid_resolution=0.5)

    print(f"\nKitchen: {kitchen.width}m x {kitchen.height}m")
    print(f"Grid: {physical_ssfr.adapter.grid_width} x {physical_ssfr.adapter.grid_height}")
    print(f"Resolution: {physical_ssfr.adapter.grid_resolution}m/cell")

    # 执行感知
    print("\n--- Perception Step 1 ---")
    hypotheses = physical_ssfr.perceive(robot_id)
    print(f"Generated {len(hypotheses)} hypotheses")

    for i, hyp in enumerate(hypotheses[:3]):
        print(f"  Hypothesis {i+1}: {hyp.name} (fitness={hyp.fitness:.3f})")
        print(f"    Spaces: {list(hyp.representations.keys())}")

    # 移动机器人后再感知
    robot = kitchen.robots[robot_id]
    robot.move_forward(force=5.0)
    for _ in range(30):
        kitchen.step()

    print(f"\n--- Perception Step 2 (after movement) ---")
    print(f"Robot new position: ({robot.position[0]:.2f}, {robot.position[1]:.2f})")

    hypotheses = physical_ssfr.perceive(robot_id)
    print(f"Generated {len(hypotheses)} hypotheses")

    # 获取空间有效性
    validity = physical_ssfr.get_space_validity(robot_id)
    print(f"\nSpace validity:")
    for name, val in validity.items():
        print(f"  {name}: {val:.3f}")

    # 统计
    stats = physical_ssfr.get_statistics()
    print(f"\nStatistics:")
    print(f"  Perceptions: {stats['physical_perceptions']}")
    print(f"  Pool size: {stats['pool_stats']['num_structures']}")
    print(f"  Avg time: {stats['avg_time_per_step']*1000:.2f}ms")

    print("\n[OK] SSFR perception demo complete")
    return physical_ssfr


def demo_ssfr_task_planning():
    """演示SSFR任务规划"""
    print("\n" + "=" * 70)
    print("DEMO: SSFR Task Planning")
    print("=" * 70)

    # 创建厨房
    kitchen = create_demo_kitchen()
    robot_id = list(kitchen.robots.keys())[0]

    # 创建SSFR规划器
    physical_ssfr = PhysicalSSFR(kitchen, grid_resolution=0.5)
    planner = SSFRTaskPlanner(physical_ssfr)

    # 分配任务
    task_name = 'make_coffee'
    success = planner.assign_task(robot_id, task_name)
    print(f"\nTask assigned: {task_name} (success={success})")

    # 运行任务
    print(f"\nRunning task...")
    max_steps = 600  # 10 seconds

    for step in range(max_steps):
        # 物理步进
        kitchen.step()

        # 规划步进
        result = planner.step(robot_id)

        # 打印进度
        if step % 60 == 0:
            status = result.get('status', 'unknown')
            progress = result.get('progress', 0)
            current_step = result.get('step', 'idle')
            best_space = result.get('best_space', 'none')
            num_hyp = result.get('num_hypotheses', 0)

            print(f"  {step//60:3d}s: {current_step:15s} | "
                  f"progress={progress*100:5.1f}% | "
                  f"space={best_space:12s} | "
                  f"hypotheses={num_hyp}")

        # 检查完成
        if result.get('status') in ['completed', 'failed', 'timeout']:
            print(f"\nTask finished: {result['status']}")
            break

    # 最终状态
    final_status = planner.get_task_status(robot_id)
    print(f"\nFinal status:")
    print(f"  Task: {final_status['task']}")
    print(f"  Progress: {final_status['progress']*100:.1f}%")
    print(f"  Steps: {final_status['current_step']}/{final_status['total_steps']}")

    # SSFR统计
    stats = physical_ssfr.get_statistics()
    print(f"\nSSFR Statistics:")
    print(f"  Total perceptions: {stats['physical_perceptions']}")
    print(f"  Structure pool size: {stats['pool_stats']['num_structures']}")
    print(f"  Average fitness: {stats['pool_stats']['avg_fitness']:.3f}")

    print("\n[OK] SSFR task planning demo complete")
    return planner


def demo_ssfr_structure_evolution():
    """演示SSFR结构演化"""
    print("\n" + "=" * 70)
    print("DEMO: SSFR Structure Evolution")
    print("=" * 70)

    # 创建厨房
    kitchen = create_demo_kitchen()
    robot_id = list(kitchen.robots.keys())[0]
    robot = kitchen.robots[robot_id]

    # 创建SSFR
    physical_ssfr = PhysicalSSFR(kitchen, grid_resolution=0.5)

    print("\nPhase 1: Initial perception (robot stationary)")
    hypotheses = physical_ssfr.perceive(robot_id)
    print(f"  Hypotheses: {len(hypotheses)}")

    # 移动机器人到不同位置，观察结构演化
    positions = [
        (2.0, 2.0),   # 起始位置
        (4.0, 2.0),   # 向右移动
        (4.0, 4.0),   # 向上移动
        (6.0, 4.0),   # 向右移动
        (6.0, 6.0),   # 向上移动（靠近咖啡机）
    ]

    for i, target_pos in enumerate(positions[1:], 1):
        print(f"\nPhase {i+1}: Moving to ({target_pos[0]}, {target_pos[1]})")

        # 移动机器人
        robot.position = target_pos
        robot.body.velocity = (0, 0)

        # 步进物理
        for _ in range(10):
            kitchen.step()

        # 感知
        hypotheses = physical_ssfr.perceive(robot_id)

        # 竞争
        winner = physical_ssfr.compete(robot_id, target_pos)

        # 演化（每5步）
        if i % 5 == 0:
            new_structs = physical_ssfr.ssfr.evolve()
            print(f"  Evolution: {len(new_structs)} new structures")

        stats = physical_ssfr.get_statistics()
        print(f"  Hypotheses: {len(hypotheses)}")
        print(f"  Pool size: {stats['pool_stats']['num_structures']}")
        print(f"  Avg fitness: {stats['pool_stats']['avg_fitness']:.3f}")
        if winner:
            print(f"  Winner: {winner.name} (fitness={winner.fitness:.3f})")

    # 最终统计
    print(f"\nFinal Statistics:")
    stats = physical_ssfr.get_statistics()
    print(f"  Total steps: {stats['step_count']}")
    print(f"  Total perceptions: {stats['physical_perceptions']}")
    print(f"  Total competitions: {stats['physical_competitions']}")
    print(f"  Pool size: {stats['pool_stats']['num_structures']}")
    print(f"  Pool generation: {stats['pool_stats']['generation']}")
    print(f"  Reuse rate: {stats.get('reuse_rate', 0):.3f}")

    print("\n[OK] SSFR structure evolution demo complete")


def demo_ssfr_comparison():
    """演示不同空间的比较"""
    print("\n" + "=" * 70)
    print("DEMO: SSFR Space Comparison")
    print("=" * 70)

    # 创建厨房
    kitchen = create_demo_kitchen()
    robot_id = list(kitchen.robots.keys())[0]

    # 测试不同空间
    space_configs = [
        (['ricci'], 'Ricci only'),
        (['fisher'], 'Fisher only'),
        (['wasserstein'], 'Wasserstein only'),
        (['ricci', 'fisher', 'wasserstein'], 'All spaces'),
    ]

    results = []

    for space_names, label in space_configs:
        print(f"\n--- {label} ---")

        # 创建新的SSFR实例
        physical_ssfr = PhysicalSSFR(kitchen, grid_resolution=0.5, space_names=space_names)
        planner = SSFRTaskPlanner(physical_ssfr)

        # 分配任务
        planner.assign_task(robot_id, 'make_coffee')

        # 运行任务
        steps_to_complete = 0
        max_steps = 600

        for step in range(max_steps):
            kitchen.step()
            result = planner.step(robot_id)

            if result.get('status') in ['completed', 'failed']:
                steps_to_complete = step
                break

        # 统计
        stats = physical_ssfr.get_statistics()
        final_status = planner.get_task_status(robot_id)

        results.append({
            'label': label,
            'spaces': space_names,
            'steps': steps_to_complete,
            'progress': final_status['progress'],
            'perceptions': stats['physical_perceptions'],
            'pool_size': stats['pool_stats']['num_structures'],
            'avg_time': stats['avg_time_per_step'] * 1000,
        })

    # 打印比较结果
    print(f"\n{'='*70}")
    print("Comparison Results")
    print(f"{'='*70}")
    print(f"{'Configuration':<20} {'Steps':<8} {'Progress':<10} {'Perceptions':<12} {'Pool':<8} {'Time(ms)':<10}")
    print("-" * 70)

    for r in results:
        print(f"{r['label']:<20} {r['steps']:<8} {r['progress']*100:>6.1f}%   "
              f"{r['perceptions']:<12} {r['pool_size']:<8} {r['avg_time']:>6.2f}")

    print("\n[OK] SSFR comparison demo complete")


def run_all_demos():
    """运行所有演示"""
    print("=" * 70)
    print("ATLAS SSFR Kitchen Integration Demos")
    print("=" * 70)

    demo_ssfr_perception()
    demo_ssfr_task_planning()
    demo_ssfr_structure_evolution()
    demo_ssfr_comparison()

    print("\n" + "=" * 70)
    print("All demos complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_demos()
