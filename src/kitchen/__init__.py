"""
ATLAS Physical Kitchen - 物理厨房沙盒

基于 pymunk 的2D物理模拟厨房环境。

核心设计:
- 真实物理: 重力、碰撞、摩擦、质量
- 机器人: 刚体 + 轮子 + 机械臂 + 传感器
- 物体: 杯子、盘子、食材等可交互物体
- 任务: 动作序列 (如"做咖啡")
- SSFR集成: 感知→结构→策略

依赖:
    pip install pymunk pygame numpy
"""

import numpy as np
import pymunk
import pymunk.pygame_util
import pygame
from typing import Dict, List, Tuple, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import uuid
import math
import time
from collections import defaultdict


# ============================================================================
# 1. 常量与配置
# ============================================================================

class PhysicsConfig:
    """物理配置"""
    # 重力 (m/s²)
    GRAVITY = (0, -9.8)

    # 时间步长
    TIME_STEP = 1/60  # 60 FPS

    # 速度迭代和位置迭代
    VELOCITY_ITERATIONS = 10
    POSITION_ITERATIONS = 10

    # 机器人参数
    ROBOT_MASS = 1.0  # kg
    ROBOT_RADIUS = 0.3  # m
    ROBOT_MAX_SPEED = 2.0  # m/s
    ROBOT_MAX_FORCE = 10.0  # N
    ROBOT_TORQUE = 5.0  # N·m

    # 机械臂参数
    ARM_LENGTH = 0.5  # m
    ARM_MASS = 0.2  # kg
    GRIP_STRENGTH = 5.0  # N

    # 传感器参数
    SENSOR_RANGE = 5.0  # m
    SENSOR_FOV = 120  # degrees
    SENSOR_RAYS = 36  # 射线数量

    # 摩擦系数
    FRICTION_FLOOR = 0.8
    FRICTION_ROBOT = 0.5
    FRICTION_OBJECT = 0.3

    # 弹性系数
    RESTITUTION = 0.2


# ============================================================================
# 2. 枚举定义
# ============================================================================

class ObjectType(Enum):
    """物体类型"""
    CUP = auto()
    PLATE = auto()
    BOWL = auto()
    UTENSIL = auto()  # 餐具（刀叉勺）
    INGREDIENT = auto()  # 食材
    APPLIANCE = auto()  # 电器（咖啡机、微波炉）
    CONTAINER = auto()  # 容器（水壶、锅）
    FURNITURE = auto()  # 家具（桌子、椅子）
    ROBOT = auto()


class MaterialType(Enum):
    """材料类型"""
    CERAMIC = auto()  # 陶瓷
    GLASS = auto()  # 玻璃
    PLASTIC = auto()  # 塑料
    METAL = auto()  # 金属
    WOOD = auto()  # 木头
    FOOD = auto()  # 食物


class RobotAction(Enum):
    """机器人动作"""
    MOVE_FORWARD = auto()
    MOVE_BACKWARD = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    STOP = auto()
    GRIP = auto()
    RELEASE = auto()
    ARM_UP = auto()
    ARM_DOWN = auto()
    ARM_EXTEND = auto()
    ARM_RETRACT = auto()


# ============================================================================
# 3. 物体系统
# ============================================================================

@dataclass
class ObjectProperties:
    """物体物理属性"""
    name: str
    obj_type: ObjectType
    material: MaterialType
    mass: float  # kg
    radius: float  # m (圆形) 或 half-width
    height: float = 0.0  # m (用于矩形)
    friction: float = 0.3
    restitution: float = 0.2
    color: Tuple[int, int, int] = (128, 128, 128)
    # 是否可抓取
    grabbable: bool = True
    # 是否可堆叠
    stackable: bool = True
    # 内容物（如杯子里的水）
    contents: Optional[str] = None
    # 状态（如：空、满、热、冷）
    state: str = "empty"

    def __post_init__(self):
        if self.height == 0.0:
            self.height = self.radius * 2


# 预定义物体库
OBJECT_LIBRARY = {
    # 杯子
    'coffee_cup': ObjectProperties(
        name='Coffee Cup',
        obj_type=ObjectType.CUP,
        material=MaterialType.CERAMIC,
        mass=0.2,
        radius=0.04,
        height=0.12,
        color=(200, 150, 100),
    ),
    'glass': ObjectProperties(
        name='Glass',
        obj_type=ObjectType.CUP,
        material=MaterialType.GLASS,
        mass=0.15,
        radius=0.035,
        height=0.15,
        color=(180, 220, 255),
    ),
    # 盘子
    'plate': ObjectProperties(
        name='Plate',
        obj_type=ObjectType.PLATE,
        material=MaterialType.CERAMIC,
        mass=0.3,
        radius=0.15,
        height=0.02,
        color=(255, 255, 255),
        stackable=True,
    ),
    # 碗
    'bowl': ObjectProperties(
        name='Bowl',
        obj_type=ObjectType.BOWL,
        material=MaterialType.CERAMIC,
        mass=0.25,
        radius=0.12,
        height=0.08,
        color=(240, 240, 240),
    ),
    # 食材
    'apple': ObjectProperties(
        name='Apple',
        obj_type=ObjectType.INGREDIENT,
        material=MaterialType.FOOD,
        mass=0.15,
        radius=0.035,
        height=0.07,
        color=(255, 50, 50),
    ),
    'bread': ObjectProperties(
        name='Bread',
        obj_type=ObjectType.INGREDIENT,
        material=MaterialType.FOOD,
        mass=0.1,
        radius=0.06,
        height=0.04,
        color=(210, 180, 140),
    ),
    # 电器
    'coffee_machine': ObjectProperties(
        name='Coffee Machine',
        obj_type=ObjectType.APPLIANCE,
        material=MaterialType.METAL,
        mass=2.0,
        radius=0.15,
        height=0.3,
        color=(50, 50, 50),
        grabbable=False,
        stackable=False,
    ),
    'microwave': ObjectProperties(
        name='Microwave',
        obj_type=ObjectType.APPLIANCE,
        material=MaterialType.METAL,
        mass=3.0,
        radius=0.2,
        height=0.15,
        color=(80, 80, 80),
        grabbable=False,
        stackable=False,
    ),
    # 容器
    'water_kettle': ObjectProperties(
        name='Water Kettle',
        obj_type=ObjectType.CONTAINER,
        material=MaterialType.METAL,
        mass=0.5,
        radius=0.08,
        height=0.2,
        color=(192, 192, 192),
    ),
    # 餐具
    'spoon': ObjectProperties(
        name='Spoon',
        obj_type=ObjectType.UTENSIL,
        material=MaterialType.METAL,
        mass=0.05,
        radius=0.02,
        height=0.15,
        color=(192, 192, 192),
    ),
    'knife': ObjectProperties(
        name='Knife',
        obj_type=ObjectType.UTENSIL,
        material=MaterialType.METAL,
        mass=0.08,
        radius=0.015,
        height=0.2,
        color=(160, 160, 160),
    ),
}


# ============================================================================
# 4. 物理实体基类
# ============================================================================

class PhysicsEntity:
    """物理实体基类"""

    def __init__(self, body: pymunk.Body, shapes: List[pymunk.Shape],
                 properties: ObjectProperties, entity_id: Optional[str] = None):
        self.body = body
        self.shapes = shapes
        self.properties = properties
        self.id = entity_id or str(uuid.uuid4())[:8]
        self.metadata: Dict[str, Any] = {}

    @property
    def position(self) -> Tuple[float, float]:
        return (self.body.position.x, self.body.position.y)

    @position.setter
    def position(self, pos: Tuple[float, float]):
        self.body.position = pos

    @property
    def angle(self) -> float:
        return self.body.angle

    @angle.setter
    def angle(self, angle: float):
        self.body.angle = angle

    def get_aabb(self) -> Tuple[float, float, float, float]:
        """获取轴对齐包围盒 (min_x, min_y, max_x, max_y)"""
        # 简化为圆形包围盒
        r = self.properties.radius
        x, y = self.position
        return (x - r, y - r, x + r, y + r)


# ============================================================================
# 5. 机器人系统
# ============================================================================

class RobotArm:
    """机器人机械臂"""

    def __init__(self, robot_body: pymunk.Body, space: pymunk.Space):
        self.robot_body = robot_body
        self.space = space
        self.length = PhysicsConfig.ARM_LENGTH
        self.grip_strength = PhysicsConfig.GRIP_STRENGTH

        # 臂的末端位置（相对于机器人）
        self.arm_angle = 0.0  # 相对于机器人方向的角度
        self.arm_extension = 0.0  # 伸展程度 (0-1)

        # 抓取状态
        self.is_gripping = False
        self.gripped_object: Optional[PhysicsEntity] = None
        self.grip_joint: Optional[pymunk.PinJoint] = None

    def update(self, dt: float):
        """更新机械臂状态"""
        if self.is_gripping and self.gripped_object:
            # 检查物体是否还在有效范围内
            grip_pos = self.get_grip_position()
            obj_pos = self.gripped_object.position
            distance = math.sqrt((grip_pos[0] - obj_pos[0])**2 +
                                (grip_pos[1] - obj_pos[1])**2)
            if distance > self.length * 1.5:
                # 物体太远，自动释放
                self.release()

    def get_grip_position(self) -> Tuple[float, float]:
        """获取抓取点位置"""
        robot_pos = (self.robot_body.position.x, self.robot_body.position.y)
        robot_angle = self.robot_body.angle

        # 计算臂末端位置
        total_angle = robot_angle + self.arm_angle
        extension = self.length * (0.3 + 0.7 * self.arm_extension)

        x = robot_pos[0] + math.cos(total_angle) * extension
        y = robot_pos[1] + math.sin(total_angle) * extension

        return (x, y)

    def set_arm_angle(self, angle: float):
        """设置机械臂角度"""
        self.arm_angle = max(-math.pi/2, min(math.pi/2, angle))

    def set_extension(self, extension: float):
        """设置伸展程度 (0-1)"""
        self.arm_extension = max(0.0, min(1.0, extension))

    def grip(self) -> bool:
        """尝试抓取物体"""
        if self.is_gripping:
            return False

        # 获取抓取点
        grip_pos = self.get_grip_position()

        # 查找最近的物体
        nearest_obj = None
        nearest_dist = float('inf')

        # 这个需要在 Kitchen 类中实现，这里先返回 False
        return False

    def release(self) -> bool:
        """释放物体"""
        if not self.is_gripping:
            return False

        if self.grip_joint:
            self.space.remove(self.grip_joint)
            self.grip_joint = None

        self.is_gripping = False
        self.gripped_object = None
        return True


class Robot:
    """物理机器人"""

    def __init__(self, name: str, position: Tuple[float, float],
                 space: pymunk.Space):
        self.name = name
        self.space = space

        # 创建刚体
        mass = PhysicsConfig.ROBOT_MASS
        radius = PhysicsConfig.ROBOT_RADIUS
        moment = pymunk.moment_for_circle(mass, 0, radius)

        self.body = pymunk.Body(mass, moment)
        self.body.position = position
        space.add(self.body)

        # 创建形状
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.friction = PhysicsConfig.FRICTION_ROBOT
        self.shape.elasticity = PhysicsConfig.RESTITUTION
        space.add(self.shape)

        # 物理实体包装
        self.entity = PhysicsEntity(
            self.body, [self.shape],
            ObjectProperties(
                name=name,
                obj_type=ObjectType.ROBOT,
                material=MaterialType.METAL,
                mass=mass,
                radius=radius,
                color=(0, 100, 255),
                grabbable=False,
            )
        )

        # 机械臂
        self.arm = RobotArm(self.body, space)

        # 传感器
        self.sensor_data: Dict[str, Any] = {}
        self.nearby_objects: List[PhysicsEntity] = []

        # 状态
        self.velocity = (0.0, 0.0)
        self.angular_velocity = 0.0
        self.current_action: Optional[RobotAction] = None

    @property
    def position(self) -> Tuple[float, float]:
        return (self.body.position.x, self.body.position.y)

    @property
    def angle(self) -> float:
        return self.body.angle

    def apply_force(self, force: Tuple[float, float]):
        """施加力"""
        self.body.apply_force_at_local_point(force, (0, 0))

    def apply_torque(self, torque: float):
        """施加扭矩"""
        self.body.torque = torque

    def move_forward(self, force: float = None):
        """向前移动"""
        if force is None:
            force = PhysicsConfig.ROBOT_MAX_FORCE * 0.5
        angle = self.body.angle
        self.apply_force((math.cos(angle) * force, math.sin(angle) * force))

    def move_backward(self, force: float = None):
        """向后移动"""
        if force is None:
            force = PhysicsConfig.ROBOT_MAX_FORCE * 0.5
        angle = self.body.angle
        self.apply_force((-math.cos(angle) * force, -math.sin(angle) * force))

    def turn_left(self, torque: float = None):
        """向左转"""
        if torque is None:
            torque = PhysicsConfig.ROBOT_TORQUE * 0.5
        self.apply_torque(torque)

    def turn_right(self, torque: float = None):
        """向右转"""
        if torque is None:
            torque = PhysicsConfig.ROBOT_TORQUE * 0.5
        self.apply_torque(-torque)

    def stop(self):
        """停止"""
        self.body.velocity = (0, 0)
        self.body.angular_velocity = 0

    def update(self, dt: float):
        """更新机器人状态"""
        # 限制最大速度
        max_speed = PhysicsConfig.ROBOT_MAX_SPEED
        vel = self.body.velocity
        speed = math.sqrt(vel.x**2 + vel.y**2)
        if speed > max_speed:
            scale = max_speed / speed
            self.body.velocity = (vel.x * scale, vel.y * scale)

        # 更新机械臂
        self.arm.update(dt)

        # 更新传感器数据
        self._update_sensor_data()

    def _update_sensor_data(self):
        """更新传感器数据"""
        self.sensor_data['position'] = self.position
        self.sensor_data['angle'] = self.angle
        self.sensor_data['velocity'] = (self.body.velocity.x, self.body.velocity.y)
        self.sensor_data['angular_velocity'] = self.body.angular_velocity

    def get_sensor_reading(self) -> Dict[str, Any]:
        """获取传感器读数"""
        return self.sensor_data.copy()


# ============================================================================
# 6. 传感器系统
# ============================================================================

class SensorSystem:
    """机器人传感器系统"""

    def __init__(self, robot: Robot, space: pymunk.Space):
        self.robot = robot
        self.space = space
        self.range = PhysicsConfig.SENSOR_RANGE
        self.fov = PhysicsConfig.SENSOR_FOV
        self.num_rays = PhysicsConfig.SENSOR_RAYS

    def raycast(self, angle_offset: float = 0.0) -> List[Dict[str, Any]]:
        """
        射线投射传感器
        返回检测到的物体信息
        """
        results = []
        robot_pos = self.robot.position
        robot_angle = self.robot.angle

        # 计算射线方向
        start_angle = robot_angle - math.radians(self.fov / 2)
        end_angle = robot_angle + math.radians(self.fov / 2)
        angle_step = (end_angle - start_angle) / self.num_rays

        for i in range(self.num_rays):
            angle = start_angle + i * angle_step + angle_offset
            end_point = (
                robot_pos[0] + math.cos(angle) * self.range,
                robot_pos[1] + math.sin(angle) * self.range,
            )

            # pymunk 射线查询
            query = self.space.segment_query(
                robot_pos, end_point, 1, pymunk.ShapeFilter()
            )

            for info in query:
                if info.shape.body != self.robot.body:
                    hit_point = info.point
                    distance = math.sqrt(
                        (hit_point.x - robot_pos[0])**2 +
                        (hit_point.y - robot_pos[1])**2
                    )
                    results.append({
                        'angle': angle,
                        'distance': distance,
                        'point': (hit_point.x, hit_point.y),
                        'shape': info.shape,
                    })

        return results

    def get_nearby_objects(self, all_objects: List[PhysicsEntity],
                          max_distance: float = None) -> List[Tuple[PhysicsEntity, float]]:
        """获取附近的物体"""
        if max_distance is None:
            max_distance = self.range

        robot_pos = self.robot.position
        nearby = []

        for obj in all_objects:
            if obj.body == self.robot.body:
                continue

            obj_pos = obj.position
            distance = math.sqrt(
                (obj_pos[0] - robot_pos[0])**2 +
                (obj_pos[1] - robot_pos[1])**2
            )

            if distance <= max_distance:
                nearby.append((obj, distance))

        # 按距离排序
        nearby.sort(key=lambda x: x[1])
        return nearby


# ============================================================================
# 7. 厨房环境
# ============================================================================

class Kitchen:
    """物理厨房环境"""

    def __init__(self, width: float = 10.0, height: float = 8.0):
        self.width = width
        self.height = height

        # 创建物理空间
        self.space = pymunk.Space()
        self.space.gravity = PhysicsConfig.GRAVITY

        # 实体管理
        self.entities: Dict[str, PhysicsEntity] = {}
        self.robots: Dict[str, Robot] = {}
        self.objects: Dict[str, PhysicsEntity] = {}

        # 创建地板和墙壁
        self._create_walls()

        # 时间
        self.time = 0.0
        self.step_count = 0

    def _create_walls(self):
        """创建墙壁和地板"""
        # 地板 - 使用 Box 形状确保碰撞
        floor_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        floor_body.position = (self.width / 2, -0.1)
        self.space.add(floor_body)
        floor_shape = pymunk.Poly.create_box(floor_body, (self.width + 2, 0.2))
        floor_shape.friction = PhysicsConfig.FRICTION_FLOOR
        self.space.add(floor_shape)

        # 天花板
        ceiling_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        ceiling_body.position = (self.width / 2, self.height + 0.1)
        self.space.add(ceiling_body)
        ceiling_shape = pymunk.Poly.create_box(ceiling_body, (self.width + 2, 0.2))
        self.space.add(ceiling_shape)

        # 左墙
        left_wall_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        left_wall_body.position = (-0.1, self.height / 2)
        self.space.add(left_wall_body)
        left_wall_shape = pymunk.Poly.create_box(left_wall_body, (0.2, self.height + 2))
        self.space.add(left_wall_shape)

        # 右墙
        right_wall_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        right_wall_body.position = (self.width + 0.1, self.height / 2)
        self.space.add(right_wall_body)
        right_wall_shape = pymunk.Poly.create_box(right_wall_body, (0.2, self.height + 2))
        self.space.add(right_wall_shape)

    def add_robot(self, name: str, position: Tuple[float, float]) -> Robot:
        """添加机器人"""
        robot = Robot(name, position, self.space)
        self.robots[robot.entity.id] = robot
        self.entities[robot.entity.id] = robot.entity
        return robot

    def add_object(self, obj_key: str, position: Tuple[float, float],
                   angle: float = 0.0) -> Optional[PhysicsEntity]:
        """添加物体"""
        if obj_key not in OBJECT_LIBRARY:
            return None

        props = OBJECT_LIBRARY[obj_key]

        # 创建刚体
        mass = props.mass
        radius = props.radius
        moment = pymunk.moment_for_circle(mass, 0, radius)

        body = pymunk.Body(mass, moment)
        body.position = position
        body.angle = angle
        self.space.add(body)

        # 创建形状
        shape = pymunk.Circle(body, radius)
        shape.friction = props.friction
        shape.elasticity = props.restitution
        self.space.add(shape)

        # 创建实体
        entity = PhysicsEntity(body, [shape], props)
        self.objects[entity.id] = entity
        self.entities[entity.id] = entity

        return entity

    def add_static_object(self, obj_key: str, position: Tuple[float, float],
                          angle: float = 0.0) -> Optional[PhysicsEntity]:
        """添加静态物体（如家具、电器）"""
        if obj_key not in OBJECT_LIBRARY:
            return None

        props = OBJECT_LIBRARY[obj_key]

        # 静态刚体
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = position
        body.angle = angle
        self.space.add(body)

        # 创建形状
        shape = pymunk.Circle(body, props.radius)
        shape.friction = props.friction
        self.space.add(shape)

        # 创建实体
        entity = PhysicsEntity(body, [shape], props)
        self.objects[entity.id] = entity
        self.entities[entity.id] = entity

        return entity

    def remove_entity(self, entity_id: str):
        """移除实体"""
        if entity_id in self.entities:
            entity = self.entities[entity_id]
            self.space.remove(entity.body, *entity.shapes)
            del self.entities[entity_id]

            if entity_id in self.robots:
                del self.robots[entity_id]
            if entity_id in self.objects:
                del self.objects[entity_id]

    def step(self, dt: float = None):
        """物理步进"""
        if dt is None:
            dt = PhysicsConfig.TIME_STEP

        # 更新机器人
        for robot in self.robots.values():
            robot.update(dt)

        # 物理步进
        self.space.step(dt)

        # 更新时间
        self.time += dt
        self.step_count += 1

    def get_state(self) -> Dict[str, Any]:
        """获取环境状态"""
        return {
            'time': self.time,
            'step_count': self.step_count,
            'robots': {
                rid: {
                    'name': robot.name,
                    'position': robot.position,
                    'angle': robot.angle,
                    'velocity': robot.get_sensor_reading().get('velocity', (0, 0)),
                }
                for rid, robot in self.robots.items()
            },
            'objects': {
                oid: {
                    'type': obj.properties.obj_type.name,
                    'position': obj.position,
                    'angle': obj.angle,
                }
                for oid, obj in self.objects.items()
            },
        }

    def setup_default_kitchen(self):
        """设置默认厨房布局"""
        # 冰箱 (左上角)
        self.add_static_object('coffee_machine', (1.5, 6.5))
        self.add_static_object('microwave', (3.0, 6.5))

        # 水槽 (右上角)
        # 用静态物体模拟
        sink_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        sink_body.position = (8.0, 6.5)
        self.space.add(sink_body)
        sink_shape = pymunk.Circle(sink_body, 0.3)
        self.space.add(sink_shape)
        sink_entity = PhysicsEntity(
            sink_body, [sink_shape],
            ObjectProperties(
                name='Sink',
                obj_type=ObjectType.FURNITURE,
                material=MaterialType.CERAMIC,
                mass=10.0,
                radius=0.3,
                color=(200, 200, 220),
                grabbable=False,
            )
        )
        self.objects[sink_entity.id] = sink_entity
        self.entities[sink_entity.id] = sink_entity

        # 灶台 (中间)
        stove_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        stove_body.position = (5.0, 6.5)
        self.space.add(stove_body)
        stove_shape = pymunk.Circle(stove_body, 0.4)
        self.space.add(stove_shape)
        stove_entity = PhysicsEntity(
            stove_body, [stove_shape],
            ObjectProperties(
                name='Stove',
                obj_type=ObjectType.APPLIANCE,
                material=MaterialType.METAL,
                mass=20.0,
                radius=0.4,
                color=(60, 60, 60),
                grabbable=False,
            )
        )
        self.objects[stove_entity.id] = stove_entity
        self.entities[stove_entity.id] = stove_entity

        # 桌子 (中间)
        table_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        table_body.position = (5.0, 3.0)
        self.space.add(table_body)
        table_shape = pymunk.Circle(table_body, 0.8)
        self.space.add(table_shape)
        table_entity = PhysicsEntity(
            table_body, [table_shape],
            ObjectProperties(
                name='Table',
                obj_type=ObjectType.FURNITURE,
                material=MaterialType.WOOD,
                mass=50.0,
                radius=0.8,
                color=(139, 90, 43),
                grabbable=False,
            )
        )
        self.objects[table_entity.id] = table_entity
        self.entities[table_entity.id] = table_entity

        # 放置一些物体
        self.add_object('coffee_cup', (4.8, 3.2))
        self.add_object('plate', (5.2, 3.0))
        self.add_object('apple', (5.0, 2.5))
        self.add_object('bread', (4.5, 3.0))
        self.add_object('water_kettle', (7.0, 6.5))
        self.add_object('spoon', (5.1, 3.1))


# ============================================================================
# 8. 渲染系统
# ============================================================================

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    pygame = None


class KitchenRenderer:
    """厨房渲染器"""

    # 像素/米比例
    PIXELS_PER_METER = 80

    def __init__(self, kitchen: Kitchen, width: int = 800, height: int = 640):
        self.kitchen = kitchen
        self.width = width
        self.height = height

        if not PYGAME_AVAILABLE:
            raise ImportError("pygame not available. Install with: pip install pygame")

        # 初始化 pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("ATLAS Physical Kitchen")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

        # 颜色
        self.colors = {
            'background': (240, 240, 230),
            'floor': (220, 220, 210),
            'wall': (100, 100, 100),
            'robot': (0, 100, 255),
            'robot_direction': (0, 200, 255),
            'text': (0, 0, 0),
        }

    def _world_to_screen(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        """世界坐标转屏幕坐标"""
        x = int(pos[0] * self.PIXELS_PER_METER)
        y = self.height - int(pos[1] * self.PIXELS_PER_METER)
        return (x, y)

    def _screen_to_world(self, pos: Tuple[int, int]) -> Tuple[float, float]:
        """屏幕坐标转世界坐标"""
        x = pos[0] / self.PIXELS_PER_METER
        y = (self.height - pos[1]) / self.PIXELS_PER_METER
        return (x, y)

    def render(self):
        """渲染一帧"""
        if not PYGAME_AVAILABLE:
            return

        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        # 清屏
        self.screen.fill(self.colors['background'])

        # 绘制地板网格
        self._draw_grid()

        # 绘制物体
        for entity in self.kitchen.entities.values():
            self._draw_entity(entity)

        # 绘制机器人
        for robot in self.kitchen.robots.values():
            self._draw_robot(robot)

        # 绘制信息
        self._draw_info()

        # 更新显示
        pygame.display.flip()
        self.clock.tick(60)

        return True

    def _draw_grid(self):
        """绘制网格"""
        grid_color = (200, 200, 200)
        for x in range(0, self.width, self.PIXELS_PER_METER):
            pygame.draw.line(self.screen, grid_color, (x, 0), (x, self.height))
        for y in range(0, self.height, self.PIXELS_PER_METER):
            pygame.draw.line(self.screen, grid_color, (0, y), (self.width, y))

    def _draw_entity(self, entity: PhysicsEntity):
        """绘制实体"""
        pos = self._world_to_screen(entity.position)
        radius = int(entity.properties.radius * self.PIXELS_PER_METER)
        color = entity.properties.color

        # 绘制圆形
        pygame.draw.circle(self.screen, color, pos, radius)
        pygame.draw.circle(self.screen, (0, 0, 0), pos, radius, 2)

        # 绘制名称
        if entity.properties.name:
            text = self.font.render(entity.properties.name, True, self.colors['text'])
            text_rect = text.get_rect(center=(pos[0], pos[1] - radius - 10))
            self.screen.blit(text, text_rect)

    def _draw_robot(self, robot: Robot):
        """绘制机器人"""
        pos = self._world_to_screen(robot.position)
        radius = int(PhysicsConfig.ROBOT_RADIUS * self.PIXELS_PER_METER)

        # 绘制机器人主体
        pygame.draw.circle(self.screen, self.colors['robot'], pos, radius)
        pygame.draw.circle(self.screen, (0, 0, 0), pos, radius, 2)

        # 绘制方向
        angle = robot.angle
        end_x = pos[0] + int(math.cos(angle) * radius * 1.5)
        end_y = pos[1] - int(math.sin(angle) * radius * 1.5)
        pygame.draw.line(self.screen, self.colors['robot_direction'], pos, (end_x, end_y), 3)

        # 绘制机械臂
        if robot.arm:
            grip_pos = robot.arm.get_grip_position()
            grip_screen = self._world_to_screen(grip_pos)
            pygame.draw.line(self.screen, (255, 100, 0), pos, grip_screen, 2)
            pygame.draw.circle(self.screen, (255, 100, 0), grip_screen, 5)

    def _draw_info(self):
        """绘制信息"""
        info_text = f"Time: {self.kitchen.time:.1f}s | "
        info_text += f"Robots: {len(self.kitchen.robots)} | "
        info_text += f"Objects: {len(self.kitchen.objects)} | "
        info_text += f"FPS: {int(self.clock.get_fps())}"

        text_surface = self.font.render(info_text, True, self.colors['text'])
        self.screen.blit(text_surface, (10, 10))

    def close(self):
        """关闭渲染器"""
        if PYGAME_AVAILABLE:
            pygame.quit()


# ============================================================================
# 9. 任务系统
# ============================================================================

@dataclass
class TaskStep:
    """任务步骤"""
    action: str
    target: Optional[str] = None
    position: Optional[Tuple[float, float]] = None
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    completed: bool = False


@dataclass
class Task:
    """任务定义"""
    name: str
    description: str
    steps: List[TaskStep]
    reward: float = 100.0
    time_limit: float = 300.0  # 秒

    def __post_init__(self):
        self.current_step = 0
        self.completed = False
        self.failed = False
        self.elapsed_time = 0.0

    def get_current_step(self) -> Optional[TaskStep]:
        """获取当前步骤"""
        if self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None

    def complete_step(self) -> bool:
        """完成当前步骤"""
        if self.current_step < len(self.steps):
            self.steps[self.current_step].completed = True
            self.current_step += 1
            if self.current_step >= len(self.steps):
                self.completed = True
            return True
        return False

    def get_progress(self) -> float:
        """获取任务进度"""
        if self.completed:
            return 1.0
        if not self.steps:
            return 0.0
        return self.current_step / len(self.steps)


# 预定义任务库
TASK_LIBRARY = {
    'make_coffee': Task(
        name='Make Coffee',
        description='Make a cup of coffee',
        steps=[
            TaskStep(action='move_to', target='coffee_machine'),
            TaskStep(action='grab', target='coffee_cup'),
            TaskStep(action='place', target='coffee_machine'),
            TaskStep(action='press_button', target='coffee_machine'),
            TaskStep(action='wait', target='coffee_machine'),
        ],
        reward=100.0,
    ),
    'prepare_breakfast': Task(
        name='Prepare Breakfast',
        description='Prepare a simple breakfast',
        steps=[
            TaskStep(action='move_to', target='plate'),
            TaskStep(action='grab', target='plate'),
            TaskStep(action='place', target='table'),
            TaskStep(action='move_to', target='bread'),
            TaskStep(action='grab', target='bread'),
            TaskStep(action='place', target='plate'),
            TaskStep(action='move_to', target='apple'),
            TaskStep(action='grab', target='apple'),
            TaskStep(action='place', target='plate'),
        ],
        reward=150.0,
    ),
    'clean_table': Task(
        name='Clean Table',
        description='Clean the table by moving items to the sink',
        steps=[
            TaskStep(action='move_to', target='table'),
            TaskStep(action='grab', target='item_on_table'),
            TaskStep(action='move_to', target='sink'),
            TaskStep(action='place', target='sink'),
        ],
        reward=80.0,
    ),
}


# ============================================================================
# 10. 与 SSFR 的集成接口
# ============================================================================

class KitchenSSFRInterface:
    """厨房环境与 SSFR 的集成接口"""

    def __init__(self, kitchen: Kitchen):
        self.kitchen = kitchen

    def get_observation(self, robot_id: str) -> Dict[str, Any]:
        """获取机器人的观察（用于 SSFR）"""
        if robot_id not in self.kitchen.robots:
            return {}

        robot = self.kitchen.robots[robot_id]
        sensor_data = robot.get_sensor_reading()

        # 获取附近物体
        all_entities = list(self.kitchen.entities.values())
        sensor = SensorSystem(robot, self.kitchen.space)
        nearby = sensor.get_nearby_objects(all_entities, max_distance=3.0)

        # 构建观察
        observation = {
            'position': robot.position,
            'angle': robot.angle,
            'velocity': sensor_data.get('velocity', (0, 0)),
            'nearby_objects': [
                {
                    'type': obj.properties.obj_type.name,
                    'name': obj.properties.name,
                    'distance': dist,
                    'position': obj.position,
                }
                for obj, dist in nearby[:10]  # 最多10个
            ],
            'timestamp': self.kitchen.time,
        }

        return observation

    def encode_state_for_ssfr(self) -> Dict[str, Any]:
        """编码整个厨房状态用于 SSFR"""
        return {
            'robots': {
                rid: {
                    'position': r.position,
                    'angle': r.angle,
                    'velocity': (r.body.velocity.x, r.body.velocity.y),
                }
                for rid, r in self.kitchen.robots.items()
            },
            'objects': {
                oid: {
                    'type': o.properties.obj_type.name,
                    'position': o.position,
                }
                for oid, o in self.kitchen.objects.items()
            },
            'time': self.kitchen.time,
        }


# ============================================================================
# 11. 主程序入口
# ============================================================================

def create_demo_kitchen() -> Kitchen:
    """创建演示厨房"""
    kitchen = Kitchen(width=10.0, height=8.0)
    kitchen.setup_default_kitchen()

    # 添加机器人
    robot = kitchen.add_robot('Chef-1', (2.0, 2.0))

    return kitchen


def run_simulation(kitchen: Kitchen, duration: float = 10.0,
                   render: bool = True) -> Dict[str, Any]:
    """运行模拟"""
    renderer = None
    if render and PYGAME_AVAILABLE:
        try:
            renderer = KitchenRenderer(kitchen)
        except Exception as e:
            print(f"Warning: Could not initialize renderer: {e}")
            render = False

    running = True
    start_time = time.time()

    try:
        while running and (time.time() - start_time) < duration:
            # 物理步进
            kitchen.step()

            # 渲染
            if renderer:
                running = renderer.render()

            # 简单的机器人控制示例
            for robot in kitchen.robots.values():
                # 随机移动
                import random
                if random.random() < 0.02:
                    robot.move_forward()
                if random.random() < 0.02:
                    robot.turn_left()

    except KeyboardInterrupt:
        print("Simulation interrupted by user")

    finally:
        if renderer:
            renderer.close()

    return kitchen.get_state()


if __name__ == "__main__":
    print("ATLAS Physical Kitchen")
    print("=" * 50)
    print("Creating kitchen...")

    kitchen = create_demo_kitchen()
    print(f"Kitchen created: {kitchen.width}m x {kitchen.height}m")
    print(f"Robots: {len(kitchen.robots)}")
    print(f"Objects: {len(kitchen.objects)}")
    print()
    print("Running simulation for 10 seconds...")
    print("(Close window or press Ctrl+C to stop)")
    print()

    state = run_simulation(kitchen, duration=10.0, render=True)

    print(f"\nSimulation complete!")
    print(f"Total steps: {state['step_count']}")
    print(f"Final time: {state['time']:.1f}s")
