"""
ATLAS Sandbox - 代码世界机器人模拟沙盒

核心概念：
- Robot: 携带组件的虚拟机器人
- Component: 可组合的模块化组件
- Sandbox: 沙盒环境（网格世界）
- Task: 需要特定组件完成的任务
- SSFR: 发现最优组件组合策略
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import uuid
import time


# ============================================================================
# 1. 组件系统
# ============================================================================

class ComponentType(Enum):
    """组件类型"""
    SENSOR = auto()      # 传感器（摄像头、雷达、GPS）
    ACTUATOR = auto()    # 执行器（轮子、机械臂、扬声器）
    PROCESSOR = auto()    # 处理器（CPU、GPU、FPGA）
    COMM = auto()        # 通信模块（WiFi、蓝牙、5G）
    BATTERY = auto()     # 电池
    ARMOR = auto()       # 装甲/防护
    TOOL = auto()        # 工具（钻头、抓手、扫描仪）


@dataclass
class Component:
    """机器人组件"""
    name: str
    type: ComponentType
    # 能力值（0-1）
    capabilities: Dict[str, float] = field(default_factory=dict)
    # 能耗（每步）
    energy_cost: float = 0.1
    # 重量
    weight: float = 1.0
    # 成本
    cost: float = 1.0
    # 耐久度
    durability: float = 1.0
    # 特殊效果
    effects: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.id = str(uuid.uuid4())[:8]

    def can_perform(self, capability: str) -> bool:
        """检查组件是否支持某能力"""
        return capability in self.capabilities and self.capabilities[capability] > 0.3

    def get_capability(self, capability: str) -> float:
        """获取能力值"""
        return self.capabilities.get(capability, 0.0)


# 预定义组件库
COMPONENT_LIBRARY = {
    # 传感器
    'basic_camera': Component(
        name='Basic Camera',
        type=ComponentType.SENSOR,
        capabilities={'vision': 0.5, 'color_recognition': 0.6, 'object_detection': 0.4},
        energy_cost=0.05, weight=0.5, cost=1.0
    ),
    'thermal_camera': Component(
        name='Thermal Camera',
        type=ComponentType.SENSOR,
        capabilities={'vision': 0.3, 'thermal_detection': 0.9, 'night_vision': 0.8},
        energy_cost=0.08, weight=0.6, cost=2.0
    ),
    'lidar': Component(
        name='LiDAR',
        type=ComponentType.SENSOR,
        capabilities={'distance_measurement': 0.9, 'obstacle_detection': 0.9, 'mapping': 0.7},
        energy_cost=0.1, weight=1.0, cost=3.0
    ),
    'gps': Component(
        name='GPS Module',
        type=ComponentType.SENSOR,
        capabilities={'positioning': 0.9, 'navigation': 0.7},
        energy_cost=0.03, weight=0.2, cost=1.5
    ),

    # 执行器
    'wheels': Component(
        name='Standard Wheels',
        type=ComponentType.ACTUATOR,
        capabilities={'movement': 0.7, 'speed': 0.6},
        energy_cost=0.1, weight=1.0, cost=1.0
    ),
    'tracks': Component(
        name='Tracked Wheels',
        type=ComponentType.ACTUATOR,
        capabilities={'movement': 0.6, 'rough_terrain': 0.8, 'speed': 0.4},
        energy_cost=0.15, weight=2.0, cost=2.0
    ),
    'mechanical_arm': Component(
        name='Mechanical Arm',
        type=ComponentType.ACTUATOR,
        capabilities={'grasping': 0.8, 'manipulation': 0.7, 'tool_use': 0.6},
        energy_cost=0.1, weight=1.5, cost=2.5
    ),
    'drill': Component(
        name='Drill',
        type=ComponentType.TOOL,
        capabilities={'drilling': 0.9, 'mining': 0.7, 'combat': 0.3},
        energy_cost=0.2, weight=2.0, cost=3.0
    ),

    # 处理器
    'basic_cpu': Component(
        name='Basic CPU',
        type=ComponentType.PROCESSOR,
        capabilities={'computation': 0.5, 'path_planning': 0.4, 'data_processing': 0.5},
        energy_cost=0.05, weight=0.3, cost=1.0
    ),
    'advanced_cpu': Component(
        name='Advanced CPU',
        type=ComponentType.PROCESSOR,
        capabilities={'computation': 0.8, 'path_planning': 0.7, 'data_processing': 0.8, 'ai_reasoning': 0.6},
        energy_cost=0.1, weight=0.5, cost=3.0
    ),

    # 通信
    'wifi': Component(
        name='WiFi Module',
        type=ComponentType.COMM,
        capabilities={'wireless_comm': 0.7, 'data_transfer': 0.8},
        energy_cost=0.03, weight=0.2, cost=0.5
    ),
    'radio': Component(
        name='Radio Module',
        type=ComponentType.COMM,
        capabilities={'long_range_comm': 0.8, 'emergency_signal': 0.9},
        energy_cost=0.05, weight=0.3, cost=1.0
    ),

    # 电池
    'small_battery': Component(
        name='Small Battery',
        type=ComponentType.BATTERY,
        capabilities={'energy_storage': 0.5},
        energy_cost=-0.5, weight=1.0, cost=1.0  # 负能耗 = 提供能量
    ),
    'large_battery': Component(
        name='Large Battery',
        type=ComponentType.BATTERY,
        capabilities={'energy_storage': 1.0},
        energy_cost=-1.0, weight=2.0, cost=2.0
    ),
}


# ============================================================================
# 2. 机器人系统
# ============================================================================

@dataclass
class Robot:
    """虚拟机器人"""
    name: str
    position: Tuple[int, int] = (0, 0)
    orientation: float = 0.0  # 角度（弧度）

    # 组件槽位
    components: List[Component] = field(default_factory=list)

    # 状态
    energy: float = 100.0
    max_energy: float = 100.0
    health: float = 100.0

    # 能力缓存（动态计算）
    _capability_cache: Dict[str, float] = field(default_factory=dict)
    _cache_stale: bool = True

    def __post_init__(self):
        self.id = str(uuid.uuid4())[:8]

    def add_component(self, component: Component) -> bool:
        """添加组件"""
        # 检查重量限制
        total_weight = sum(c.weight for c in self.components) + component.weight
        if total_weight > 10.0:  # 最大载重
            return False

        self.components.append(component)
        self._cache_stale = True
        return True

    def remove_component(self, component_name: str) -> bool:
        """移除组件"""
        for i, c in enumerate(self.components):
            if c.name == component_name:
                self.components.pop(i)
                self._cache_stale = True
                return True
        return False

    def get_capabilities(self) -> Dict[str, float]:
        """获取机器人整体能力（组件组合）"""
        if not self._cache_stale:
            return self._capability_cache

        capabilities = {}

        # 合并所有组件的能力
        for component in self.components:
            for cap, value in component.capabilities.items():
                if cap in capabilities:
                    # 同类能力取最大值（或累加）
                    capabilities[cap] = max(capabilities[cap], value)
                else:
                    capabilities[cap] = value

        self._capability_cache = capabilities
        self._cache_stale = False
        return capabilities

    def can_perform(self, capability: str) -> bool:
        """检查机器人是否具备某能力"""
        caps = self.get_capabilities()
        return capability in caps and caps[capability] > 0.3

    def get_capability(self, capability: str) -> float:
        """获取某能力的值"""
        caps = self.get_capabilities()
        return caps.get(capability, 0.0)

    def consume_energy(self, amount: float) -> bool:
        """消耗能量"""
        if self.energy >= amount:
            self.energy -= amount
            return True
        return False

    def recharge(self, amount: float):
        """充电"""
        self.energy = min(self.max_energy, self.energy + amount)

    def step(self) -> float:
        """每步消耗"""
        # 计算组件总耗能
        total_cost = sum(c.energy_cost for c in self.components)
        # 电池提供能量
        self.energy -= total_cost
        self.energy = max(0, self.energy)
        return total_cost

    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            'name': self.name,
            'position': self.position,
            'energy': self.energy,
            'health': self.health,
            'components': [c.name for c in self.components],
            'capabilities': self.get_capabilities(),
            'total_weight': sum(c.weight for c in self.components),
            'total_cost': sum(c.cost for c in self.components),
        }


# ============================================================================
# 3. 沙盒环境
# ============================================================================

class Sandbox:
    """沙盒环境"""

    def __init__(self, width: int = 30, height: int = 30):
        self.width = width
        self.height = height

        # 地形
        self.terrain = np.zeros((width, height), dtype=int)
        # 0: 平地, 1: 障碍, 2: 水域, 3: 山地, 4: 危险区

        # 物品/资源
        self.items: Dict[Tuple[int, int], List[str]] = {}

        # 机器人
        self.robots: Dict[str, Robot] = {}

        # 任务
        self.tasks: List['Task'] = []

        # 时间步
        self.step_count = 0

    def set_terrain(self, x: int, y: int, terrain_type: int):
        """设置地形"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.terrain[x, y] = terrain_type

    def add_robot(self, robot: Robot):
        """添加机器人"""
        self.robots[robot.id] = robot

    def add_task(self, task: 'Task'):
        """添加任务"""
        self.tasks.append(task)

    def get_terrain_at(self, position: Tuple[int, int]) -> int:
        """获取位置地形"""
        x, y = position
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.terrain[x, y]
        return 1  # 边界视为障碍

    def is_passable(self, position: Tuple[int, int], robot: Robot) -> bool:
        """检查位置是否可通过"""
        x, y = position
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False

        terrain = self.terrain[x, y]

        if terrain == 1:  # 障碍
            return False
        elif terrain == 2:  # 水域
            return robot.get_capability('water_traversal') > 0.5
        elif terrain == 3:  # 山地
            return robot.get_capability('rough_terrain') > 0.5
        elif terrain == 4:  # 危险区
            return robot.get_capability('armor') > 0.5

        return True

    def step(self):
        """环境步进"""
        for robot in self.robots.values():
            # 机器人消耗能量
            robot.step()

            # 地形影响
            terrain = self.get_terrain_at(robot.position)
            if terrain == 4:  # 危险区
                robot.health -= 5.0

        self.step_count += 1

    def get_state(self) -> Dict[str, Any]:
        """获取环境状态"""
        return {
            'step_count': self.step_count,
            'robots': {rid: r.get_status() for rid, r in self.robots.items()},
            'tasks': [t.get_status() for t in self.tasks],
        }


# ============================================================================
# 4. 任务系统
# ============================================================================

@dataclass
class Task:
    """任务定义"""
    name: str
    description: str
    # 所需能力
    required_capabilities: Dict[str, float] = field(default_factory=dict)
    # 目标位置
    target_position: Optional[Tuple[int, int]] = None
    # 奖励
    reward: float = 100.0
    # 时间限制
    time_limit: int = 100
    # 状态
    completed: bool = False
    # 进度
    progress: float = 0.0

    def check_completion(self, robot: Robot) -> bool:
        """检查任务是否完成"""
        caps = robot.get_capabilities()

        for cap, required_value in self.required_capabilities.items():
            actual_value = caps.get(cap, 0.0)
            if actual_value < required_value:
                return False

        # 检查位置
        if self.target_position and robot.position != self.target_position:
            return False

        return True

    def get_progress(self, robot: Robot) -> float:
        """获取任务进度"""
        if self.completed:
            return 1.0

        caps = robot.get_capabilities()
        progress = 0.0
        total = len(self.required_capabilities)

        for cap, required_value in self.required_capabilities.items():
            actual_value = caps.get(cap, 0.0)
            if actual_value >= required_value:
                progress += 1.0
            else:
                progress += actual_value / max(required_value, 0.01)

        return progress / total if total > 0 else 0.0

    def get_status(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'required_capabilities': self.required_capabilities,
            'completed': self.completed,
            'progress': self.progress,
            'reward': self.reward,
        }


# 预定义任务
TASK_LIBRARY = {
    'exploration': Task(
        name='Area Exploration',
        description='Explore the entire area',
        required_capabilities={'movement': 0.5, 'vision': 0.4},
        reward=50.0
    ),
    'rescue': Task(
        name='Search and Rescue',
        description='Find and rescue targets',
        required_capabilities={'movement': 0.6, 'thermal_detection': 0.7, 'grasping': 0.5},
        reward=150.0
    ),
    'mining': Task(
        name='Resource Mining',
        description='Mine resources from the ground',
        required_capabilities={'movement': 0.5, 'drilling': 0.8, 'mining': 0.7},
        reward=100.0
    ),
    'surveillance': Task(
        name='Area Surveillance',
        description='Monitor the area for threats',
        required_capabilities={'vision': 0.7, 'object_detection': 0.6, 'wireless_comm': 0.5},
        reward=80.0
    ),
    'construction': Task(
        name='Construction',
        description='Build structures',
        required_capabilities={'movement': 0.5, 'manipulation': 0.7, 'tool_use': 0.6},
        reward=120.0
    ),
    'combat': Task(
        name='Combat Patrol',
        description='Patrol and engage threats',
        required_capabilities={'movement': 0.6, 'vision': 0.5, 'combat': 0.8},
        reward=200.0
    ),
}


# ============================================================================
# 5. 任务分配器（使用 SSFR）
# ============================================================================

class TaskAllocator:
    """任务分配器 - 使用 SSFR 发现最优组件组合"""

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox
        self.task_history: List[Dict] = []

    def evaluate_robot_for_task(self, robot: Robot, task: Task) -> float:
        """评估机器人对任务的适配度"""
        # 能力匹配度
        match_score = task.get_progress(robot)

        # 能量充足度
        energy_ratio = robot.energy / robot.max_energy

        # 距离因素
        distance_penalty = 0.0
        if task.target_position:
            dx = robot.position[0] - task.target_position[0]
            dy = robot.position[1] - task.target_position[1]
            distance = abs(dx) + abs(dy)
            distance_penalty = distance / (self.sandbox.width + self.sandbox.height)

        # 综合评分
        score = (match_score * 0.5 +
                 energy_ratio * 0.3 +
                 (1.0 - distance_penalty) * 0.2)

        return score

    def allocate_task(self, robot: Robot, available_tasks: List[Task]) -> Optional[Task]:
        """为机器人分配最优任务"""
        if not available_tasks:
            return None

        best_task = None
        best_score = -1.0

        for task in available_tasks:
            if task.completed:
                continue

            score = self.evaluate_robot_for_task(robot, task)
            if score > best_score:
                best_score = score
                best_task = task

        return best_task

    def suggest_components(self, task: Task, robot: Robot) -> List[str]:
        """建议完成任务所需的组件"""
        suggestions = []

        for cap, required in task.required_capabilities.items():
            current = robot.get_capability(cap)
            if current < required:
                # 找到能提供该能力的组件
                for name, comp in COMPONENT_LIBRARY.items():
                    if comp.get_capability(cap) >= required:
                        suggestions.append(name)

        return suggestions


# ============================================================================
# 6. 示例用法
# ============================================================================

def create_example_scenario():
    """创建示例场景"""
    # 创建沙盒
    sandbox = Sandbox(width=20, height=20)

    # 设置地形
    for x in range(5, 10):
        for y in range(5, 10):
            sandbox.set_terrain(x, y, 2)  # 水域

    for x in range(12, 15):
        for y in range(12, 15):
            sandbox.set_terrain(x, y, 3)  # 山地

    # 创建机器人
    robot = Robot(name='Explorer-1', position=(1, 1))
    robot.add_component(COMPONENT_LIBRARY['basic_camera'])
    robot.add_component(COMPONENT_LIBRARY['wheels'])
    robot.add_component(COMPONENT_LIBRARY['basic_cpu'])
    robot.add_component(COMPONENT_LIBRARY['small_battery'])

    sandbox.add_robot(robot)

    # 添加任务
    sandbox.add_task(TASK_LIBRARY['exploration'])
    sandbox.add_task(TASK_LIBRARY['surveillance'])

    return sandbox


if __name__ == "__main__":
    sandbox = create_example_scenario()
    print("ATLAS Sandbox initialized")
    print(f"Robots: {len(sandbox.robots)}")
    print(f"Tasks: {len(sandbox.tasks)}")

    for rid, robot in sandbox.robots.items():
        print(f"\nRobot: {robot.name}")
        print(f"  Components: {[c.name for c in robot.components]}")
        print(f"  Capabilities: {robot.get_capabilities()}")
