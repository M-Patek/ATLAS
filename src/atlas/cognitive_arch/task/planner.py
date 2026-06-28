"""
Task Planning - 任务规划

子目标和任务管理
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

import sys
sys.path.append('..')
from interfaces import (
    ISubgoal, ITask, ITaskPlanner,
    IEnvironment, ISpatialEnvironment, IObjectEnvironment,
    Action
)


# ============================================================================
# 子目标实现
# ============================================================================

@dataclass
class Subgoal(ISubgoal):
    """子目标实现"""
    _id: str
    _type: str  # "goto", "find", "pickup", "use", "interact"
    _target: str
    _room: Optional[str] = None
    _is_completed: bool = False
    _max_attempts: int = 100
    _attempts: int = 0

    # 属性访问
    @property
    def id(self) -> str:
        return self._id

    @property
    def is_completed(self) -> bool:
        return self._is_completed

    def check_completion(self, env: IEnvironment, agent_state: Dict) -> bool:
        """检查是否完成"""
        self._attempts += 1

        if self._attempts > self._max_attempts:
            self._is_completed = True  # 放弃
            return True

        pos = agent_state.get('position', (0, 0))

        if self._type == "goto":
            # 检查是否到达目标房间
            if isinstance(env, ISpatialEnvironment):
                current = env.get_region_at(pos[0], pos[1])
                if current == self._target:
                    self._is_completed = True
                    return True

        elif self._type == "find":
            # 检查是否接近目标物体
            if isinstance(env, IObjectEnvironment):
                nearby = env.get_objects_near(pos[0], pos[1], 3.0)
                for obj in nearby:
                    if obj.name == self._target or obj.id == self._target:
                        self._is_completed = True
                        return True

        elif self._type == "pickup":
            # 检查是否在背包中
            if isinstance(env, IObjectEnvironment):
                if self._target in env.get_inventory():
                    self._is_completed = True
                    return True

        elif self._type == "use":
            # 检查是否使用成功
            if isinstance(env, IObjectEnvironment):
                if self._target in env.get_inventory():
                    self._is_completed = True
                    return True

        return self._is_completed

    def get_required_action(self, env: IEnvironment,
                           agent_state: Dict) -> Optional[Action]:
        """获取达成此子目标的动作建议"""
        pos = agent_state.get('position', (0, 0))

        if self._type == "goto":
            if isinstance(env, ISpatialEnvironment):
                # 获取目标房间中心
                target_center = env.get_room_center(self._target)
                if target_center:
                    # 使用环境的导航功能
                    action = env.navigate_toward(pos, target_center)
                    return action

        elif self._type in ["find", "pickup"]:
            if isinstance(env, IObjectEnvironment):
                nearby = env.get_objects_near(pos[0], pos[1], 20.0)
                for obj in nearby:
                    if obj.name == self._target or obj.id == self._target:
                        # 导航向物体
                        if isinstance(env, ISpatialEnvironment):
                            action = env.navigate_toward(pos, (obj.x, obj.y))
                            return action

        elif self._type == "interact":
            if isinstance(env, IObjectEnvironment):
                if env.can_interact(pos[0], pos[1]):
                    return Action.INTERACT

        return None


# ============================================================================
# 任务实现
# ============================================================================

class HierarchicalTask(ITask):
    """
    层次化任务实现
    """

    def __init__(self, name: str, description: str = ""):
        self._name = name
        self._description = description
        self._subgoals: List[Subgoal] = []
        self._current_idx: int = 0
        self._is_completed: bool = False
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None

    # 属性
    @property
    def name(self) -> str:
        return self._name

    @property
    def is_completed(self) -> bool:
        return self._is_completed

    # 修改方法
    def add_subgoal(self, subgoal: Subgoal) -> None:
        """添加子目标"""
        self._subgoals.append(subgoal)

    def get_current_subgoal(self) -> Optional[ISubgoal]:
        """获取当前子目标"""
        if self._current_idx < len(self._subgoals):
            return self._subgoals[self._current_idx]
        return None

    def advance(self) -> bool:
        """推进到下一个子目标"""
        if self._current_idx < len(self._subgoals):
            self._current_idx += 1

            if self._current_idx >= len(self._subgoals):
                self._is_completed = True
                self._end_time = datetime.now()
                return True  # 任务完成

        return False

    def update_progress(self, env: IEnvironment, agent_state: Dict) -> None:
        """更新任务进度"""
        if self._is_completed:
            return

        current = self.get_current_subgoal()
        if current and isinstance(current, Subgoal):
            if current.check_completion(env, agent_state):
                self.advance()

    def get_progress(self) -> Tuple[int, int]:
        """获取进度 (当前, 总数)"""
        return (self._current_idx, len(self._subgoals))

    def __repr__(self) -> str:
        status = "✓" if self._is_completed else f"进行中[{self._current_idx}/{len(self._subgoals)}]"
        return f"任务({self._name}): {status}"


# ============================================================================
# 任务规划器实现
# ============================================================================

class TaskPlanner(ITaskPlanner):
    """
    任务规划器

    负责任务执行协调
    """

    def __init__(self):
        self._current_task: Optional[HierarchicalTask] = None
        self._task_history: List[HierarchicalTask] = []
        self._step_count: int = 0

    def assign_task(self, task: ITask) -> None:
        """分配任务"""
        if isinstance(task, HierarchicalTask):
            self._current_task = task
            self._task_history.append(task)
            task._start_time = datetime.now()

    def get_next_action(self, percept, agent_state: Dict) -> Optional[Action]:
        """
        获取任务的下一步动作

        返回Action表示有任务建议，返回None表示任务无建议
        """
        if self._current_task is None or self._current_task.is_completed:
            return None

        # 更新进度
        env = agent_state.get('env')
        if env:
            self._current_task.update_progress(env, agent_state)

        # 检查是否刚完成任务
        if self._current_task.is_completed:
            return None

        # 获取当前子目标的动作建议
        subgoal = self._current_task.get_current_subgoal()
        if subgoal:
            action = subgoal.get_required_action(env, agent_state)
            if action:
                return action

        return None

    def get_current_task(self) -> Optional[ITask]:
        return self._current_task

    def get_task_state(self) -> Dict[str, Any]:
        """获取任务状态"""
        if not self._current_task:
            return {"active": False}

        current, total = self._current_task.get_progress()
        subgoal = self._current_task.get_current_subgoal()

        return {
            "active": True,
            "task_name": self._current_task.name,
            "progress": f"{current}/{total}",
            "is_completed": self._current_task.is_completed,
            "current_subgoal": subgoal.id if subgoal else None,
            "subgoal_type": subgoal._type if isinstance(subgoal, Subgoal) else None,
        }


# ============================================================================
# 任务工厂
# ============================================================================

class TaskFactory:
    """
    任务工厂

    用于创建预定义任务
    """

    @staticmethod
    def create_fetch_water() -> HierarchicalTask:
        """创建接水任务"""
        task = HierarchicalTask("接一杯水", "从卧室取杯子，到厨房接水")

        task.add_subgoal(Subgoal("sg1", "goto", "bedroom", "bedroom"))
        task.add_subgoal(Subgoal("sg2", "find", "cup", "bedroom"))
        task.add_subgoal(Subgoal("sg3", "pickup", "cup", "bedroom"))
        task.add_subgoal(Subgoal("sg4", "goto", "kitchen", "kitchen"))
        task.add_subgoal(Subgoal("sg5", "find", "water", "kitchen"))
        task.add_subgoal(Subgoal("sg6", "use", "cup", "kitchen"))

        return task

    @staticmethod
    def create_find_key() -> HierarchicalTask:
        """创建找钥匙任务"""
        task = HierarchicalTask("找钥匙", "去储藏室找到钥匙")

        task.add_subgoal(Subgoal("sg1", "goto", "storage", "storage"))
        task.add_subgoal(Subgoal("sg2", "find", "key", "storage"))
        task.add_subgoal(Subgoal("sg3", "pickup", "key", "storage"))

        return task
