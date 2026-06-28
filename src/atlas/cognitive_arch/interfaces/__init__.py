"""
Core Interfaces - 核心接口定义

定义认知架构的契约边界，所有实现必须遵守这些接口。
遵循依赖倒置原则：高层模块依赖接口，而非具体实现。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum
import numpy as np
from datetime import datetime


# ============================================================================
# 基础数据类型
# ============================================================================

class Action(Enum):
    """标准动作空间"""
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    INTERACT = 5


@dataclass(frozen=True)
class Percept:
    """感知数据，不可变"""
    features: np.ndarray
    position: Tuple[float, float]
    timestamp: datetime
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        # 确保features是只读的
        self.features.flags.writeable = False


@dataclass
class ActionResult:
    """动作执行结果"""
    percept: Percept
    reward: float
    done: bool
    info: Dict[str, Any] = None


# ============================================================================
# 环境接口
# ============================================================================

class IEnvironment(ABC):
    """
    环境接口

    责任：提供感知数据，执行动作，维护世界状态
    """

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """状态空间维度"""
        pass

    @property
    @abstractmethod
    def agent_position(self) -> Tuple[float, float]:
        """智能体当前位置"""
        pass

    @abstractmethod
    def get_percept(self) -> Percept:
        """获取当前感知"""
        pass

    @abstractmethod
    def step(self, action: Action) -> ActionResult:
        """执行动作，返回结果"""
        pass

    @abstractmethod
    def reset(self) -> Percept:
        """重置环境"""
        pass

    @abstractmethod
    def is_valid_position(self, x: float, y: float) -> bool:
        """检查位置是否有效"""
        pass


class ISpatialEnvironment(IEnvironment, ABC):
    """
    空间环境接口（扩展）

    责任：提供空间信息支持导航
    """

    @abstractmethod
    def get_region_at(self, x: float, y: float) -> Optional[str]:
        """获取位置所属区域（房间/区域ID）"""
        pass

    @abstractmethod
    def get_room_center(self, room_id: str) -> Optional[Tuple[float, float]]:
        """获取房间中心坐标"""
        pass

    @abstractmethod
    def navigate_toward(self, from_pos: Tuple[float, float],
                       to_pos: Tuple[float, float]) -> Action:
        """提供导航建议（可选的高级功能）"""
        pass


class IObjectEnvironment(ISpatialEnvironment, ABC):
    """
    支持物体的环境接口

    责任：管理环境中的物体
    """

    @abstractmethod
    def get_objects_near(self, x: float, y: float, radius: float) -> List[Any]:
        """获取附近物体"""
        pass

    @abstractmethod
    def get_inventory(self) -> List[str]:
        """获取智能体背包"""
        pass

    @abstractmethod
    def can_interact(self, x: float, y: float) -> bool:
        """检查当前位置是否可以交互"""
        pass


# ============================================================================
# 认知接口
# ============================================================================

class ICognitiveLayer(ABC):
    """
    认知层接口

    责任：根据感知产生动作决策
    """

    @abstractmethod
    def process(self, percept: Percept, context: Dict[str, Any]) -> Tuple[Action, Dict]:
        """
        处理感知，产生动作

        Returns:
            (动作, 元数据)
        """
        pass

    @abstractmethod
    def learn(self, experience: Dict[str, Any]) -> None:
        """从经验中学习"""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """获取层状态"""
        pass


class IFastLayer(ICognitiveLayer, ABC):
    """快层接口：透明执行，自动化"""
    pass


class ISlowLayer(ICognitiveLayer, ABC):
    """慢层接口：意识升起，搜索推理"""
    pass


# ============================================================================
# 记忆接口
# ============================================================================

class IMemory(ABC):
    """基础记忆接口"""

    @abstractmethod
    def store(self, key: str, data: Any) -> None:
        """存储记忆"""
        pass

    @abstractmethod
    def retrieve(self, key: str) -> Optional[Any]:
        """检索记忆"""
        pass


class IEpisodicMemory(IMemory, ABC):
    """
    情节记忆接口

    责任：存储和检索经验序列
    """

    @abstractmethod
    def record_experience(self,
                         state_key: str,
                         position: Tuple[float, float],
                         room: Optional[str],
                         action: Action,
                         context: Dict[str, Any],
                         result: Dict[str, Any],
                         reward: float) -> None:
        """记录一次经验"""
        pass

    @abstractmethod
    def suggest_action(self,
                      position: Tuple[float, float],
                      room: Optional[str],
                      context: Dict[str, Any]) -> Optional[Action]:
        """基于记忆建议动作"""
        pass

    @abstractmethod
    def start_episode(self, episode_type: str) -> str:
        """开始新情节，返回episode_id"""
        pass

    @abstractmethod
    def end_episode(self, success: bool) -> None:
        """结束当前情节"""
        pass


class IProceduralMemory(IMemory, ABC):
    """
    程序性记忆（快层缓存）

    责任：存储自动化技能
    """

    @abstractmethod
    def get_skill(self, context_key: str) -> Optional[Callable]:
        """获取自动化技能"""
        pass

    @abstractmethod
    def consolidate(self, context_key: str, skill: Callable, confidence: float) -> None:
        """固化技能"""
        pass


# ============================================================================
# 任务接口
# ============================================================================

class ISubgoal(ABC):
    """子目标接口"""

    @property
    @abstractmethod
    def id(self) -> str:
        pass

    @property
    @abstractmethod
    def is_completed(self) -> bool:
        pass

    @abstractmethod
    def check_completion(self, env: IEnvironment, agent_state: Dict) -> bool:
        """检查是否完成"""
        pass

    @abstractmethod
    def get_required_action(self, env: IEnvironment, agent_state: Dict) -> Optional[Action]:
        """获取达成此子目标建议的动作"""
        pass


class ITask(ABC):
    """任务接口"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def is_completed(self) -> bool:
        pass

    @abstractmethod
    def get_current_subgoal(self) -> Optional[ISubgoal]:
        """获取当前子目标"""
        pass

    @abstractmethod
    def advance(self) -> bool:
        """推进到下一个子目标，返回是否任务完成"""
        pass

    @abstractmethod
    def update_progress(self, env: IEnvironment, agent_state: Dict) -> None:
        """更新任务进度"""
        pass


class ITaskPlanner(ABC):
    """
    任务规划器接口

    责任：管理任务执行，协调子目标
    """

    @abstractmethod
    def assign_task(self, task: ITask) -> None:
        """分配任务"""
        pass

    @abstractmethod
    def get_next_action(self, percept: Percept, agent_state: Dict) -> Optional[Action]:
        """
        获取任务的下一步动作建议
        如果当前有活跃子目标，返回子目标导向的动作
        """
        pass

    @abstractmethod
    def get_current_task(self) -> Optional[ITask]:
        """获取当前任务"""
        pass


# ============================================================================
# 智能体接口
# ============================================================================

class ICognitiveAgent(ABC):
    """
    认知智能体接口

    责任：协调各认知模块，产生统一行动
    """

    @abstractmethod
    def perceive(self) -> Percept:
        """感知环境"""
        pass

    @abstractmethod
    def think(self, percept: Percept) -> Tuple[Action, Dict]:
        """思考决策"""
        pass

    @abstractmethod
    def act(self, action: Action) -> ActionResult:
        """执行动作"""
        pass

    @abstractmethod
    def learn(self, experience: Dict[str, Any]) -> None:
        """学习"""
        pass

    @abstractmethod
    def get_cognitive_state(self) -> Dict[str, Any]:
        """获取认知状态"""
        pass


class IExtendedCognitiveAgent(ICognitiveAgent, ABC):
    """扩展认知智能体（支持任务和记忆）"""

    @abstractmethod
    def assign_task(self, task: ITask) -> None:
        """分配任务"""
        pass

    @abstractmethod
    def get_episodic_memory_stats(self) -> Dict[str, Any]:
        """获取情节记忆统计"""
        pass
