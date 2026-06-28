"""
ATLAS Core: World Model
世界模型抽象

世界模型的核心职责: 根据观测更新认知空间

设计原则:
- World Model 持有对 CognitiveSpace 的引用
- 不直接参与决策，只更新空间
- 预测 = 预测空间将如何变化
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from collections import deque

from .space import CognitiveSpace


class WorldModel(ABC):
    """
    世界模型抽象基类

    世界模型负责:
    1. 解析观测数据
    2. 更新认知空间
    3. 预测未来状态（可选）

    关键洞察:
    不是预测"世界状态"，而是预测"哪里的信息有价值"
    """

    def __init__(self, space: CognitiveSpace):
        self.space = space
        self.observation_history: deque = deque(maxlen=1000)
        self.update_count = 0

    @abstractmethod
    def process_observation(self, position: Tuple[int, int],
                           observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理观测数据，提取对空间更新有用的信息

        Args:
            position: 观测发生的位置
            observation: 原始观测数据

        Returns:
            处理后的更新指令
        """
        pass

    def update(self, position: Tuple[int, int],
              observation: Dict[str, Any]) -> None:
        """
        执行完整的更新流程

        1. 记录观测历史
        2. 处理观测
        3. 更新空间
        """
        self.observation_history.append({
            'position': position,
            'observation': observation,
            'step': self.update_count
        })

        # 处理观测并更新空间
        processed = self.process_observation(position, observation)
        self.space.update_from_observation(position, processed)

        self.update_count += 1

    def predict(self, position: Tuple[int, int],
                action: Any) -> Optional[Dict[str, Any]]:
        """
        预测执行动作后的空间变化

        默认实现: 不预测（简单模型）
        子类可覆盖实现预测功能

        Returns:
            预测的空间变化，或 None
        """
        return None

    def get_history(self, n: int = 10) -> List[Dict]:
        """获取最近的 n 条观测历史"""
        return list(self.observation_history)[-n:]


class SimpleWorldModel(WorldModel):
    """
    简单的世界模型实现

    直接传递观测数据给空间
    """

    def process_observation(self, position: Tuple[int, int],
                           observation: Dict[str, Any]) -> Dict[str, Any]:
        """直接传递观测"""
        return observation


class UncertaintyWorldModel(WorldModel):
    """
    基于不确定度的世界模型

    规则:
    - 障碍物附近: uncertainty 增加
    - 访问过的位置: uncertainty 降低
    - 目标附近: uncertainty 降低
    """

    def __init__(self, space: CognitiveSpace,
                 obstacle_boost: float = 0.3,
                 visit_decay: float = 0.05,
                 goal_decay: float = 0.2):
        super().__init__(space)
        self.obstacle_boost = obstacle_boost
        self.visit_decay = visit_decay
        self.goal_decay = goal_decay

        # 跟踪访问位置
        self.visit_count = np.zeros((space.width, space.height), dtype=np.int32)

    def process_observation(self, position: Tuple[int, int],
                           observation: Dict[str, Any]) -> Dict[str, Any]:
        """处理观测，生成更新指令"""
        x, y = position

        # 更新访问计数
        if 0 <= x < self.space.width and 0 <= y < self.space.height:
            self.visit_count[x, y] += 1

        processed = observation.copy()

        # 添加访问信息
        processed['visit_count'] = self.visit_count[x, y]
        processed['familiarity'] = min(1.0, 0.1 * self.visit_count[x, y])

        return processed


class CuriosityWorldModel(WorldModel):
    """
    好奇心驱动的世界模型

    特别关注新颖性和信息增益
    """

    def __init__(self, space: CognitiveSpace,
                 novelty_weight: float = 1.0,
                 exploration_bonus: float = 0.1):
        super().__init__(space)
        self.novelty_weight = novelty_weight
        self.exploration_bonus = exploration_bonus

        self.visit_entropy = np.zeros((space.width, space.height))
        self.known_positions: set = set()

    def process_observation(self, position: Tuple[int, int],
                           observation: Dict[str, Any]) -> Dict[str, Any]:
        """处理观测，添加好奇心信号"""
        x, y = position

        # 计算新颖性
        is_novel = position not in self.known_positions
        self.known_positions.add(position)

        processed = observation.copy()
        processed['novelty'] = 1.0 if is_novel else 0.0
        processed['exploration_value'] = self._compute_exploration_value(position)

        return processed

    def _compute_exploration_value(self, position: Tuple[int, int]) -> float:
        """计算位置的探索价值"""
        x, y = position

        # 邻居的新颖性
        novel_neighbors = 0
        total_neighbors = 0

        from .space import neighbors_4
        for nx, ny in neighbors_4(position, self.space.width, self.space.height):
            total_neighbors += 1
            if (nx, ny) not in self.known_positions:
                novel_neighbors += 1

        if total_neighbors == 0:
            return 0.0

        return self.exploration_bonus * (novel_neighbors / total_neighbors)
