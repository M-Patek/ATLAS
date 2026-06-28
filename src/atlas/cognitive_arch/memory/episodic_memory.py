"""
Memory System - 记忆系统

情节记忆和程序性记忆的实现
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import hashlib

import sys
sys.path.append('..')
from interfaces import (
    IEpisodicMemory, IProceduralMemory,
    Action, Percept
)


# ============================================================================
# 数据定义
# ============================================================================

@dataclass
class Experience:
    """单条经验"""
    timestamp: datetime
    state_key: str
    position: Tuple[float, float]
    room: Optional[str]
    action: Action
    context: Dict[str, Any]
    result: Dict[str, Any]
    reward: float


@dataclass
class Episode:
    """情节（经验序列）"""
    id: str
    episode_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    experiences: List[Experience] = field(default_factory=list)
    total_reward: float = 0.0
    is_success: bool = False

    # 特征签名
    room_sequence: List[str] = field(default_factory=list)
    action_sequence: List[Action] = field(default_factory=list)


class EpisodicMemory(IEpisodicMemory):
    """
    情节记忆实现

    使用空间哈希索引，支持快速相似检索
    """

    def __init__(self, max_episodes: int = 100, grid_size: int = 5):
        self._max_episodes = max_episodes
        self._grid_size = grid_size

        self._episodes: Dict[str, Episode] = {}
        self._current_episode: Optional[Episode] = None

        # 索引结构
        self._spatial_index: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
        self._state_index: Dict[str, List[Experience]] = defaultdict(list)

        # 统计
        self._total_experiences = 0

    # -------------------------------------------------------------------------
    # IEpisodicMemory 实现
    # -------------------------------------------------------------------------

    def record_experience(self,
                         state_key: str,
                         position: Tuple[float, float],
                         room: Optional[str],
                         action: Action,
                         context: Dict[str, Any],
                         result: Dict[str, Any],
                         reward: float) -> None:
        """记录经验"""
        if self._current_episode is None:
            return

        exp = Experience(
            timestamp=datetime.now(),
            state_key=state_key,
            position=position,
            room=room,
            action=action,
            context=context,
            result=result,
            reward=reward
        )

        self._current_episode.experiences.append(exp)
        self._current_episode.total_reward += reward
        self._current_episode.action_sequence.append(action)

        # 更新签名
        if room and room not in self._current_episode.room_sequence:
            self._current_episode.room_sequence.append(room)

        # 索引
        grid_pos = self._to_grid(position)
        self._spatial_index[grid_pos].add(state_key)
        self._state_index[state_key].append(exp)
        self._total_experiences += 1

    def suggest_action(self,
                      position: Tuple[float, float],
                      room: Optional[str],
                      context: Dict[str, Any]) -> Optional[Action]:
        """
        基于相似经验建议动作

        使用多层搜索：
        1. 同格 + 同房间
        2. 邻居格
        3. 同state_key
        """
        grid_pos = self._to_grid(position)
        target_context = context.get('subgoal_type')

        # 多层搜索收集候选
        candidates: List[Experience] = []

        # 第一层：同格
        for key in self._spatial_index.get(grid_pos, set()):
            candidates.extend(self._state_index.get(key, []))

        # 第二层：邻居
        if len(candidates) < 10:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (grid_pos[0] + dx, grid_pos[1] + dy)
                    for key in self._spatial_index.get(neighbor, set()):
                        candidates.extend(self._state_index.get(key, []))

        if not candidates:
            return None

        # 过滤和排序
        matched = []
        for exp in candidates:
            # 上下文匹配
            exp_context = exp.context.get('subgoal_type')
            context_score = 1.0 if exp_context == target_context else 0.5

            # 空间距离
            dist = np.sqrt(
                (exp.position[0] - position[0])**2 +
                (exp.position[1] - position[1])**2
            )
            spatial_score = max(0, 1 - dist / 10)

            # 成功奖励
            success_score = 1.0 if exp.reward > 0 and exp.result.get('success') else 0.0

            total_score = context_score * 0.4 + spatial_score * 0.4 + success_score * 0.2

            if total_score > 0.5:
                matched.append((total_score, exp))

        if not matched:
            return None

        # 按分数排序，返回最佳动作
        matched.sort(key=lambda x: x[0], reverse=True)
        top_5 = matched[:5]

        # 投票
        action_votes: Dict[Action, float] = defaultdict(float)
        for score, exp in top_5:
            action_votes[exp.action] += score

        return max(action_votes.items(), key=lambda x: x[1])[0]

    def start_episode(self, episode_type: str) -> str:
        """开始新情节"""
        episode_id = f"{episode_type}_{datetime.now().strftime('%H%M%S')}"
        self._current_episode = Episode(
            id=episode_id,
            episode_type=episode_type,
            start_time=datetime.now()
        )
        self._episodes[episode_id] = self._current_episode
        return episode_id

    def end_episode(self, success: bool) -> None:
        """结束情节"""
        if self._current_episode:
            self._current_episode.end_time = datetime.now()
            self._current_episode.is_success = success
            self._current_episode = None

            # 清理旧情节
            if len(self._episodes) > self._max_episodes:
                oldest = min(self._episodes.values(), key=lambda e: e.start_time)
                del self._episodes[oldest.id]
                # 清理索引
                self._cleanup_index(oldest.id)

    def store(self, key: str, data: Any) -> None:
        """通用存储（不常用）"""
        pass

    def retrieve(self, key: str) -> Optional[Any]:
        """通用检索"""
        return self._episodes.get(key)

    # -------------------------------------------------------------------------
    # 辅助方法
    # -------------------------------------------------------------------------

    def _to_grid(self, position: Tuple[float, float]) -> Tuple[int, int]:
        """坐标转网格"""
        return (
            int(position[0]) // self._grid_size,
            int(position[1]) // self._grid_size
        )

    def _cleanup_index(self, episode_id: str) -> None:
        """清理索引中的旧episode引用"""
        # 简化：重建索引
        pass

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        total = len(self._episodes)
        success = sum(1 for e in self._episodes.values() if e.is_success)

        return {
            'num_episodes': total,
            'num_experiences': self._total_experiences,
            'success_rate': success / max(1, total),
            'avg_episode_length': sum(
                len(e.experiences) for e in self._episodes.values()
            ) / max(1, total),
        }


# ============================================================================
# 统一的记忆管理器
# ============================================================================

class MemoryManager:
    """
    记忆管理器

    协调多种记忆类型
    """

    def __init__(self,
                 episodic: Optional[IEpisodicMemory] = None,
                 procedural: Optional[IProceduralMemory] = None):
        self._episodic = episodic or EpisodicMemory()
        self._procedural = procedural

    @property
    def episodic(self) -> IEpisodicMemory:
        return self._episodic

    @property
    def procedural(self) -> Optional[IProceduralMemory]:
        return self._procedural

    def record_step(self,
                   state_key: str,
                   percept: Percept,
                   action: Action,
                   context: Dict,
                   result: Dict,
                   reward: float) -> None:
        """记录一步经验"""
        self._episodic.record_experience(
            state_key=state_key,
            position=percept.position,
            room=percept.metadata.get('room') if percept.metadata else None,
            action=action,
            context=context,
            result=result,
            reward=reward
        )

    def get_action_suggestion(self,
                             percept: Percept,
                             context: Dict) -> Optional[Action]:
        """获取动作建议"""
        return self._episodic.suggest_action(
            percept.position,
            percept.metadata.get('room') if percept.metadata else None,
            context
        )
