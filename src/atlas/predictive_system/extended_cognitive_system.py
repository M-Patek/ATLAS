"""
Extended Cognitive System - 扩展认知系统

整合三个方向：
1. 扩空间 (Expanded Space): 100x100网格 + 房间/门/障碍物
2. 加层次 (Hierarchical Tasks): 子目标分解系统
3. 上记忆 (Episodic Memory): 经验存储与检索

核心设计：
- 空间分层：像素 → 物体 → 房间 → 建筑
- 意图分层：动作 → 子目标 → 任务 → 目标
- 记忆分层：快层缓存 → 情节记忆 → 语义记忆
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Deque, Set, Any
from enum import Enum, auto
from collections import deque, defaultdict
import random
import math
import hashlib
import json
from datetime import datetime

# Reuse core classes from multi_scale_predictive_system
import sys
sys.path.append('.')
from multi_scale_predictive_system import (
    Action, Percept, FastLayer,
    MotorIntention, GoalIntention, ValueIntention, IntentionField,
    Consequence, Interoception, MultiScalePredictiveAgent
)


# ============================================================================
# 第一层：扩展空间 - 层次化环境
# ============================================================================

@dataclass
class Room:
    """房间：空间的基本单元"""
    id: str
    name: str
    x: int  # 左上角x
    y: int  # 左上角y
    width: int
    height: int
    room_type: str = "generic"  # bedroom, kitchen, hallway, etc.
    features: np.ndarray = field(default_factory=lambda: np.zeros(8))

    def contains(self, x: float, y: float) -> bool:
        """检查坐标是否在房间内"""
        return (self.x <= x < self.x + self.width and
                self.y <= y < self.y + self.height)

    def center(self) -> Tuple[float, float]:
        """房间中心点"""
        return (self.x + self.width / 2, self.y + self.height / 2)

    def random_position(self) -> Tuple[float, float]:
        """房间内随机位置"""
        return (
            random.uniform(self.x + 1, self.x + self.width - 1),
            random.uniform(self.y + 1, self.y + self.height - 1)
        )


@dataclass
class Door:
    """门：连接房间的通道"""
    id: str
    room_a: str
    room_b: str
    x: float
    y: float
    is_open: bool = True
    required_action: Optional[str] = None  # 如 "USE_KEY", "PUSH"

    def connects(self, room_id: str) -> bool:
        """检查门是否连接给定房间"""
        return room_id in (self.room_a, self.room_b)

    def other_side(self, room_id: str) -> str:
        """获取门的另一侧房间"""
        return self.room_b if room_id == self.room_a else self.room_a


@dataclass
class Object:
    """环境中的物体"""
    id: str
    name: str
    type: str  # container, tool, key, etc.
    x: float
    y: float
    room_id: str
    properties: Dict[str, Any] = field(default_factory=dict)
    is_pickupable: bool = True
    is_usable: bool = True
    contained_objects: List[str] = field(default_factory=list)

    def get_feature_vector(self) -> np.ndarray:
        """获取物体特征向量"""
        # 编码物体类型
        type_encoding = {
            'container': [1, 0, 0, 0],
            'tool': [0, 1, 0, 0],
            'key': [0, 0, 1, 0],
            'food': [0, 0, 0, 1],
        }.get(self.type, [0, 0, 0, 0])
        return np.array(type_encoding + [float(self.is_pickupable), float(self.is_usable)])


class HierarchicalEnvironment:
    """
    层次化环境

    设计：原始世界 → 物体层 → 房间层 → 任务层
    """
    def __init__(self, width: int = 100, height: int = 100):
        self.width = width
        self.height = height

        # 原始世界（像素级）
        self.world_features = np.zeros((height, width, 16))  # RGB + depth + semantic + affordance
        self.obstacles = np.zeros((height, width), dtype=bool)  # 障碍物掩码

        # 物体层
        self.objects: Dict[str, Object] = {}
        self.object_positions: Dict[Tuple[int, int], str] = {}  # (x,y) -> object_id

        # 房间层
        self.rooms: Dict[str, Room] = {}
        self.doors: Dict[str, Door] = {}
        self.room_map = np.zeros((height, width), dtype=int)  # 每个格子属于哪个房间
        self.room_id_map: Dict[int, str] = {}  # room_index -> room_id

        # 智能体状态 - 初始位置在走廊
        self.agent_pos = np.array([50.0, 50.0])
        self.agent_room: Optional[str] = None
        self.inventory: List[str] = []

        # 初始化
        self._generate_hierarchical_world()
        self._update_agent_room()

    def _generate_hierarchical_world(self):
        """生成层次化世界结构"""
        # 创建房间布局： hallway + 3个房间
        room_configs = [
            ("hallway", "走廊", 40, 45, 20, 10, "hallway"),
            ("bedroom", "卧室", 10, 20, 25, 25, "bedroom"),
            ("kitchen", "厨房", 65, 20, 25, 25, "kitchen"),
            ("storage", "储藏室", 30, 70, 40, 20, "storage"),
        ]

        for i, (rid, name, x, y, w, h, rtype) in enumerate(room_configs):
            room = Room(rid, name, x, y, w, h, rtype)
            self.rooms[rid] = room
            self.room_id_map[i + 1] = rid

            # 标记房间区域
            self.room_map[y:y+h, x:x+w] = i + 1

            # 设置房间视觉特征
            room.features = np.random.rand(8) * 0.3 + 0.1
            room.features[0:3] = {
                'hallway': [0.7, 0.7, 0.6],
                'bedroom': [0.8, 0.6, 0.5],
                'kitchen': [0.5, 0.7, 0.8],
                'storage': [0.6, 0.6, 0.6],
            }[rtype]

            # 填充世界特征
            for dy in range(h):
                for dx in range(w):
                    if 0 <= y+dy < self.height and 0 <= x+dx < self.width:
                        self.world_features[y+dy, x+dx, :3] = room.features[0:3]
                        self.world_features[y+dy, x+dx, 3] = 0.5  # base activity
                        self.world_features[y+dy, x+dx, 4:12] = room.features

        # 创建门连接
        door_configs = [
            ("d1", "hallway", "bedroom", 40, 50),
            ("d2", "hallway", "kitchen", 60, 50),
            ("d3", "hallway", "storage", 55, 65),
        ]

        for did, ra, rb, x, y in door_configs:
            self.doors[did] = Door(did, ra, rb, x, y)
            # 门位置标记
            self.obstacles[int(y), int(x)] = False  # 确保门位置可通行
            self.world_features[int(y), int(x), 12:14] = [1.0, 1.0]  # door marker

        # 在房间中放置物体
        self._place_objects()

        # 设置障碍物（墙壁）
        self._build_walls()

    def _place_objects(self):
        """在房间中放置物体"""
        object_configs = [
            ("cup", "杯子", "container", "bedroom", {"contains": []}),
            ("key", "钥匙", "key", "storage", {"unlocks": "d2"}),
            ("water", "水", "food", "kitchen", {}),
            ("knife", "刀", "tool", "kitchen", {}),
            ("book", "书", "container", "bedroom", {}),
        ]

        for oid, name, otype, room_id, props in object_configs:
            if room_id in self.rooms:
                x, y = self.rooms[room_id].random_position()
                obj = Object(oid, name, otype, x, y, room_id, props)
                self.objects[oid] = obj
                self.object_positions[(int(x), int(y))] = oid
                # 在世界中标记物体
                self.world_features[int(y), int(x), 14:16] = [1.0, {
                    'container': 0.1, 'tool': 0.3, 'key': 0.5, 'food': 0.7
                }[otype]]

    def _build_walls(self):
        """构建墙壁（房间边界）"""
        for room in self.rooms.values():
            # 上下边界
            for x in range(room.x, room.x + room.width):
                self.obstacles[room.y, x] = True
                self.obstacles[min(room.y + room.height - 1, self.height-1), x] = True
            # 左右边界
            for y in range(room.y, room.y + room.height):
                self.obstacles[y, room.x] = True
                self.obstacles[y, min(room.x + room.width - 1, self.width-1)] = True

        # 清除门位置的墙壁
        for door in self.doors.values():
            self.obstacles[int(door.y), int(door.x)] = False

    def _update_agent_room(self):
        """更新智能体所在房间"""
        x, y = int(self.agent_pos[0]), int(self.agent_pos[1])
        if 0 <= x < self.width and 0 <= y < self.height:
            room_idx = self.room_map[y, x]
            if room_idx > 0:
                self.agent_room = self.room_id_map.get(room_idx)

    def get_percept(self, x: Optional[float] = None, y: Optional[float] = None) -> Percept:
        """获取多尺度感知"""
        if x is None:
            x, y = self.agent_pos

        x_int, y_int = int(x), int(y)

        # 基础局部感知（像素级）
        if not (0 <= x_int < self.width and 0 <= y_int < self.height):
            return Percept(
                local_features=np.zeros(16),
                proprioception=np.array([x, y]),
                prediction_error=1.0
            )

        local = self.world_features[y_int, x_int].copy()
        local += np.random.randn(16) * 0.05
        local = np.clip(local, 0, 1)

        # 添加物体信息（物体级）
        nearby_objects = self._get_nearby_objects(x, y, radius=3)
        object_features = np.zeros(8)
        for i, obj in enumerate(nearby_objects[:2]):  # 最多2个最近物体
            obj_feat = obj.get_feature_vector()
            dist = np.sqrt((obj.x - x)**2 + (obj.y - y)**2)
            weight = max(0, 1 - dist / 3)
            object_features[i*4:(i+1)*4] = obj_feat[:4] * weight

        local = np.concatenate([local, object_features])

        # 更新预测误差基于环境复杂度
        prediction_error = 0.05
        if nearby_objects:
            prediction_error += 0.1  # 物体增加不确定性

        return Percept(
            local_features=local,
            proprioception=np.array([x, y]),
            prediction_error=prediction_error
        )

    def _get_nearby_objects(self, x: float, y: float, radius: float) -> List[Object]:
        """获取附近物体"""
        nearby = []
        for obj in self.objects.values():
            dist = np.sqrt((obj.x - x)**2 + (obj.y - y)**2)
            if dist <= radius:
                nearby.append((dist, obj))
        nearby.sort(key=lambda x: x[0])
        return [obj for _, obj in nearby]

    def step(self, action: Action, agent) -> Tuple[Percept, float, bool]:
        """执行动作，返回感知、奖励、是否结束"""
        new_pos = self.agent_pos.copy()
        reward = 0.0

        if action == Action.UP:
            new_pos[1] -= 1
        elif action == Action.DOWN:
            new_pos[1] += 1
        elif action == Action.LEFT:
            new_pos[0] -= 1
        elif action == Action.RIGHT:
            new_pos[0] += 1
        elif action == Action.INTERACT:
            reward += self._handle_interact()

        # 检查碰撞
        x_int, y_int = int(new_pos[0]), int(new_pos[1])
        if (0 <= x_int < self.width and 0 <= y_int < self.height and
            not self.obstacles[y_int, x_int]):
            self.agent_pos = new_pos
            self._update_agent_room()
        else:
            reward -= 0.1  # 碰撞惩罚

        # 奖励信号
        percept = self.get_percept()
        reward += percept.local_features[3] * 0.5

        # 检查是否穿过门
        if self.agent_room:
            for door in self.doors.values():
                if door.connects(self.agent_room):
                    dist_to_door = np.sqrt((door.x - self.agent_pos[0])**2 +
                                          (door.y - self.agent_pos[1])**2)
                    if dist_to_door < 2.0 and door.is_open:
                        reward += 0.5  # 成功通过门

        return percept, reward, False

    def _handle_interact(self) -> float:
        """处理交互动作"""
        reward = 0.0
        x, y = self.agent_pos

        # 检查附近是否有物体
        for obj in self.objects.values():
            dist = np.sqrt((obj.x - x)**2 + (obj.y - y)**2)
            if dist < 2.0:
                if obj.is_pickupable and obj.id not in self.inventory:
                    self.inventory.append(obj.id)
                    obj.is_pickupable = False  # 已经被拾取
                    reward += 1.0
                    print(f"    [环境] 拾取了 {obj.name}")
                elif obj.is_usable:
                    reward += 0.5
                    print(f"    [环境] 使用了 {obj.name}")

        # 检查门交互
        for door in self.doors.values():
            dist = np.sqrt((door.x - x)**2 + (door.y - y)**2)
            if dist < 2.0 and not door.is_open:
                # 检查是否有钥匙
                has_key = any(self.objects.get(kid, Object("", "", "", 0, 0, "")).type == "key"
                             for kid in self.inventory)
                if door.required_action is None or has_key:
                    door.is_open = True
                    reward += 2.0
                    print(f"    [环境] 打开了通往 {door.other_side(self.agent_room)} 的门")

        return reward

    def get_room_of_position(self, x: float, y: float) -> Optional[str]:
        """获取坐标所属房间"""
        x_int, y_int = int(x), int(y)
        if 0 <= x_int < self.width and 0 <= y_int < self.height:
            room_idx = self.room_map[y_int, x_int]
            return self.room_id_map.get(room_idx)
        return None


# ============================================================================
# 第二层：层级任务系统
# ============================================================================

@dataclass
class Subgoal:
    """子目标"""
    id: str
    type: str  # "goto", "find", "pickup", "place", "use", "interact"
    target: str  # 房间名、物体名等
    room: Optional[str] = None  # 期望在哪个房间完成
    is_completed: bool = False
    max_attempts: int = 100
    attempts: int = 0

    def __repr__(self):
        status = "✓" if self.is_completed else "○"
        return f"[{status}] {self.type}:{self.target}@{self.room}"


class HierarchicalTask:
    """
    层次化任务

    示例：
    "接一杯水" → [
        goto(bedroom), find(cup), pickup(cup),
        goto(kitchen), find(water), use(cup, water)
    ]
    """
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.subgoals: List[Subgoal] = []
        self.current_subgoal_idx: int = 0
        self.is_completed: bool = False
        self.total_reward: float = 0.0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # 任务级意图（超慢层）- 使用ValueIntention
        self.task_intention = ValueIntention(
            exploration_vs_exploitation=0.3,
            risk_tolerance=0.5,
            coherence_threshold=0.7
        )

    def add_subgoal(self, subgoal: Subgoal):
        """添加子目标"""
        self.subgoals.append(subgoal)

    def get_current_subgoal(self) -> Optional[Subgoal]:
        """获取当前子目标"""
        if self.current_subgoal_idx < len(self.subgoals):
            return self.subgoals[self.current_subgoal_idx]
        return None

    def advance(self) -> bool:
        """推进到下一个子目标"""
        if self.current_subgoal_idx < len(self.subgoals):
            self.subgoals[self.current_subgoal_idx].is_completed = True
            self.current_subgoal_idx += 1

            if self.current_subgoal_idx >= len(self.subgoals):
                self.is_completed = True
                self.end_time = datetime.now()
                return True  # 任务完成
        return False

    def check_subgoal_progress(self, env: HierarchicalEnvironment,
                               agent_pos: np.ndarray) -> Tuple[bool, float]:
        """
        检查当前子目标进展
        返回：(是否完成, 进展度0-1)
        """
        subgoal = self.get_current_subgoal()
        if subgoal is None:
            return True, 1.0

        x, y = agent_pos
        progress = 0.0
        completed = False

        if subgoal.type == "goto":
            # 检查是否到达目标房间
            current_room = env.get_room_of_position(x, y)
            if current_room == subgoal.target:
                completed = True
                progress = 1.0
            elif subgoal.target in env.rooms:
                room = env.rooms[subgoal.target]
                dist = np.sqrt((room.center()[0] - x)**2 + (room.center()[1] - y)**2)
                max_dist = np.sqrt(env.width**2 + env.height**2)
                progress = max(0, 1 - dist / max_dist)

        elif subgoal.type == "find":
            # 检查是否接近目标物体
            for obj in env.objects.values():
                if obj.name == subgoal.target or obj.id == subgoal.target:
                    dist = np.sqrt((obj.x - x)**2 + (obj.y - y)**2)
                    if dist < 3.0:
                        completed = True
                        progress = 1.0
                    else:
                        progress = max(0, 1 - dist / 20)
                    break

        elif subgoal.type == "pickup":
            # 检查是否在背包中
            if subgoal.target in env.inventory:
                completed = True
                progress = 1.0
            else:
                # 接近物体中
                for obj in env.objects.values():
                    if obj.name == subgoal.target or obj.id == subgoal.target:
                        dist = np.sqrt((obj.x - x)**2 + (obj.y - y)**2)
                        progress = max(0, 1 - dist / 10)
                        break

        elif subgoal.type == "use":
            # 使用物体
            if subgoal.target in env.inventory:
                completed = True
                progress = 1.0

        subgoal.attempts += 1
        if subgoal.attempts > subgoal.max_attempts:
            completed = True  # 放弃

        return completed, progress

    def __repr__(self):
        status = "✓完成" if self.is_completed else f"进行中[{self.current_subgoal_idx}/{len(self.subgoals)}]"
        return f"任务({self.name}): {status}\n  " + "\n  ".join(str(sg) for sg in self.subgoals)


# ============================================================================
# 第三层：情节记忆系统
# ============================================================================

@dataclass
class Experience:
    """单个经验条目（一次状态-动作-结果）"""
    timestamp: datetime
    state_key: str  # 状态哈希键
    position: Tuple[float, float]
    room: Optional[str]
    action: Action
    subgoal_type: Optional[str]  # 当时的子目标类型
    result: Dict[str, Any]  # 结果：奖励、新位置、是否成功等
    reward: float

    def to_vector(self) -> np.ndarray:
        """转换为向量表示用于相似度计算"""
        return np.array([
            self.position[0] / 100.0,
            self.position[1] / 100.0,
            hash(self.subgoal_type or "") % 100 / 100.0,
            self.reward,
            self.result.get('success', 0),
        ])


@dataclass
class Episode:
    """情节：一系列经验组成的一个完整任务执行"""
    id: str
    task_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    experiences: List[Experience] = field(default_factory=list)
    total_reward: float = 0.0
    is_success: bool = False

    # 情景签名：用于快速匹配类似场景
    room_sequence: List[str] = field(default_factory=list)
    key_objects_found: List[str] = field(default_factory=list)


class EpisodicMemory:
    """
    情节记忆系统

    功能：
    1. 存储经验序列
    2. 基于状态相似度检索
    3. 提供快速决策建议
    """
    def __init__(self, max_episodes: int = 100, similarity_threshold: float = 0.8):
        self.max_episodes = max_episodes
        self.similarity_threshold = similarity_threshold

        self.episodes: Dict[str, Episode] = {}
        self.experience_index: Dict[str, List[Experience]] = defaultdict(list)
        self.current_episode: Optional[Episode] = None

        # 状态到经验的快速查找（近似最近邻）
        self.spatial_hash: Dict[Tuple[int, int], Set[str]] = defaultdict(set)

        # 统计
        self.total_experiences = 0

    def start_episode(self, task_name: str) -> Episode:
        """开始一个新情节"""
        episode_id = f"{task_name}_{datetime.now().strftime('%H%M%S')}"
        self.current_episode = Episode(
            id=episode_id,
            task_name=task_name,
            start_time=datetime.now()
        )
        self.episodes[episode_id] = self.current_episode
        return self.current_episode

    def record_experience(self, position: Tuple[float, float], room: Optional[str],
                         action: Action, subgoal_type: Optional[str],
                         result: Dict[str, Any], reward: float):
        """记录一次经验"""
        if self.current_episode is None:
            return

        # 生成状态键
        state_key = self._hash_state(position, room, subgoal_type)

        exp = Experience(
            timestamp=datetime.now(),
            state_key=state_key,
            position=position,
            room=room,
            action=action,
            subgoal_type=subgoal_type,
            result=result,
            reward=reward
        )

        self.current_episode.experiences.append(exp)
        self.experience_index[state_key].append(exp)
        self.spatial_hash[(int(position[0])//5, int(position[1])//5)].add(state_key)
        self.total_experiences += 1
        self.current_episode.total_reward += reward

        # 更新情节信息
        if room and room not in self.current_episode.room_sequence:
            self.current_episode.room_sequence.append(room)

    def end_episode(self, is_success: bool):
        """结束当前情节"""
        if self.current_episode:
            self.current_episode.end_time = datetime.now()
            self.current_episode.is_success = is_success
            self.current_episode = None

            # 清理旧情节
            if len(self.episodes) > self.max_episodes:
                oldest = min(self.episodes.values(), key=lambda e: e.start_time)
                del self.episodes[oldest.id]

    def _hash_state(self, position: Tuple[float, float], room: Optional[str],
                   subgoal_type: Optional[str]) -> str:
        """生成状态哈希键（离散化用于近似匹配）"""
        # 位置离散化到5x5网格
        pos_key = f"{int(position[0])//5},{int(position[1])//5}"
        room_key = room or "unknown"
        goal_key = subgoal_type or "none"
        return f"{pos_key}|{room_key}|{goal_key}"

    def retrieve_similar_experiences(self, position: Tuple[float, float],
                                     room: Optional[str],
                                     subgoal_type: Optional[str],
                                     k: int = 5) -> List[Experience]:
        """
        检索相似经验

        使用两层搜索：
        1. 空间哈希快速筛选
        2. 详细特征匹配排序
        """
        state_key = self._hash_state(position, room, subgoal_type)

        # 第一层：同状态键的经验
        candidates = list(self.experience_index.get(state_key, []))

        # 第二层：邻域搜索
        if len(candidates) < k:
            grid_x, grid_y = int(position[0])//5, int(position[1])//5
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    neighbor_keys = self.spatial_hash.get((grid_x+dx, grid_y+dy), set())
                    for key in neighbor_keys:
                        if key != state_key:
                            candidates.extend(self.experience_index.get(key, []))

        # 计算相似度并排序
        query_vec = np.array([
            position[0] / 100.0,
            position[1] / 100.0,
            hash(subgoal_type or "") % 100 / 100.0,
        ])

        scored = []
        for exp in candidates:
            exp_vec = np.array([
                exp.position[0] / 100.0,
                exp.position[1] / 100.0,
                hash(exp.subgoal_type or "") % 100 / 100.0,
            ])
            similarity = 1 - np.linalg.norm(query_vec - exp_vec) / np.sqrt(3)
            if similarity >= self.similarity_threshold:
                scored.append((similarity, exp))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored[:k]]

    def suggest_action(self, position: Tuple[float, float],
                      room: Optional[str],
                      subgoal_type: Optional[str]) -> Optional[Action]:
        """
        基于情节记忆建议动作

        返回：建议的动作，或None如果没有可靠记忆
        """
        similar = self.retrieve_similar_experiences(position, room, subgoal_type, k=10)

        if not similar:
            return None

        # 统计成功动作
        action_scores: Dict[Action, float] = defaultdict(float)
        for exp in similar:
            if exp.result.get('success', False) and exp.reward > 0:
                action_scores[exp.action] += exp.reward

        if action_scores:
            return max(action_scores.items(), key=lambda x: x[1])[0]
        return None

    def get_memory_stats(self) -> Dict:
        """获取记忆统计"""
        return {
            'num_episodes': len(self.episodes),
            'num_experiences': self.total_experiences,
            'success_rate': sum(1 for e in self.episodes.values() if e.is_success) / max(1, len(self.episodes)),
            'avg_episode_length': sum(len(e.experiences) for e in self.episodes.values()) / max(1, len(self.episodes)),
        }


# ============================================================================
# 整合：增强型多尺度智能体
# ============================================================================

class ExtendedCognitiveAgent:
    """
    扩展认知智能体

    整合：
    - 基础多尺度预测系统 (Fast/Slow层)
    - 层级任务管理
    - 情节记忆
    """
    def __init__(self, env: HierarchicalEnvironment):
        self.env = env

        # 基础认知系统 - 使用24维感知（16基础 + 8物体）
        self.base_agent = MultiScalePredictiveAgent(env, perception_dim=24, learning_rate=0.1)

        # 层级任务
        self.current_task: Optional[HierarchicalTask] = None
        self.task_history: List[HierarchicalTask] = []

        # 情节记忆
        self.episodic_memory = EpisodicMemory()

        # 任务相关的慢层覆盖（任务改变时重新计算）
        self.task_specific_field: Optional['PredictiveField'] = None

        # 统计
        self.step_count = 0
        self.episode_count = 0

    def assign_task(self, task: HierarchicalTask):
        """分配新任务"""
        self.current_task = task
        self.task_history.append(task)
        task.start_time = datetime.now()

        # 开始情节记忆
        self.episodic_memory.start_episode(task.name)

        # 根据任务调整预测场
        self._setup_task_predictive_field(task)

        print(f"\n[任务分配] {task.name}")
        print(f"  子目标: {len(task.subgoals)}个")
        for i, sg in enumerate(task.subgoals):
            print(f"    {i+1}. {sg.type}({sg.target})")
        print()

    def _setup_task_predictive_field(self, task: HierarchicalTask):
        """根据任务设置预测场"""
        # 任务目标影响预测场的探索区域
        goal_rooms = set()
        for sg in task.subgoals:
            if sg.room:
                goal_rooms.add(sg.room)
            if sg.type == "goto":
                goal_rooms.add(sg.target)

        # 增强目标房间的预测场
        for room_id in goal_rooms:
            if room_id in self.env.rooms:
                room = self.env.rooms[room_id]
                cx, cy = int(room.center()[0]), int(room.center()[1])
                # 在基础预测场上叠加任务导向的场
                if hasattr(self.base_agent, 'mind'):
                    field = self.base_agent.mind.slow_field
                    # 增加目标区域的探索优先级
                    for dy in range(-5, 6):
                        for dx in range(-5, 6):
                            x, y = cx + dx, cy + dy
                            if 0 <= x < self.env.width and 0 <= y < self.env.height:
                                if field.visit_count[y, x] == 0:
                                    field.visit_count[y, x] = -5  # 负值表示鼓励探索

    def think_and_act_with_task(self) -> Tuple[Action, Dict]:
        """
        任务感知的思考与行动

        决策优先级：
        1. 检查当前子目标
        2. 尝试情节记忆
        3. 使用基础快慢层系统
        """
        meta = {'layer': 'slow', 'source': 'mcts'}

        # 获取当前子目标
        subgoal = self.current_task.get_current_subgoal() if self.current_task else None
        subgoal_type = subgoal.type if subgoal else None

        # 检查子目标是否完成
        if subgoal and self.current_task:
            completed, progress = self.current_task.check_subgoal_progress(
                self.env, self.env.agent_pos
            )

            if completed:
                print(f"  [子目标完成] {subgoal}")
                self.current_task.advance()
                subgoal = self.current_task.get_current_subgoal()
                if subgoal:
                    print(f"  [新子目标] {subgoal}")
                else:
                    print(f"  [任务完成] {self.current_task.name}")
                    self.episodic_memory.end_episode(is_success=True)

        # 尝试情节记忆
        action = self.episodic_memory.suggest_action(
            tuple(self.env.agent_pos),
            self.env.agent_room,
            subgoal_type
        )

        if action:
            meta['source'] = 'episodic_memory'
            # 记录使用记忆
            meta['subgoal'] = subgoal_type
            return action, meta

        # 使用基础系统
        action, meta = self.base_agent.think_and_act()

        # 根据子目标调整动作
        if subgoal:
            action = self._adjust_action_for_subgoal(action, subgoal)
            meta['subgoal'] = subgoal.type

        return action, meta

    def _adjust_action_for_subgoal(self, base_action: Action,
                                   subgoal: Subgoal) -> Action:
        """
        根据子目标调整动作 - 使用简化的梯度导航
        """
        if subgoal.type == "goto":
            # 导航到目标房间
            target_room = subgoal.target
            if target_room in self.env.rooms:
                room = self.env.rooms[target_room]
                cx, cy = room.center()
                ax, ay = self.env.agent_pos

                # 检查当前是否已经在目标房间
                current_room = self.env.get_room_of_position(ax, ay)
                if current_room == target_room:
                    return Action.STAY

                # 简化的导航：优先x方向，然后y方向
                dx = cx - ax
                dy = cy - ay

                # 如果距离远，选择主方向
                if abs(dx) > 0.5:
                    next_x = ax + (1 if dx > 0 else -1)
                    next_y = int(ay)
                    if (0 <= next_x < self.env.width and
                        0 <= next_y < self.env.height and
                        not self.env.obstacles[next_y, int(ax) + (1 if dx > 0 else -1)]):
                        return Action.RIGHT if dx > 0 else Action.LEFT

                if abs(dy) > 0.5:
                    next_x = int(ax)
                    next_y = ay + (1 if dy > 0 else -1)
                    if (0 <= next_x < self.env.width and
                        0 <= next_y < self.env.height and
                        not self.env.obstacles[int(ay) + (1 if dy > 0 else -1), int(ax)]):
                        return Action.DOWN if dy > 0 else Action.UP

                # 默认沿主方向
                if abs(dx) > abs(dy) and abs(dx) > 0.5:
                    return Action.RIGHT if dx > 0 else Action.LEFT
                elif abs(dy) > 0.5:
                    return Action.DOWN if dy > 0 else Action.UP

        elif subgoal.type in ["find", "pickup", "use"]:
            # 寻找/使用物体
            for obj in self.env.objects.values():
                if obj.name == subgoal.target or obj.id == subgoal.target:
                    ax, ay = self.env.agent_pos
                    dx = obj.x - ax
                    dy = obj.y - ay
                    dist = np.sqrt(dx**2 + dy**2)

                    if dist < 2.0:
                        return Action.INTERACT

                    # 导航到物体
                    if abs(dx) > abs(dy) and abs(dx) > 0.5:
                        return Action.RIGHT if dx > 0 else Action.LEFT
                    elif abs(dy) > 0.5:
                        return Action.DOWN if dy > 0 else Action.UP

        return base_action

    def execute_and_learn(self, action: Action, meta: Dict) -> Tuple[Percept, float, bool]:
        """执行动作并学习"""
        current_pos = self.env.agent_pos.copy()
        current_room = self.env.agent_room

        # 执行
        percept, reward, done = self.base_agent.execute_and_learn(action, meta)

        # 记录到情节记忆
        subgoal = self.current_task.get_current_subgoal() if self.current_task else None
        self.episodic_memory.record_experience(
            position=tuple(current_pos),
            room=current_room,
            action=action,
            subgoal_type=subgoal.type if subgoal else None,
            result={
                'new_position': tuple(self.env.agent_pos),
                'new_room': self.env.agent_room,
                'success': reward > 0,
            },
            reward=reward
        )

        self.step_count += 1

        return percept, reward, done

    def get_extended_state(self) -> Dict:
        """获取扩展状态"""
        base_state = self.base_agent.get_cognitive_state()

        return {
            **base_state,
            'task': {
                'name': self.current_task.name if self.current_task else None,
                'progress': f"{self.current_task.current_subgoal_idx}/{len(self.current_task.subgoals)}" if self.current_task else "N/A",
                'current_subgoal': str(self.current_task.get_current_subgoal()) if self.current_task else None,
            },
            'episodic_memory': self.episodic_memory.get_memory_stats(),
            'room': self.env.agent_room,
            'inventory': self.env.inventory,
        }


# ============================================================================
# 演示
# ============================================================================

def create_water_fetching_task() -> HierarchicalTask:
    """创建"接水"任务"""
    task = HierarchicalTask(
        name="接一杯水",
        description="从卧室取杯子，到厨房接水"
    )

    task.add_subgoal(Subgoal("sg1", "goto", "bedroom", room="bedroom"))
    task.add_subgoal(Subgoal("sg2", "find", "cup", room="bedroom"))
    task.add_subgoal(Subgoal("sg3", "pickup", "cup", room="bedroom"))
    task.add_subgoal(Subgoal("sg4", "goto", "kitchen", room="kitchen"))
    task.add_subgoal(Subgoal("sg5", "find", "water", room="kitchen"))
    task.add_subgoal(Subgoal("sg6", "use", "cup", room="kitchen"))

    return task


def run_extended_demo():
    """运行扩展认知系统演示"""
    print("=" * 70)
    print("扩展认知系统演示")
    print("=" * 70)
    print()
    print("特性:")
    print("  - 空间: 100x100网格 + 4个房间 + 3个门 + 5个物体")
    print("  - 任务: 分层子目标（接一杯水）")
    print("  - 记忆: 情节记忆加速复用")
    print()

    # 创建环境
    print("【环境初始化】")
    env = HierarchicalEnvironment(width=100, height=100)
    print(f"  环境大小: {env.width}x{env.height}")
    print(f"  房间数: {len(env.rooms)}")
    for rid, room in env.rooms.items():
        print(f"    - {room.name}({rid}): {room.width}x{room.height} @ ({room.x},{room.y})")
    print(f"  门数: {len(env.doors)}")
    print(f"  物体数: {len(env.objects)}")
    for oid, obj in env.objects.items():
        print(f"    - {obj.name}({oid}): {obj.type} in {obj.room_id}")
    print()

    # 创建智能体
    agent = ExtendedCognitiveAgent(env)

    # 运行第一阶段：无任务探索（建立基础感知）
    print("【阶段1】无目的探索（建立基础地图）")
    print("-" * 50)
    for step in range(50):
        percept = agent.base_agent.perceive()
        action, meta = agent.base_agent.think_and_act()
        percept, reward, done = agent.execute_and_learn(action, meta)

        if step % 20 == 0:
            pos = env.agent_pos
            print(f"  Step {step}: 位置({pos[0]:.0f}, {pos[1]:.0f}), 房间:{env.agent_room}, 层:{meta['layer']}")
    print()

    # 运行第二阶段：执行任务
    print("【阶段2】执行任务：接一杯水")
    print("-" * 50)

    task = create_water_fetching_task()
    agent.assign_task(task)

    total_reward = 0
    for step in range(50, 400):
        action, meta = agent.think_and_act_with_task()
        percept, reward, done = agent.execute_and_learn(action, meta)
        total_reward += reward

        if step % 50 == 0:
            state = agent.get_extended_state()
            print(f"\n[Step {step}]")
            print(f"  位置: ({env.agent_pos[0]:.0f}, {env.agent_pos[1]:.0f}) 房间: {state['room']}")
            print(f"  任务: {state['task']['progress']} 当前: {state['task']['current_subgoal']}")
            print(f"  动作: {action.name} | 来源: {meta.get('source', 'base')}")
            print(f"  背包: {state['inventory']}")
            print(f"  记忆: {state['episodic_memory']['num_experiences']}条经验")
            print(f"  累计奖励: {total_reward:.2f}")

        if task.is_completed:
            print(f"\n  [任务提前完成] Step {step}")
            break

    print("\n" + "=" * 70)
    print("第一阶段完成")
    mem_stats = agent.episodic_memory.get_memory_stats()
    print(f"  情节记忆: {mem_stats['num_experiences']}条经验, {mem_stats['num_episodes']}个情节")
    print(f"  累计奖励: {total_reward:.2f}")
    print("=" * 70)
    print()

    # 运行第三阶段：重复任务（测试情节记忆加速）
    print("【阶段3】重复任务（情节记忆加速测试）")
    print("-" * 50)

    # 重置位置到hallway
    env.agent_pos = np.array([50.0, 50.0])
    env.inventory = []

    task2 = create_water_fetching_task()
    agent.assign_task(task2)

    total_reward2 = 0
    memory_uses = 0

    for step in range(400, 700):
        action, meta = agent.think_and_act_with_task()
        if meta.get('source') == 'episodic_memory':
            memory_uses += 1
        percept, reward, done = agent.execute_and_learn(action, meta)
        total_reward2 += reward

        if step % 50 == 0:
            print(f"  Step {step}: {meta.get('source', 'base'):15s} | {action.name:10s} | 奖励:{total_reward2:.1f}")

        if task2.is_completed:
            print(f"  [任务完成] Step {step}")
            break

    print("\n" + "=" * 70)
    print("演示完成")
    print()
    print("对比:")
    print(f"  任务1步数: ~350 (探索+学习)")
    print(f"  任务2步数: ~{step-400} (记忆加速)")
    print(f"  记忆使用率: {memory_uses}/{step-400} ({memory_uses/max(1,step-400)*100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    run_extended_demo()
