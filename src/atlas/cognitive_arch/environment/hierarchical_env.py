"""
Hierarchical Environment Implementation
层次化环境实现

基于IObjectEnvironment接口的具体实现
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import random

import sys
sys.path.append('..')
from interfaces import (
    IObjectEnvironment, Percept, Action, ActionResult
)


@dataclass
class Room:
    """房间定义"""
    id: str
    name: str
    x: int
    y: int
    width: int
    height: int
    room_type: str = "generic"
    features: np.ndarray = field(default_factory=lambda: np.zeros(8))

    def contains(self, x: float, y: float) -> bool:
        return (self.x <= x < self.x + self.width and
                self.y <= y < self.y + self.height)

    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)


@dataclass
class Door:
    """门定义"""
    id: str
    room_a: str
    room_b: str
    x: float
    y: float
    is_open: bool = True

    def other_side(self, room_id: str) -> str:
        return self.room_b if room_id == self.room_a else self.room_a


@dataclass
class Object:
    """物体定义"""
    id: str
    name: str
    obj_type: str
    x: float
    y: float
    room_id: str
    is_pickupable: bool = True
    is_usable: bool = True


class HierarchicalEnvironment(IObjectEnvironment):
    """
    层次化环境实现

    100x100网格，4房间，3门，多个物体
    """

    def __init__(self, width: int = 100, height: int = 100, seed: int = 42):
        random.seed(seed)
        self._width = width
        self._height = height
        self._state_dim = 24  # 16基础 + 8物体

        # 空间结构
        self._rooms: Dict[str, Room] = {}
        self._doors: Dict[str, Door] = {}
        self._room_map = np.zeros((height, width), dtype=int)
        self._room_id_map: Dict[int, str] = {}
        self._obstacles = np.zeros((height, width), dtype=bool)

        # 物体
        self._objects: Dict[str, Object] = {}
        self._inventory: List[str] = []

        # 世界特征
        self._world_features = np.zeros((height, width, 16))

        # 智能体状态
        self._agent_pos = np.array([50.0, 50.0])

        self._build_world()
        self._update_agent_region()

    # -------------------------------------------------------------------------
    # 世界构建
    # -------------------------------------------------------------------------

    def _build_world(self):
        """构建世界结构"""
        self._build_rooms()
        # 门现在已经在 _build_rooms 中构建
        self._build_walls()
        self._place_objects()

    def _build_rooms(self):
        """构建房间 - 修复后的布局确保门正确连接"""
        configs = [
            # id, name, x, y, width, height, type, color
            # 左侧 - 卧室 (与hallway在x=30处相邻)
            ("bedroom", "卧室", 5, 40, 25, 20, "bedroom", [0.8, 0.6, 0.5]),
            # 中央走廊 - 连接所有房间
            ("hallway", "走廊", 30, 45, 30, 10, "hallway", [0.7, 0.7, 0.6]),
            # 右侧 - 厨房 (与hallway在x=60处相邻)
            ("kitchen", "厨房", 60, 40, 25, 20, "kitchen", [0.5, 0.7, 0.8]),
            # 下方 - 储藏室 (与hallway在y=55处相邻)
            ("storage", "储藏室", 35, 55, 20, 20, "storage", [0.6, 0.6, 0.6]),
        ]

        for i, (rid, name, x, y, w, h, rtype, color) in enumerate(configs):
            room = Room(rid, name, x, y, w, h, rtype, np.array(color + [0.5] * 5))
            self._rooms[rid] = room
            self._room_id_map[i + 1] = rid

            # 标记房间地图
            self._room_map[y:y+h, x:x+w] = i + 1

            # 填充视觉特征
            for dy in range(h):
                for dx in range(w):
                    if 0 <= y+dy < self._height and 0 <= x+dx < self._width:
                        self._world_features[y+dy, x+dx, :3] = color
                        self._world_features[y+dy, x+dx, 3] = 0.5

        # 门位置 - 确保在相邻房间边界上
        door_configs = [
            ("d1", "hallway", "bedroom", 30, 50),   # x=30: hallway左侧 = bedroom右侧
            ("d2", "hallway", "kitchen", 59, 50),   # x=59: hallway右侧(30+30-1) = kitchen左侧(60-1)
            ("d3", "hallway", "storage", 45, 55),   # y=55: hallway下侧(45+10) = storage上侧
        ]

        for did, ra, rb, x, y in door_configs:
            self._doors[did] = Door(did, ra, rb, x, y)
            self._obstacles[int(y), int(x)] = False
            self._world_features[int(y), int(x), 12:14] = [1.0, 1.0]

    def _build_walls(self):
        """构建墙壁 - 在房间边界创建障碍，但门位置可通行"""
        for room in self._rooms.values():
            # 上下边界
            for x in range(room.x, room.x + room.width):
                self._obstacles[room.y, x] = True
                self._obstacles[min(room.y + room.height - 1, self._height-1), x] = True
            # 左右边界
            for y in range(room.y, room.y + room.height):
                self._obstacles[y, room.x] = True
                self._obstacles[y, min(room.x + room.width - 1, self._width-1)] = True

        # 清除门位置及其两侧（确保可以穿过门）
        for door in self._doors.values():
            x, y = int(door.x), int(door.y)
            # 清除门位置
            self._obstacles[y, x] = False
            # 清除相邻位置（允许穿过门进入相邻房间）
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self._width and 0 <= ny < self._height:
                    self._obstacles[ny, nx] = False

    def _place_objects(self):
        """放置物体"""
        configs = [
            ("cup", "杯子", "container", "bedroom"),
            ("key", "钥匙", "key", "storage"),
            ("water", "水", "food", "kitchen"),
            ("knife", "刀", "tool", "kitchen"),
        ]

        for oid, name, otype, room_id in configs:
            if room_id in self._rooms:
                room = self._rooms[room_id]
                x = random.uniform(room.x + 2, room.x + room.width - 2)
                y = random.uniform(room.y + 2, room.y + room.height - 2)
                self._objects[oid] = Object(oid, name, otype, x, y, room_id)
                self._world_features[int(y), int(x), 14:16] = [1.0, hash(otype) % 100 / 100]

    def _update_agent_region(self):
        """更新智能体区域缓存"""
        pass  # 实时计算

    # -------------------------------------------------------------------------
    # IEnvironment 实现
    # -------------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def agent_position(self) -> Tuple[float, float]:
        return (float(self._agent_pos[0]), float(self._agent_pos[1]))

    def get_percept(self) -> Percept:
        x, y = int(self._agent_pos[0]), int(self._agent_pos[1])

        if not (0 <= x < self._width and 0 <= y < self._height):
            return Percept(
                features=np.zeros(self._state_dim),
                position=self.agent_position,
                timestamp=datetime.now(),
                metadata={"oob": True}
            )

        # 基础特征
        local = self._world_features[y, x].copy()
        local += np.random.randn(16) * 0.05
        local = np.clip(local, 0, 1)

        # 物体特征
        nearby = self.get_objects_near(x, y, 3.0)
        obj_features = np.zeros(8)
        for i, obj in enumerate(nearby[:2]):
            dist = np.sqrt((obj.x - x)**2 + (obj.y - y)**2)
            weight = max(0, 1 - dist / 3)
            obj_features[i*4:(i+1)*4] = [
                obj.obj_type == 'container',
                obj.obj_type == 'tool',
                obj.obj_type == 'key',
                obj.obj_type == 'food'
            ]
            obj_features[i*4:(i+1)*4] = np.array(obj_features[i*4:(i+1)*4]) * weight

        features = np.concatenate([local, obj_features])

        return Percept(
            features=features,
            position=self.agent_position,
            timestamp=datetime.now(),
            metadata={
                "room": self.get_region_at(x, y),
                "nearby_objects": len(nearby)
            }
        )

    def step(self, action: Action) -> ActionResult:
        new_pos = self._agent_pos.copy()
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

        # 碰撞检测
        x_int, y_int = int(new_pos[0]), int(new_pos[1])
        if (0 <= x_int < self._width and 0 <= y_int < self._height and
            not self._obstacles[y_int, x_int]):
            self._agent_pos = new_pos
        else:
            reward -= 0.1

        percept = self.get_percept()
        reward += percept.features[3] * 0.3  # 基础奖励

        return ActionResult(percept, reward, False, {})

    def _handle_interact(self) -> float:
        """处理交互"""
        x, y = self._agent_pos
        reward = 0.0

        for obj in list(self._objects.values()):
            dist = np.sqrt((obj.x - x)**2 + (obj.y - y)**2)
            if dist < 2.0 and obj.is_pickupable and obj.id not in self._inventory:
                self._inventory.append(obj.id)
                obj.is_pickupable = False
                reward += 1.0

        return reward

    def reset(self) -> Percept:
        self._agent_pos = np.array([50.0, 50.0])
        self._inventory.clear()
        for obj in self._objects.values():
            obj.is_pickupable = True
        return self.get_percept()

    def is_valid_position(self, x: float, y: float) -> bool:
        x_int, y_int = int(x), int(y)
        if not (0 <= x_int < self._width and 0 <= y_int < self._height):
            return False
        return not self._obstacles[y_int, x_int]

    # -------------------------------------------------------------------------
    # ISpatialEnvironment 实现
    # -------------------------------------------------------------------------

    def get_region_at(self, x: float, y: float) -> Optional[str]:
        x_int, y_int = int(x), int(y)
        if 0 <= x_int < self._width and 0 <= y_int < self._height:
            idx = self._room_map[y_int, x_int]
            return self._room_id_map.get(idx)
        return None

    def navigate_toward(self,
                       from_pos: Tuple[float, float],
                       to_pos: Tuple[float, float]) -> Action:
        """简单的梯度导航"""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]

        # 检查障碍物
        fx, fy = int(from_pos[0]), int(from_pos[1])

        if abs(dx) > abs(dy):
            if dx > 0 and self.is_valid_position(fx + 1, fy):
                return Action.RIGHT
            elif dx < 0 and self.is_valid_position(fx - 1, fy):
                return Action.LEFT

        if dy > 0 and self.is_valid_position(fx, fy + 1):
            return Action.DOWN
        elif dy < 0 and self.is_valid_position(fx, fy - 1):
            return Action.UP

        # 尝试其他方向
        for action in [Action.RIGHT, Action.LEFT, Action.DOWN, Action.UP]:
            if action == Action.RIGHT and self.is_valid_position(fx + 1, fy):
                return Action.RIGHT
            elif action == Action.LEFT and self.is_valid_position(fx - 1, fy):
                return Action.LEFT
            elif action == Action.DOWN and self.is_valid_position(fx, fy + 1):
                return Action.DOWN
            elif action == Action.UP and self.is_valid_position(fx, fy - 1):
                return Action.UP

        return Action.STAY

    # -------------------------------------------------------------------------
    # IObjectEnvironment 实现
    # -------------------------------------------------------------------------

    def get_objects_near(self, x: float, y: float, radius: float) -> List[Object]:
        nearby = []
        for obj in self._objects.values():
            dist = np.sqrt((obj.x - x)**2 + (obj.y - y)**2)
            if dist <= radius:
                nearby.append(obj)
        nearby.sort(key=lambda o: np.sqrt((o.x - x)**2 + (o.y - y)**2))
        return nearby

    def get_inventory(self) -> List[str]:
        return self._inventory.copy()

    def can_interact(self, x: float, y: float) -> bool:
        for obj in self._objects.values():
            dist = np.sqrt((obj.x - x)**2 + (obj.y - y)**2)
            if dist < 2.0 and obj.is_pickupable:
                return True
        return False

    # -------------------------------------------------------------------------
    # 扩展API
    # -------------------------------------------------------------------------

    def get_room_center(self, room_id: str) -> Optional[Tuple[float, float]]:
        """获取房间中心"""
        room = self._rooms.get(room_id)
        if room:
            return room.center()
        return None

    def get_room(self, room_id: str) -> Optional[Room]:
        return self._rooms.get(room_id)

    def get_object(self, obj_id: str) -> Optional[Object]:
        return self._objects.get(obj_id)

    def get_all_rooms(self) -> Dict[str, Room]:
        return self._rooms.copy()
