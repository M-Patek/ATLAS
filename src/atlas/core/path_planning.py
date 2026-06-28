"""
障碍物感知的路径规划工具

核心设计：
- A* 搜索，使用认知空间的距离作为代价
- 支持动态障碍（每步重新规划）
- 与 CognitiveSpace 接口兼容
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import heapq


def astar_path(space,
                start: Tuple[int, int],
                goal: Tuple[int, int],
                obstacles: Set[Tuple[int, int]],
                width: int,
                height: int,
                max_steps: int = 1000) -> List[Tuple[int, int]]:
    """
    A* 路径搜索，使用认知空间的距离

    Args:
        space: CognitiveSpace 实例，提供 compute_distance 和 get_heuristic
        start: 起点
        goal: 终点
        obstacles: 障碍物集合
        width: 网格宽度
        height: 网格高度
        max_steps: 最大搜索步数

    Returns:
        路径点列表（包含起点和终点），如果不可达则返回空列表
    """
    if start == goal:
        return [start]

    # 检查目标是否被阻挡
    if goal in obstacles:
        return []

    # A* 数据结构
    open_set = [(space.get_heuristic(start, goal), 0, start)]  # (f, g, pos)
    came_from = {}
    g_score = {start: 0.0}
    visited = set()
    step = 0

    while open_set and step < max_steps:
        step += 1
        _, current_g, current = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            # 重建路径
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        # 扩展邻居
        for nx, ny in _neighbors_4(current, width, height):
            neighbor = (nx, ny)
            if neighbor in obstacles or neighbor in visited:
                continue

            # 使用空间的距离
            tentative_g = current_g + space.compute_distance(current, neighbor)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + space.get_heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, tentative_g, neighbor))

    return []  # 不可达


def greedy_step(space,
                position: Tuple[int, int],
                goal: Tuple[int, int],
                obstacles: Set[Tuple[int, int]],
                width: int,
                height: int) -> Optional[Tuple[int, int]]:
    """
    贪心单步选择（考虑障碍物）

    返回最佳下一步位置，如果没有可行动作则返回 None
    """
    best_pos = None
    best_score = float('inf')

    for nx, ny in _neighbors_4(position, width, height):
        neighbor = (nx, ny)
        if neighbor in obstacles:
            continue

        # 使用空间的距离（考虑曲率等）
        dist = space.compute_distance(neighbor, goal)
        if dist < best_score:
            best_score = dist
            best_pos = neighbor

    return best_pos


def _neighbors_4(pos: Tuple[int, int], width: int, height: int) -> List[Tuple[int, int]]:
    """4连通邻居"""
    x, y = pos
    result = []
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            result.append((nx, ny))
    return result


def action_from_positions(current: Tuple[int, int],
                          next_pos: Tuple[int, int]) -> str:
    """
    从当前位置到下一步位置，返回动作字符串
    """
    dx = next_pos[0] - current[0]
    dy = next_pos[1] - current[1]

    if dx == 1 and dy == 0:
        return 'right'
    elif dx == -1 and dy == 0:
        return 'left'
    elif dx == 0 and dy == 1:
        return 'down'
    elif dx == 0 and dy == -1:
        return 'up'

    # 对角线或不动
    if dx > 0:
        return 'right'
    elif dx < 0:
        return 'left'
    elif dy > 0:
        return 'down'
    elif dy < 0:
        return 'up'

    return 'right'  # 默认
