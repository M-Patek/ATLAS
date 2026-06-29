"""
ATLAS Core: Optimized Geodesic Solvers
高性能求解器实现

优化策略：
1. JPS (Jump Point Search) - 减少扩展节点数
2. 迭代加深A* (IDA*) - 内存优化
3. 启发式缓存 - 避免重复计算
"""

import heapq
import time
import numpy as np
from typing import List, Tuple, Optional, Set, Dict, Callable
from abc import ABC, abstractmethod

from .solver import SolverResult, GeodesicSolver
from .space import CognitiveSpace, neighbors_4


class JPSNode:
    """JPS节点"""
    __slots__ = ['x', 'y', 'g', 'f', 'parent', 'direction']

    def __init__(self, x: int, y: int, g: float = 0.0, f: float = 0.0,
                 parent: Optional['JPSNode'] = None,
                 direction: Optional[Tuple[int, int]] = None):
        self.x = x
        self.y = y
        self.g = g
        self.f = f
        self.parent = parent
        self.direction = direction

    def __lt__(self, other):
        return self.f < other.f

    def pos(self) -> Tuple[int, int]:
        return (self.x, self.y)


class JPSGeodesicSolver(GeodesicSolver):
    """
    Jump Point Search (JPS) 求解器

    JPS通过识别"跳点"来减少需要扩展的节点数量：
    - 在开放区域直接跳跃，不扩展中间节点
    - 只在关键位置（跳点）进行扩展

    对于几何空间（如Ricci、Conformal），可以显著减少节点扩展数。
    """

    def __init__(self, space: CognitiveSpace,
                 use_diagonal: bool = False,
                 max_iterations: Optional[int] = None):
        super().__init__(space, use_diagonal, max_iterations)
        self._heuristic_cache: Dict[Tuple[int, int, int, int], float] = {}

    def _get_cached_heuristic(self, pos: Tuple[int, int],
                               goal: Tuple[int, int]) -> float:
        """带缓存的启发式计算"""
        key = (pos[0], pos[1], goal[0], goal[1])
        if key not in self._heuristic_cache:
            self._heuristic_cache[key] = self.space.get_heuristic(pos, goal)
        return self._heuristic_cache[key]

    def _jump(self, x: int, y: int, dx: int, dy: int,
              goal: Tuple[int, int],
              obstacles: Set[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        沿方向(dx, dy)跳跃，直到找到跳点或遇到障碍

        Returns:
            跳点坐标，或None（遇到障碍或边界）
        """
        nx, ny = x + dx, y + dy

        # 检查边界和障碍
        while (0 <= nx < self.space.width and
               0 <= ny < self.space.height and
               (nx, ny) not in obstacles):

            # 如果到达目标，返回
            if (nx, ny) == goal:
                return (nx, ny)

            # 检查是否是跳点
            if self._is_jump_point(nx, ny, dx, dy, obstacles):
                return (nx, ny)

            # 对角线移动时需要检查两个方向
            if dx != 0 and dy != 0:
                # 检查水平方向是否有跳点
                if self._jump(nx, ny, dx, 0, goal, obstacles) is not None:
                    return (nx, ny)
                # 检查垂直方向是否有跳点
                if self._jump(nx, ny, 0, dy, goal, obstacles) is not None:
                    return (nx, ny)

            nx += dx
            ny += dy

        return None

    def _is_jump_point(self, x: int, y: int,
                       dx: int, dy: int,
                       obstacles: Set[Tuple[int, int]]) -> bool:
        """
        检查位置(x, y)是否是跳点

        跳点定义：
        1. 有强制邻居（forced neighbor）
        2. 对角线移动的拐角点
        """
        # 检查强制邻居（简化版：只检查基本几何约束）
        if dx != 0 and dy == 0:  # 水平移动
            # 检查上方或下方是否有障碍导致的强制邻居
            if ((x, y - 1) in obstacles and (x - dx, y - 1) not in obstacles) or \
               ((x, y + 1) in obstacles and (x - dx, y + 1) not in obstacles):
                return True

        elif dx == 0 and dy != 0:  # 垂直移动
            if ((x - 1, y) in obstacles and (x - 1, y - dy) not in obstacles) or \
               ((x + 1, y) in obstacles and (x + 1, y - dy) not in obstacles):
                return True

        return False

    def _prune_neighbors(self, x: int, y: int,
                         parent: Optional[JPSNode],
                         obstacles: Set[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
        """
        剪枝邻居：只返回需要考虑的邻居

        Returns:
            [(nx, ny, cost), ...]
        """
        neighbors = []

        if parent is None:
            # 起始点：考虑所有4个方向
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.space.width and
                    0 <= ny < self.space.height and
                    (nx, ny) not in obstacles):
                    cost = self.space.compute_distance((x, y), (nx, ny))
                    neighbors.append((nx, ny, cost))
        else:
            # 根据父节点方向进行剪枝
            dx = self._sign(x - parent.x)
            dy = self._sign(y - parent.y)

            # 自然邻居（直线方向）
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.space.width and
                0 <= ny < self.space.height and
                (nx, ny) not in obstacles):
                cost = self.space.compute_distance((x, y), (nx, ny))
                neighbors.append((nx, ny, cost))

            # 对角线扩展（如果需要）
            if dx != 0 and dy != 0:
                # 对角线时还需要考虑水平和垂直
                for ddx, ddy in [(dx, 0), (0, dy)]:
                    nx, ny = x + ddx, y + ddy
                    if (0 <= nx < self.space.width and
                        0 <= ny < self.space.height and
                        (nx, ny) not in obstacles):
                        cost = self.space.compute_distance((x, y), (nx, ny))
                        neighbors.append((nx, ny, cost))

            # 强制邻居（简化处理）
            # 实际JPS需要更复杂的强制邻居检测

        return neighbors

    def _sign(self, x: int) -> int:
        """符号函数"""
        if x > 0:
            return 1
        elif x < 0:
            return -1
        return 0

    def solve(self, start: Tuple[int, int],
             goal: Tuple[int, int],
             obstacles: Optional[Set[Tuple[int, int]]] = None,
             timeout_ms: float = 1000.0) -> SolverResult:
        """
        JPS求解

        注：这是JPS的简化实现，针对认知空间的连续度量进行了适配。
        完整JPS需要更复杂的剪枝规则。
        """
        start_time = time.time()

        if obstacles is None:
            obstacles = set()

        if start == goal:
            return SolverResult([start], True, 0, 0.0, 0.0)

        if start in obstacles or goal in obstacles:
            return SolverResult(None, False, 0, 0.0)

        # 使用标准A*（因为JPS的剪枝规则需要欧氏距离假设，
        # 而认知空间的度量可能不满足）
        # 但使用启发式缓存优化
        return self._solve_with_cache(start, goal, obstacles, timeout_ms)

    def _solve_with_cache(self, start: Tuple[int, int],
                          goal: Tuple[int, int],
                          obstacles: Set[Tuple[int, int]],
                          timeout_ms: float) -> SolverResult:
        """带启发式缓存的标准A*"""
        start_time = time.time()

        open_set: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, start))

        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        f_score: Dict[Tuple[int, int], float] = {
            start: self._get_cached_heuristic(start, goal)
        }
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed_set: Set[Tuple[int, int]] = set()

        iterations = 0
        nodes_expanded = 0

        while open_set and iterations < self.max_iterations:
            iterations += 1

            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > timeout_ms:
                return SolverResult(None, False, nodes_expanded, elapsed_ms)

            _, current = heapq.heappop(open_set)

            if current == goal:
                path = self._reconstruct_path(came_from, current)
                cost = self.space.compute_path_cost(path)
                return SolverResult(path, True, nodes_expanded, elapsed_ms, cost)

            if current in closed_set:
                continue

            closed_set.add(current)
            nodes_expanded += 1

            # 扩展邻居
            for neighbor in neighbors_4(current, self.space.width, self.space.height):
                if neighbor in closed_set or neighbor in obstacles:
                    continue

                edge_cost = self.space.compute_distance(current, neighbor)
                tentative_g = g_score[current] + edge_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._get_cached_heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        elapsed_ms = (time.time() - start_time) * 1000
        return SolverResult(None, False, nodes_expanded, elapsed_ms)


class BidirectionalGeodesicSolver(GeodesicSolver):
    """
    双向搜索求解器

    同时从起点和终点进行搜索，在中间相遇。
    可以显著减少需要探索的节点数。
    """

    def solve(self, start: Tuple[int, int],
             goal: Tuple[int, int],
             obstacles: Optional[Set[Tuple[int, int]]] = None,
             timeout_ms: float = 1000.0) -> SolverResult:
        """双向A*搜索"""
        start_time = time.time()

        if obstacles is None:
            obstacles = set()

        if start == goal:
            return SolverResult([start], True, 0, 0.0, 0.0)

        if start in obstacles or goal in obstacles:
            return SolverResult(None, False, 0, 0.0)

        # 从起点搜索
        open_set_f: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
        g_score_f: Dict[Tuple[int, int], float] = {start: 0.0}
        came_from_f: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed_set_f: Set[Tuple[int, int]] = set()

        # 从终点搜索
        open_set_b: List[Tuple[float, Tuple[int, int]]] = [(0.0, goal)]
        g_score_b: Dict[Tuple[int, int], float] = {goal: 0.0}
        came_from_b: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed_set_b: Set[Tuple[int, int]] = set()

        meeting_point: Optional[Tuple[int, int]] = None
        best_path_cost = float('inf')
        nodes_expanded = 0

        iterations = 0

        while (open_set_f or open_set_b) and iterations < self.max_iterations:
            iterations += 1

            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > timeout_ms:
                break

            # 选择扩展方向（交替）
            if iterations % 2 == 0 and open_set_f:
                self._expand_node(open_set_f, g_score_f, came_from_f, closed_set_f,
                                 goal, obstacles)
            elif open_set_b:
                self._expand_node(open_set_b, g_score_b, came_from_b, closed_set_b,
                                 start, obstacles)
            elif open_set_f:
                self._expand_node(open_set_f, g_score_f, came_from_f, closed_set_f,
                                 goal, obstacles)

            # 检查相遇
            intersection = closed_set_f & closed_set_b
            if intersection:
                meet = intersection.pop()
                path_cost = g_score_f.get(meet, float('inf')) + g_score_b.get(meet, float('inf'))
                if path_cost < best_path_cost:
                    best_path_cost = path_cost
                    meeting_point = meet
                    # 不立即返回，可能还有更好的路径

            nodes_expanded = len(closed_set_f) + len(closed_set_b)

        elapsed_ms = (time.time() - start_time) * 1000

        if meeting_point is not None:
            # 重建路径
            path_f = self._reconstruct_path_bidirectional(came_from_f, meeting_point, reverse=False)
            path_b = self._reconstruct_path_bidirectional(came_from_b, meeting_point, reverse=True)
            path = path_f + path_b[1:]  # 避免重复中间点

            return SolverResult(path, True, nodes_expanded, elapsed_ms, best_path_cost)

        return SolverResult(None, False, nodes_expanded, elapsed_ms)

    def _expand_node(self, open_set: List, g_score: Dict, came_from: Dict,
                     closed_set: Set, target: Tuple[int, int],
                     obstacles: Set[Tuple[int, int]]):
        """扩展一个节点"""
        if not open_set:
            return

        _, current = heapq.heappop(open_set)

        if current in closed_set:
            return

        closed_set.add(current)

        for neighbor in neighbors_4(current, self.space.width, self.space.height):
            if neighbor in closed_set or neighbor in obstacles:
                continue

            edge_cost = self.space.compute_distance(current, neighbor)
            tentative_g = g_score[current] + edge_cost

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + self.space.get_heuristic(neighbor, target)
                heapq.heappush(open_set, (f_score, neighbor))

    def _reconstruct_path_bidirectional(self, came_from: Dict,
                                        current: Tuple[int, int],
                                        reverse: bool = False) -> List[Tuple[int, int]]:
        """重建路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)

        if reverse:
            path.reverse()
        return path


class IDAStarGeodesicSolver(GeodesicSolver):
    """
    迭代加深A* (IDA*)

    优点：
    - 内存占用极低（只需要当前路径）
    - 适合内存受限场景

    缺点：
    - 可能重复扩展节点
    - 深度限制需要迭代调整
    """

    def solve(self, start: Tuple[int, int],
             goal: Tuple[int, int],
             obstacles: Optional[Set[Tuple[int, int]]] = None,
             timeout_ms: float = 1000.0) -> SolverResult:
        """IDA*搜索"""
        start_time = time.time()

        if obstacles is None:
            obstacles = set()

        if start == goal:
            return SolverResult([start], True, 0, 0.0, 0.0)

        if start in obstacles or goal in obstacles:
            return SolverResult(None, False, 0, 0.0)

        threshold = self.space.get_heuristic(start, goal)
        path = [start]
        nodes_expanded = 0

        while True:
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > timeout_ms:
                return SolverResult(None, False, nodes_expanded, elapsed_ms)

            result = self._search(path, 0, threshold, goal, obstacles)
            nodes_expanded += result[1]

            if result[0] is True:  # 找到路径
                elapsed_ms = (time.time() - start_time) * 1000
                cost = self.space.compute_path_cost(path)
                return SolverResult(path.copy(), True, nodes_expanded, elapsed_ms, cost)

            if result[2] == float('inf'):  # 无解
                elapsed_ms = (time.time() - start_time) * 1000
                return SolverResult(None, False, nodes_expanded, elapsed_ms)

            threshold = result[2]  # 新的阈值

    def _search(self, path: List, g: float, threshold: float,
                goal: Tuple[int, int],
                obstacles: Set[Tuple[int, int]]) -> Tuple:
        """
        递归搜索

        Returns:
            (found, nodes_expanded, new_threshold)
        """
        current = path[-1]
        f = g + self.space.get_heuristic(current, goal)

        if f > threshold:
            return (False, 0, f)

        if current == goal:
            return (True, 0, threshold)

        nodes_expanded = 1
        min_threshold = float('inf')

        for neighbor in neighbors_4(current, self.space.width, self.space.height):
            if neighbor in path or neighbor in obstacles:
                continue

            edge_cost = self.space.compute_distance(current, neighbor)
            path.append(neighbor)

            result = self._search(path, g + edge_cost, threshold, goal, obstacles)
            nodes_expanded += result[1]

            if result[0] is True:
                return (True, nodes_expanded, threshold)

            if result[2] < min_threshold:
                min_threshold = result[2]

            path.pop()

        return (False, nodes_expanded, min_threshold)
