"""
ATLAS Core: Geodesic Solver
测地线求解器

在认知空间中求解最短路径（测地线）
设计原则:
- 求解器与具体空间解耦
- 通过 CognitiveSpace 接口操作
- 支持不同的搜索算法
"""

import heapq
import time
from typing import List, Tuple, Optional, Set, Dict, Callable
from abc import ABC, abstractmethod

from .space import CognitiveSpace, neighbors_4


class SolverResult:
    """求解器结果"""

    def __init__(self, path: Optional[List[Tuple[int, int]]],
                 success: bool,
                 nodes_expanded: int,
                 time_ms: float,
                 cost: float = 0.0):
        self.path = path
        self.success = success
        self.nodes_expanded = nodes_expanded
        self.time_ms = time_ms
        self.cost = cost

    def __repr__(self) -> str:
        if self.success:
            return f"SolverResult(success=True, steps={len(self.path)}, expanded={self.nodes_expanded}, time={self.time_ms:.2f}ms)"
        return f"SolverResult(success=False, expanded={self.nodes_expanded}, time={self.time_ms:.2f}ms)"


class GeodesicSolver:
    """
    通用测地线求解器

    使用 A* 算法在任意 CognitiveSpace 中求解最短路径
    """

    def __init__(self, space: CognitiveSpace,
                 use_diagonal: bool = False,
                 max_iterations: Optional[int] = None):
        self.space = space
        self.use_diagonal = use_diagonal
        self.max_iterations = max_iterations or (space.width * space.height * 2)

    def solve(self, start: Tuple[int, int],
             goal: Tuple[int, int],
             obstacles: Optional[Set[Tuple[int, int]]] = None,
             timeout_ms: float = 1000.0) -> SolverResult:
        """
        求解从 start 到 goal 的测地线

        Args:
            start: 起点
            goal: 终点
            obstacles: 障碍物集合
            timeout_ms: 超时时间（毫秒）

        Returns:
            SolverResult 包含路径和统计信息
        """
        start_time = time.time()

        if obstacles is None:
            obstacles = set()

        # 边界检查
        if start == goal:
            return SolverResult([start], True, 0, 0.0, 0.0)

        if start in obstacles or goal in obstacles:
            return SolverResult(None, False, 0, 0.0)

        # A* 搜索
        open_set: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, start))

        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        f_score: Dict[Tuple[int, int], float] = {
            start: self.space.get_heuristic(start, goal)
        }
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed_set: Set[Tuple[int, int]] = set()

        iterations = 0
        nodes_expanded = 0

        while open_set and iterations < self.max_iterations:
            iterations += 1

            # 检查超时
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
            if self.use_diagonal:
                neighbors = self._neighbors_8(current)
            else:
                neighbors = neighbors_4(current, self.space.width, self.space.height)

            for neighbor in neighbors:
                if neighbor in closed_set or neighbor in obstacles:
                    continue

                # 计算边成本
                edge_cost = self.space.compute_distance(current, neighbor)
                tentative_g = g_score[current] + edge_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.space.get_heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # 失败
        elapsed_ms = (time.time() - start_time) * 1000
        return SolverResult(None, False, nodes_expanded, elapsed_ms)

    def _neighbors_8(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """8连通邻居"""
        x, y = pos
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.space.width and 0 <= ny < self.space.height:
                    neighbors.append((nx, ny))
        return neighbors

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]],
                         current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """重建路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


class DijkstraSolver(GeodesicSolver):
    """
    Dijkstra 求解器

    特殊化的求解器，启发式为0，保证找到全局最优
    """

    def solve(self, start: Tuple[int, int],
             goal: Tuple[int, int],
             obstacles: Optional[Set[Tuple[int, int]]] = None,
             timeout_ms: float = 1000.0) -> SolverResult:
        """Dijkstra 搜索（启发式为0）"""
        start_time = time.time()

        if obstacles is None:
            obstacles = set()

        if start == goal:
            return SolverResult([start], True, 0, 0.0, 0.0)

        # Dijkstra: 启发式始终为0
        open_set: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed_set: Set[Tuple[int, int]] = set()

        nodes_expanded = 0

        while open_set:
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

            for neighbor in neighbors_4(current, self.space.width, self.space.height):
                if neighbor in closed_set or neighbor in obstacles:
                    continue

                edge_cost = self.space.compute_distance(current, neighbor)
                tentative_g = g_score[current] + edge_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    heapq.heappush(open_set, (tentative_g, neighbor))

        elapsed_ms = (time.time() - start_time) * 1000
        return SolverResult(None, False, nodes_expanded, elapsed_ms)


class GreedySolver(GeodesicSolver):
    """
    贪婪求解器

    只使用启发式，不保证最优，但速度快
    """

    def solve(self, start: Tuple[int, int],
             goal: Tuple[int, int],
             obstacles: Optional[Set[Tuple[int, int]]] = None,
             timeout_ms: float = 1000.0) -> SolverResult:
        """贪婪最佳优先搜索"""
        start_time = time.time()

        if obstacles is None:
            obstacles = set()

        if start == goal:
            return SolverResult([start], True, 0, 0.0, 0.0)

        open_set: List[Tuple[float, Tuple[int, int]]] = [
            (self.space.get_heuristic(start, goal), start)
        ]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        visited: Set[Tuple[int, int]] = {start}

        nodes_expanded = 0

        while open_set:
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > timeout_ms:
                return SolverResult(None, False, nodes_expanded, elapsed_ms)

            _, current = heapq.heappop(open_set)

            if current == goal:
                path = self._reconstruct_path(came_from, current)
                cost = self.space.compute_path_cost(path)
                return SolverResult(path, True, nodes_expanded, elapsed_ms, cost)

            nodes_expanded += 1

            for neighbor in neighbors_4(current, self.space.width, self.space.height):
                if neighbor in visited or neighbor in obstacles:
                    continue

                visited.add(neighbor)
                came_from[neighbor] = current
                h = self.space.get_heuristic(neighbor, goal)
                heapq.heappush(open_set, (h, neighbor))

        elapsed_ms = (time.time() - start_time) * 1000
        return SolverResult(None, False, nodes_expanded, elapsed_ms)
