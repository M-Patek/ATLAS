"""
3D Geodesic Solver
3D测地线求解器

扩展A*算法到3D网格空间。
"""

import heapq
import time
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass
from ..core.solver import SolverResult


@dataclass
class SolverResult3D:
    """3D求解器结果"""
    success: bool
    path: List[Tuple[int, int, int]]
    cost: float
    nodes_expanded: int
    search_tree: Dict[Tuple[int, int, int], Tuple[int, int, int]]


class GeodesicSolver3D:
    """
    3D测地线求解器

    在3D认知空间中寻找最短路径。
    """

    def __init__(self, space):
        """
        Args:
            space: CognitiveSpace3D实例
        """
        self.space = space

    def solve(self,
              start: Tuple[int, int, int],
              goal: Tuple[int, int, int],
              obstacles: Optional[Set[Tuple[int, int, int]]] = None,
              max_iterations: int = 10000) -> SolverResult3D:
        """
        使用A*算法在3D空间中寻找路径

        Args:
            start: 起点 (x, y, z)
            goal: 终点 (x, y, z)
            obstacles: 障碍物集合
            max_iterations: 最大迭代次数

        Returns:
            SolverResult3D
        """
        if obstacles is None:
            obstacles = set()

        # 检查起点终点
        if start in obstacles:
            return SolverResult3D(False, [], float('inf'), 0, {})
        if goal in obstacles:
            return SolverResult3D(False, [], float('inf'), 0, {})

        # A*算法
        open_set = [(self.space.get_heuristic(start, goal), 0, start)]
        heapq.heapify(open_set)

        g_score = {start: 0.0}
        came_from = {}
        closed_set = set()
        nodes_expanded = 0

        while open_set and nodes_expanded < max_iterations:
            _, current_g, current = heapq.heappop(open_set)

            if current in closed_set:
                continue

            closed_set.add(current)
            nodes_expanded += 1

            if current == goal:
                path = self._reconstruct_path(came_from, current)
                return SolverResult3D(
                    success=True,
                    path=path,
                    cost=g_score[current],
                    nodes_expanded=nodes_expanded,
                    search_tree=came_from
                )

            for neighbor in self.space.get_neighbors(current):
                if neighbor in obstacles or neighbor in closed_set:
                    continue

                tentative_g = g_score[current] + self.space.compute_distance(current, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.space.get_heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        # 未找到路径
        return SolverResult3D(False, [], float('inf'), nodes_expanded, came_from)

    def _reconstruct_path(self,
                         came_from: Dict,
                         current: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """重建路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def solve_multi_goal(self,
                        start: Tuple[int, int, int],
                        goals: List[Tuple[int, int, int]],
                        obstacles: Optional[Set[Tuple[int, int, int]]] = None) -> Dict[str, Any]:
        """
        找到到多个目标中最近的一个

        Returns:
            {'goal': (x,y,z), 'path': [...], 'cost': float}
        """
        best_result = None
        best_cost = float('inf')

        for goal in goals:
            result = self.solve(start, goal, obstacles)
            if result.success and result.cost < best_cost:
                best_cost = result.cost
                best_result = {
                    'goal': goal,
                    'path': result.path,
                    'cost': result.cost,
                    'nodes_expanded': result.nodes_expanded
                }

        return best_result or {'goal': None, 'path': [], 'cost': float('inf')}

