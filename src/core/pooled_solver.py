"""
示例：使用内存池优化的求解器

展示如何集成内存池到求解器中。
"""
import heapq
import time
from typing import List, Tuple, Optional, Set, Dict

from .solver import GeodesicSolver, SolverResult
from .space import CognitiveSpace, neighbors_4
from .memory_pool import get_array_pool, get_workspace_pool


class PooledGeodesicSolver(GeodesicSolver):
    """
    使用内存池的求解器

    减少求解过程中的内存分配。
    """

    def solve(self, start: Tuple[int, int],
             goal: Tuple[int, int],
             obstacles: Optional[Set[Tuple[int, int]]] = None,
             timeout_ms: float = 1000.0) -> SolverResult:
        """使用内存池的求解"""
        start_time = time.time()

        if obstacles is None:
            obstacles = set()

        if start == goal:
            return SolverResult([start], True, 0, 0.0, 0.0)

        if start in obstacles or goal in obstacles:
            return SolverResult(None, False, 0, 0.0)

        pool = get_workspace_pool()

        # 从池中获取容器（避免重复分配）
        open_set: List[Tuple[float, Tuple[int, int]]] = pool.acquire_list()
        heapq.heappush(open_set, (0.0, start))

        g_score: Dict[Tuple[int, int], float] = pool.acquire_dict()
        g_score[start] = 0.0

        f_score: Dict[Tuple[int, int], float] = pool.acquire_dict()
        f_score[start] = self.space.get_heuristic(start, goal)

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = pool.acquire_dict()
        closed_set: Set[Tuple[int, int]] = set()

        iterations = 0
        nodes_expanded = 0

        try:
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
                        f_score[neighbor] = tentative_g + self.space.get_heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

            elapsed_ms = (time.time() - start_time) * 1000
            return SolverResult(None, False, nodes_expanded, elapsed_ms)

        finally:
            # 释放回池
            pool.release_list(open_set)
            pool.release_dict(g_score)
            pool.release_dict(f_score)
            pool.release_dict(came_from)
