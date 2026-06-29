"""
ATLAS Core: Incremental Path Planning
增量路径规划

D* Lite 算法实现 - 支持动态环境下的高效重新规划
"""

import heapq
import time
from typing import Dict, List, Tuple, Optional, Set, Callable
from dataclasses import dataclass

from .space import CognitiveSpace, neighbors_4
from .solver import SolverResult


class DStarLiteSolver:
    """
    D* Lite 增量求解器 (简化实现)

    特点:
    - 反向搜索 (从目标到起点)
    - 增量更新 (只更新受环境变化影响的节点)
    - 适合动态环境 (障碍物移动时高效重新规划)

    参考文献:
    Koenig, S., & Likhachev, M. (2002). D* Lite.
    """

    def __init__(self, space: CognitiveSpace,
                 use_diagonal: bool = False,
                 max_iterations: int = 100000):
        self.space = space
        self.use_diagonal = use_diagonal
        self.max_iterations = max_iterations

        # D* Lite 状态
        self.start: Optional[Tuple[int, int]] = None
        self.goal: Optional[Tuple[int, int]] = None
        self.obstacles: Set[Tuple[int, int]] = set()

        # 核心场
        self.g: Dict[Tuple[int, int], float] = {}  # 实际代价估计
        self.rhs: Dict[Tuple[int, int], float] = {}  # 前瞻值

    def _heuristic(self, s: Tuple[int, int], target: Tuple[int, int]) -> float:
        """启发式：估计距离"""
        return abs(s[0] - target[0]) + abs(s[1] - target[1])

    def _get_neighbors(self, s: Tuple[int, int]) -> List[Tuple[int, int]]:
        """获取邻居"""
        return neighbors_4(s, self.space.width, self.space.height)

    def _cost(self, s: Tuple[int, int], s_next: Tuple[int, int]) -> float:
        """计算边代价（考虑障碍物）"""
        if s_next in self.obstacles or s in self.obstacles:
            return float('inf')
        return 1.0  # 简化：使用单位代价

    def _update_rhs(self, s: Tuple[int, int]):
        """重新计算rhs(s) = min_{s'}(c(s,s') + g(s'))"""
        if s == self.goal:
            self.rhs[s] = 0.0
            return

        min_val = float('inf')
        for s_next in self._get_neighbors(s):
            cost = self._cost(s, s_next)
            g_next = self.g.get(s_next, float('inf'))
            if cost + g_next < min_val:
                min_val = cost + g_next

        self.rhs[s] = min_val

    def compute_shortest_path(self) -> bool:
        """
        计算最短路径 (简化版Dijkstra)

        返回是否成功找到路径
        """
        if not self.start or not self.goal:
            return False

        # 使用Dijkstra从goal反向计算
        self.g.clear()
        self.rhs.clear()

        open_set: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, self.goal))

        self.g[self.goal] = 0.0

        visited = set()
        iterations = 0

        while open_set and iterations < self.max_iterations:
            iterations += 1

            current_cost, current = heapq.heappop(open_set)

            if current in visited:
                continue
            visited.add(current)

            # 到达起点，完成
            if current == self.start:
                break

            for neighbor in self._get_neighbors(current):
                if neighbor in visited:
                    continue

                edge_cost = self._cost(neighbor, current)
                if edge_cost == float('inf'):
                    continue

                new_cost = current_cost + edge_cost
                if new_cost < self.g.get(neighbor, float('inf')):
                    self.g[neighbor] = new_cost
                    heapq.heappush(open_set, (new_cost, neighbor))

        return self.start in self.g and self.g[self.start] < float('inf')

    def solve(self, start: Tuple[int, int],
             goal: Tuple[int, int],
             obstacles: Optional[Set[Tuple[int, int]]] = None,
             timeout_ms: float = 1000.0) -> SolverResult:
        """
        求解路径（与GeodesicSolver兼容的接口）
        """
        start_time = time.time()

        self.start = start
        self.goal = goal
        self.obstacles = obstacles or set()

        # 计算最短路径
        success = self.compute_shortest_path()

        elapsed = (time.time() - start_time) * 1000

        if success:
            path = self._extract_path(start, goal)
            if path:
                cost = self.space.compute_path_cost(path)
                return SolverResult(path, True, len(self.g), elapsed, cost)

        return SolverResult(None, False, len(self.g), elapsed, 0.0)

    def _extract_path(self, start: Tuple[int, int],
                     goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """从g值中提取路径（贪心）"""
        path = [start]
        current = start
        visited = {start}

        max_steps = self.space.width * self.space.height

        for _ in range(max_steps):
            if current == goal:
                return path

            # 找最小g值的邻居
            best_next = None
            best_g = float('inf')

            for neighbor in self._get_neighbors(current):
                if neighbor in visited:
                    continue
                g = self.g.get(neighbor, float('inf'))
                cost = self._cost(current, neighbor)
                total = cost + g
                if total < best_g:
                    best_g = total
                    best_next = neighbor

            if best_next is None or best_g == float('inf'):
                return []  # 无路径

            current = best_next
            path.append(current)
            visited.add(current)

        return []

    def update_obstacles(self, new_obstacles: Set[Tuple[int, int]],
                        removed_obstacles: Set[Tuple[int, int]] = None):
        """
        增量更新障碍物

        参数:
            new_obstacles: 新添加的障碍物
            removed_obstacles: 移除的障碍物
        """
        changed = False

        if new_obstacles:
            for obs in new_obstacles:
                if obs not in self.obstacles:
                    self.obstacles.add(obs)
                    changed = True

        if removed_obstacles:
            for obs in removed_obstacles:
                if obs in self.obstacles:
                    self.obstacles.discard(obs)
                    changed = True

        # 如果有变化，重新计算
        if changed:
            self.compute_shortest_path()


class AdaptiveNavigator:
    """
    自适应导航器

    整合 D* Lite + CognitiveSpace + 闭环控制
    支持实时重新规划和环境适应
    """

    def __init__(self, space: CognitiveSpace,
                 replan_threshold: float = 1.0,
                 lookahead_steps: int = 3):
        self.space = space
        self.solver = DStarLiteSolver(space)
        self.replan_threshold = replan_threshold
        self.lookahead_steps = lookahead_steps

        # 状态
        self.current_path: List[Tuple[int, int]] = []
        self.path_index = 0
        self.last_update_position: Optional[Tuple[int, int]] = None

        # 统计
        self.stats = {
            'replan_count': 0,
            'total_steps': 0,
            'path_deviations': 0,
        }

    def initialize(self, start: Tuple[int, int],
                   goal: Tuple[int, int],
                   obstacles: Optional[Set[Tuple[int, int]]] = None) -> bool:
        """初始化导航任务"""
        result = self.solver.solve(start, goal, obstacles)

        if result.success:
            self.current_path = result.path
            self.path_index = 0
            self.last_update_position = start
            return True
        else:
            self.current_path = []
            return False

    def step(self, current_position: Tuple[int, int],
             observation: Optional[Dict] = None) -> Optional[Tuple[int, int]]:
        """
        导航步进

        Args:
            current_position: 当前位置
            observation: 观测数据，可能包含新障碍物

        Returns:
            推荐的下一步位置，或 None 如果已到达目标
        """
        self.stats['total_steps'] += 1

        # 1. 更新空间
        if observation:
            self.space.update_from_observation(current_position, observation)

            # 检查是否有新障碍物
            new_obstacles = observation.get('obstacles', set())
            if new_obstacles:
                obs_set = set(new_obstacles) if not isinstance(new_obstacles, set) else new_obstacles
                existing = self.solver.obstacles
                truly_new = obs_set - existing

                if truly_new:
                    self.stats['replan_count'] += 1
                    self.solver.update_obstacles(truly_new)
                    # 重新提取路径
                    if self.solver.start and self.solver.goal:
                        new_path = self.solver._extract_path(
                            current_position, self.solver.goal
                        )
                        if new_path:
                            self.current_path = new_path
                            self.path_index = 0

        # 2. 定期重新规划（基于距离阈值）
        if self.last_update_position:
            dist_since_update = self.space.compute_distance(
                self.last_update_position, current_position
            )
            if dist_since_update > self.replan_threshold:
                self.stats['replan_count'] += 1
                result = self.solver.solve(
                    current_position, self.solver.goal, self.solver.obstacles
                )
                if result.success:
                    self.current_path = result.path
                    self.path_index = 0
                    self.last_update_position = current_position

        # 3. 获取下一步
        if not self.current_path:
            return None

        # 找到当前位置在路径中的索引
        try:
            idx = self.current_path.index(current_position)
        except ValueError:
            idx = self._find_nearest_path_index(current_position)
            self.stats['path_deviations'] += 1

        if idx + 1 < len(self.current_path):
            return self.current_path[idx + 1]
        else:
            return None

    def _find_nearest_path_index(self, position: Tuple[int, int]) -> int:
        """找到路径上最近的点"""
        min_dist = float('inf')
        min_idx = 0

        for i, path_point in enumerate(self.current_path):
            dist = self.space.compute_distance(position, path_point)
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        return min_idx

    def get_path(self) -> List[Tuple[int, int]]:
        """获取当前规划路径"""
        return self.current_path.copy()

    def get_statistics(self) -> Dict:
        """获取导航统计"""
        return self.stats.copy()
