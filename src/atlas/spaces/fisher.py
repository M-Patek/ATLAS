"""
Fisher Information Space
Fisher 信息度量空间

基于统计流形的信息几何
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional

from ..core.space import CognitiveSpace, register_space


@register_space("fisher")
class FisherSpace(CognitiveSpace):
    """
    Fisher 信息空间

    基于信念置信度的信息度量:
    g_ij ≈ 1 / confidence

    低置信度 = 高 Fisher 距离 = 需要更多探索
    """

    def __init__(self, width: int, height: int,
                 temperature: float = 1.0,
                 **kwargs):
        super().__init__(width, height, name="fisher")

        self.temperature = temperature

        # 信念场 [0, 1]
        self.belief = np.ones((width, height)) * 0.5

        # 置信度场
        self.confidence = np.zeros((width, height))

        # 观测计数
        self.observation_count = np.zeros((width, height), dtype=np.int32)

    def _fisher_metric(self, x: int, y: int) -> float:
        """计算 Fisher 度量"""
        conf = max(0.01, self.confidence[x, y])
        return 1.0 / conf

    def compute_distance(self, pos1: Tuple[int, int],
                        pos2: Tuple[int, int]) -> float:
        """Fisher 距离"""
        x1, y1 = pos1
        x2, y2 = pos2

        euclidean_dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        n_samples = max(int(euclidean_dist), 1)

        total = 0.0
        for i in range(n_samples):
            t = i / max(1, n_samples - 1)
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))

            fisher_factor = self._fisher_metric(x, y)
            total += fisher_factor * (euclidean_dist / n_samples)

        return total

    def get_heuristic(self, pos: Tuple[int, int],
                     goal: Tuple[int, int]) -> float:
        """启发式"""
        return self.compute_distance(pos, goal)

    def update_from_observation(self, position: Tuple[int, int],
                                observation: Dict[str, Any]) -> None:
        """贝叶斯更新信念"""
        x, y = position

        if 0 <= x < self.width and 0 <= y < self.height:
            self.observation_count[x, y] += 1

            # 更新置信度（观测越多越自信）
            self.confidence[x, y] = min(1.0,
                1.0 - np.exp(-0.2 * self.observation_count[x, y]))

        # 障碍物降低信念
        if 'obstacles' in observation:
            for ox, oy in observation['obstacles']:
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        nx, ny = ox + dx, oy + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            self.belief[nx, ny] *= 0.9
                            self.confidence[nx, ny] = max(0.0,
                                self.confidence[nx, ny] - 0.05)

        # 目标增加信念
        if 'goal_position' in observation:
            gx, gy = observation['goal_position']
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < 5:
                            self.belief[nx, ny] = min(1.0,
                                self.belief[nx, ny] + 0.1 * (1 - dist/5))

    def get_visualization_fields(self) -> Dict[str, np.ndarray]:
        """可视化数据"""
        return {
            'belief': self.belief.copy(),
            'confidence': self.confidence.copy(),
            'fisher_metric': 1.0 / np.maximum(0.01, self.confidence),
        }

    def predict_next_state(self, position: Tuple[int, int],
                           observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于 Fisher 场数据预测下一步

        预测逻辑：
        - 高置信度 = 高 Fisher 信息 = 更"便宜"的移动
        - 高信念 = 更可信的区域
        """
        from ..core.space import neighbors_4

        goal = observation.get('goal_position')
        obstacles = set(observation.get('obstacles', []))

        best_pos = position
        best_score = float('-inf')

        for nx, ny in neighbors_4(position, self.width, self.height):
            if (nx, ny) in obstacles:
                continue

            # Fisher 特有的评分：高置信度 + 高信念 = 好
            confidence_bonus = self.confidence[nx, ny]
            belief_bonus = self.belief[nx, ny]

            # 低 Fisher metric = 便宜
            fisher_metric = 1.0 / max(0.01, self.confidence[nx, ny])
            cost_penalty = -fisher_metric * 0.1

            # 目标导向
            goal_bonus = 0.0
            if goal:
                dist_to_goal = np.sqrt((nx - goal[0])**2 + (ny - goal[1])**2)
                goal_bonus = -dist_to_goal * 0.1

            score = confidence_bonus + belief_bonus + cost_penalty + goal_bonus

            if score > best_score:
                best_score = score
                best_pos = (nx, ny)

        x, y = best_pos
        return {
            'predicted_position': best_pos,
            'predicted_cost': self.compute_distance(position, best_pos),
            'predicted_uncertainty': 1.0 - self.confidence[x, y],
            'passable': best_pos != position,
            'fisher_score': float(best_score),
            'predicted_confidence': float(self.confidence[x, y]),
            'predicted_belief': float(self.belief[x, y]),
        }

    def compute_validity(self, position: Tuple[int, int],
                        observation: Dict[str, Any],
                        actual: Optional[Dict[str, Any]] = None) -> float:
        """
        Fisher空间的适用性判定

        Fisher空间基于置信度，适用条件：
        - 高置信度区域：统计学习有效
        - 低置信度区域：需要更多探索
        - 观测不足时：置信度低，Fisher优势不明显
        """
        x, y = position

        if not (0 <= x < self.width and 0 <= y < self.height):
            return 0.0

        current_confidence = self.confidence[x, y]
        current_belief = self.belief[x, y]

        # 高置信度 = Fisher有效
        # 低置信度 = 需要更多数据
        if current_confidence > 0.7:
            # 高置信度：Fisher非常有效
            validity = 0.7 + 0.3 * min((current_confidence - 0.7) / 0.3, 1.0)
        elif current_confidence > 0.3:
            # 中等置信度：Fisher有效但有限
            validity = 0.4 + 0.3 * ((current_confidence - 0.3) / 0.4)
        else:
            # 低置信度：Fisher优势不明显
            validity = 0.2 + 0.2 * (current_confidence / 0.3)

        # 信念也影响 validity
        belief_boost = 0.1 * current_belief

        return min(1.0, validity + belief_boost)

    def get_validity_fields(self) -> Dict[str, np.ndarray]:
        """Fisher空间的有效性场"""
        validity = np.zeros((self.width, self.height))

        for x in range(self.width):
            for y in range(self.height):
                conf = self.confidence[x, y]
                if conf > 0.7:
                    v = 0.7 + 0.3 * min((conf - 0.7) / 0.3, 1.0)
                elif conf > 0.3:
                    v = 0.4 + 0.3 * ((conf - 0.3) / 0.4)
                else:
                    v = 0.2 + 0.2 * (conf / 0.3)
                validity[x, y] = v

        return {
            'validity': validity,
            'confidence': self.confidence.copy(),
            'belief': self.belief.copy(),
        }
