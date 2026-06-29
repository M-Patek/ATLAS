"""
ATLAS Spaces Tests
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np

from src.core.registry import create_space


class TestEuclideanSpace:
    """测试欧氏空间"""

    def test_distance(self):
        """测试距离计算"""
        space = create_space('euclidean', width=10, height=10)
        dist = space.compute_distance((0, 0), (3, 4))
        assert dist == 5.0

    def test_heuristic(self):
        """测试启发式函数"""
        space = create_space('euclidean', width=10, height=10)
        h = space.get_heuristic((0, 0), (3, 4))
        assert h == 5.0


class TestRicciSpace:
    """测试Ricci空间"""

    def test_update_and_distance(self):
        """测试更新后距离计算"""
        space = create_space('ricci', width=10, height=10)
        space.update_from_observation((5, 5), {
            'obstacles': [(5, 6)],
            'goal_position': (9, 9),
        })
        dist = space.compute_distance((0, 0), (9, 9))
        assert dist > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
