"""
ATLAS Core Tests

使用 pytest 运行: pytest tests/ -v
"""

import sys
import os

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np

from src.core.space import CognitiveSpace
from src.core.registry import registry, create_space, list_available_spaces
from src.core.solver import GeodesicSolver


class TestRegistry:
    """测试注册表功能"""

    def test_list_spaces(self):
        """测试列出可用空间"""
        spaces = list_available_spaces()
        assert len(spaces) > 0
        assert 'euclidean' in spaces

    def test_create_euclidean(self):
        """测试创建欧氏空间"""
        space = create_space('euclidean', width=10, height=10)
        assert space is not None
        assert hasattr(space, 'compute_distance')

    def test_create_ricci(self):
        """测试创建Ricci空间"""
        space = create_space('ricci', width=10, height=10)
        assert space is not None


class TestSolver:
    """测试求解器"""

    def test_geodesic_solver(self):
        """测试测地线求解器"""
        solver = GeodesicSolver()
        assert solver is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
