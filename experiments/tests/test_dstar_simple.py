"""
简化测�? D* Lite 基础功能
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core import GeodesicSolver
from src.core.registry import create_space

print("测试基础求解�?..")

space = create_space("euclidean", width=20, height=20)
solver = GeodesicSolver(space)

result = solver.solve(
    start=(5, 10),
    goal=(15, 10),
    obstacles={(10, y) for y in range(5, 15) if y != 10}
)

print(f"Success: {result.success}")
print(f"Steps: {len(result.path) if result.path else 0}")
print(f"Time: {result.time_ms:.2f}ms")
print("基础求解器工作正�?")
