"""
理论边界切换验证测试

验证核心机制：
1. 空间自我判定 validity
2. 理论边界空间自动切换
3. 不同场景下选择不同空间

场景设计（类比物理）：
- 低速场景（平坦地形）：欧氏空间 validity=0.9 → 选择欧氏
- 中速场景（有目标）：Conformal validity=0.78 → 选择 Conformal
- 高速场景（高曲率）：Ricci validity=0.5 → 选择 Ricci
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from atlas.core.registry import create_space
from atlas.spaces.theory_boundary import TheoryBoundarySpace


def test_theory_boundary():
    """测试理论边界切换"""
    print("=" * 70)
    print("Theory Boundary Switching Test")
    print("=" * 70)

    # 创建候选空间
    spaces = {
        'euclidean': create_space('euclidean', 20, 20),
        'ricci': create_space('ricci', 20, 20, curvature_scale=2.0),
        'conformal': create_space('conformal', 20, 20),
        'fisher': create_space('fisher', 20, 20),
    }

    # 创建理论边界空间
    boundary = TheoryBoundarySpace(20, 20, spaces=spaces)

    # 场景1：平坦地形（低速 = 欧氏足够）
    print("\n--- Scenario 1: Flat Terrain (Low Speed) ---")
    for step in range(5):
        pos = (5 + step, 10)
        obs = {
            'position': pos,
            'obstacles': [],
        }
        boundary.update_from_observation(pos, obs)

    stats = boundary.get_statistics()
    print(f"  Current space: {stats['current_space']}")
    print(f"  Validities: {stats.get('latest_validities', {})}")
    print(f"  Expected: euclidean (flat terrain)")

    # 场景2：有明确目标（中速 = Conformal 有效）
    print("\n--- Scenario 2: With Goal (Medium Speed) ---")
    for step in range(5):
        pos = (10 + step, 10)
        obs = {
            'position': pos,
            'goal_position': (18, 18),
            'obstacles': [],
        }
        boundary.update_from_observation(pos, obs)

    stats = boundary.get_statistics()
    print(f"  Current space: {stats['current_space']}")
    print(f"  Validities: {stats.get('latest_validities', {})}")
    print(f"  Expected: conformal (has goal)")

    # 场景3：高曲率地形（高速 = Ricci 有效）
    print("\n--- Scenario 3: High Curvature (High Speed) ---")
    # 制造高曲率
    for step in range(5):
        pos = (15, 5 + step)
        obs = {
            'position': pos,
            'obstacles': [(i, j) for i in range(12, 18) for j in range(8, 12)],
        }
        boundary.update_from_observation(pos, obs)

    stats = boundary.get_statistics()
    print(f"  Current space: {stats['current_space']}")
    print(f"  Validities: {stats.get('latest_validities', {})}")
    print(f"  Expected: ricci (high curvature)")

    # 场景4：回到平坦地形（应该切回欧氏）
    print("\n--- Scenario 4: Back to Flat (Low Speed) ---")
    for step in range(5):
        pos = (5, 5 + step)
        obs = {
            'position': pos,
            'obstacles': [],
        }
        boundary.update_from_observation(pos, obs)

    stats = boundary.get_statistics()
    print(f"  Current space: {stats['current_space']}")
    print(f"  Validities: {stats.get('latest_validities', {})}")
    print(f"  Expected: euclidean (back to flat)")

    # 最终统计
    print("\n" + "=" * 70)
    print("Final Statistics")
    print("=" * 70)
    stats = boundary.get_statistics()
    print(f"Total switches: {stats['switch_count']}")
    print(f"Space usage: {stats['space_usage']}")
    print(f"Switch history:")
    for switch in boundary.switch_history:
        print(f"  {switch['from']} -> {switch['to']} "
              f"(from_v={switch['from_validity']:.2f}, "
              f"to_v={switch['to_validity']:.2f})")

    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    test_theory_boundary()
