"""
ATLAS 空间更新验证测试
验证各个空间的 update_from_observation 是否正确工作
"""

import numpy as np
from typing import Dict, Any
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from atlas.core.registry import create_space, list_available_spaces


def test_space_updates():
    """测试空间更新"""
    print("=" * 70)
    print("ATLAS Space Update Verification")
    print("=" * 70)

    # 获取可用空间
    spaces = list_available_spaces()
    print(f"\nAvailable spaces: {list(spaces.keys())}")

    # 测试每个空间
    for space_name in ['ricci', 'euclidean', 'fisher', 'wasserstein', 'finsler']:
        if space_name not in spaces:
            continue

        print(f"\n--- Testing: {space_name} ---")

        try:
            # 创建空间
            space = create_space(space_name, 10, 10)

            # 获取初始状态
            initial_stats = space.get_statistics()
            print(f"  Initial: {initial_stats}")

            # 更新：遇到障碍
            space.update_from_observation(
                (5, 5),
                {
                    'obstacles': [(3, 3), (5, 5)],
                    'goal_position': (9, 9)
                }
            )

            # 获取更新后状态
            after_obstacle_stats = space.get_statistics()
            print(f"  After obstacle: {after_obstacle_stats}")

            # 计算距离
            try:
                dist = space.compute_distance((0, 0), (9, 9))
                print(f"  Distance (0,0)->(9,9): {dist:.2f}")
            except Exception as e:
                print(f"  Distance error: {e}")

            # 计算启发式
            try:
                h = space.get_heuristic((0, 0), (9, 9))
                print(f"  Heuristic (0,0)->(9,9): {h:.2f}")
            except Exception as e:
                print(f"  Heuristic error: {e}")

            # 可视化字段
            try:
                fields = space.get_visualization_fields()
                print(f"  Fields: {list(fields.keys())}")
            except Exception as e:
                print(f"  Fields error: {e}")

        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)


def test_ricci_curvature():
    """专门测试 Ricci 曲率更新"""
    print("\n" + "=" * 70)
    print("Ricci Curvature Update Test")
    print("=" * 70)

    # 创建 Ricci 空间
    space = create_space('ricci', 10, 10)

    # 初始状态
    print("\nInitial state:")
    fields = space.get_visualization_fields()
    print(f"  Uncertainty at (5,5): {fields['uncertainty'][5, 5]:.2f}")
    print(f"  Curvature at (5,5): {fields['curvature'][5, 5]:.2f}")
    print(f"  Familiarity at (5,5): {fields['familiarity'][5, 5]:.2f}")

    # 更新：在 (5,5) 遇到障碍
    print("\nAfter obstacle at (5,5):")
    space.update_from_observation(
        (5, 5),
        {'obstacles': [(5, 5)]}
    )
    fields = space.get_visualization_fields()
    print(f"  Uncertainty at (5,5): {fields['uncertainty'][5, 5]:.2f}")
    print(f"  Curvature at (5,5): {fields['curvature'][5, 5]:.2f}")
    print(f"  Familiarity at (5,5): {fields['familiarity'][5, 5]:.2f}")

    # 更新：在 (5,5) 再次访问
    print("\nAfter revisiting (5,5):")
    space.update_from_observation(
        (5, 5),
        {'obstacles': []}
    )
    fields = space.get_visualization_fields()
    print(f"  Uncertainty at (5,5): {fields['uncertainty'][5, 5]:.2f}")
    print(f"  Curvature at (5,5): {fields['curvature'][5, 5]:.2f}")
    print(f"  Familiarity at (5,5): {fields['familiarity'][5, 5]:.2f}")

    # 更新：到达目标
    print("\nAfter reaching goal at (9,9):")
    space.update_from_observation(
        (9, 9),
        {
            'obstacles': [],
            'goal_position': (9, 9),
            'goal_reached': True
        }
    )
    fields = space.get_visualization_fields()
    print(f"  Uncertainty at (9,9): {fields['uncertainty'][9, 9]:.2f}")
    print(f"  Curvature at (9,9): {fields['curvature'][9, 9]:.2f}")
    print(f"  Familiarity at (9,9): {fields['familiarity'][9, 9]:.2f}")

    # 距离计算
    print("\nDistance calculations:")
    dist1 = space.compute_distance((0, 0), (9, 9))
    dist2 = space.compute_distance((0, 0), (5, 5))
    dist3 = space.compute_distance((5, 5), (9, 9))
    print(f"  (0,0)->(9,9): {dist1:.2f}")
    print(f"  (0,0)->(5,5): {dist2:.2f}")
    print(f"  (5,5)->(9,9): {dist3:.2f}")

    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    test_space_updates()
    test_ricci_curvature()
