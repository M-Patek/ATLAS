"""
结构复用机制分析：当前问题 vs 理想行为

当前问题：
- _generate_hypothesis_from_space 每步都生成新结构（uuid）
- 没有检查现有结构是否匹配当前场景
- 导致结构池线性增长，没有真正的"复用"

理想行为：
- 生成新结构前，先检查现有结构是否"足够好"
- 如果现有结构匹配当前场景，复用而不是新建
- 结构池增长应该放缓，最终稳定

这个测试对比：
1. 当前行为（盲目生成）
2. 理想行为（基于相似度复用）
"""

import numpy as np
import sys
import os
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from atlas.core.ssfr_enhanced import SSFREnhanced, StructureHypothesis


def test_current_behavior():
    """测试当前行为：结构池线性增长"""
    print("=" * 80)
    print("CURRENT BEHAVIOR: Blind Structure Generation")
    print("=" * 80)

    ssfr = SSFREnhanced(
        width=30, height=30,
        space_names=['ricci', 'fisher', 'wasserstein'],
        max_structures=500
    )

    # 模拟 200 步，每步在相似场景
    pool_sizes = []
    for step in range(200):
        pos = (step % 30, (step * 3) % 30)
        obs = {
            'position': pos,
            'goal_position': (28, 15),
            'obstacles': [],
        }
        ssfr.perceive(pos, obs, active_space_name='ricci')
        pool_sizes.append(len(ssfr.structure_pool.structures))

        if step % 50 == 0:
            print(f"  Step {step:3d}: pool_size={pool_sizes[-1]}")

    print(f"\n  Final pool size: {pool_sizes[-1]}")
    print(f"  Growth rate: {pool_sizes[-1]/200:.2f} structures/step")
    print(f"  Problem: Linear growth - no reuse!")

    return pool_sizes


def test_ideal_behavior():
    """测试理想行为：基于位置的结构复用"""
    print("\n" + "=" * 80)
    print("IDEAL BEHAVIOR: Position-Based Structure Reuse")
    print("=" * 80)

    ssfr = SSFREnhanced(
        width=30, height=30,
        space_names=['ricci', 'fisher', 'wasserstein'],
        max_structures=500
    )

    # 模拟 200 步，但复用相同位置的结构
    pool_sizes = []
    reuse_count = 0
    position_to_structure = {}  # 位置 -> 结构ID 映射

    for step in range(200):
        pos = (step % 30, (step * 3) % 30)
        pos_key = (pos[0] // 5, pos[1] // 5)  # 5x5 区域作为复用单元

        obs = {
            'position': pos,
            'goal_position': (28, 15),
            'obstacles': [],
        }

        # 检查该位置是否已有结构
        if pos_key in position_to_structure:
            # 复用现有结构
            reuse_count += 1
            # 只更新，不生成新结构
            pass
        else:
            # 生成新结构
            ssfr.perceive(pos, obs, active_space_name='ricci')
            position_to_structure[pos_key] = True

        pool_sizes.append(len(ssfr.structure_pool.structures))

        if step % 50 == 0:
            print(f"  Step {step:3d}: pool_size={pool_sizes[-1]}")

    print(f"\n  Final pool size: {pool_sizes[-1]}")
    print(f"  Growth rate: {pool_sizes[-1]/200:.2f} structures/step")
    print(f"  Reuse count: {reuse_count}/200 ({reuse_count/200*100:.1f}%)")
    print(f"  Improvement: Linear -> Sub-linear growth!")

    return pool_sizes


def test_with_obstacles():
    """测试有障碍物时的复用行为"""
    print("\n" + "=" * 80)
    print("SCENARIO: Reuse with Obstacles")
    print("=" * 80)

    ssfr = SSFREnhanced(
        width=30, height=30,
        space_names=['ricci', 'fisher', 'wasserstein'],
        max_structures=500
    )

    # 模拟在相同区域但不同障碍物配置
    pool_sizes = []
    for step in range(200):
        pos = (step % 30, 15)

        # 障碍物随时间变化
        if step % 20 < 10:
            obstacles = [(pos[0] + 2, 15), (pos[0] + 3, 15)]
        else:
            obstacles = [(pos[0] - 2, 15), (pos[0] - 3, 15)]

        obs = {
            'position': pos,
            'goal_position': (28, 15),
            'obstacles': obstacles,
        }

        ssfr.perceive(pos, obs, active_space_name='ricci')
        pool_sizes.append(len(ssfr.structure_pool.structures))

        if step % 50 == 0:
            print(f"  Step {step:3d}: pool_size={pool_sizes[-1]}")

    print(f"\n  Final pool size: {pool_sizes[-1]}")
    print(f"  Growth rate: {pool_sizes[-1]/200:.2f} structures/step")


def analyze_structure_similarity():
    """分析结构相似度"""
    print("\n" + "=" * 80)
    print("ANALYSIS: Structure Similarity")
    print("=" * 80)

    ssfr = SSFREnhanced(
        width=30, height=30,
        space_names=['ricci', 'fisher', 'wasserstein'],
        max_structures=500
    )

    # 生成一些结构
    for i in range(20):
        pos = (i % 30, (i * 3) % 30)
        obs = {
            'position': pos,
            'goal_position': (28, 15),
            'obstacles': [],
        }
        ssfr.perceive(pos, obs, active_space_name='ricci')

    # 分析结构特征
    structures = list(ssfr.structure_pool.structures.values())
    print(f"  Total structures: {len(structures)}")

    if len(structures) >= 2:
        # 检查前两个结构的相似度
        s1 = structures[0]
        s2 = structures[1]

        print(f"\n  Structure 1: {s1.name}")
        print(f"    Context: {s1.context.get('space_type', 'unknown')}")
        print(f"    Features: {list(s1.context.get('features', {}).keys()) if 'features' in s1.context else 'N/A'}")

        print(f"\n  Structure 2: {s2.name}")
        print(f"    Context: {s2.context.get('space_type', 'unknown')}")
        print(f"    Features: {list(s2.context.get('features', {}).keys()) if 'features' in s2.context else 'N/A'}")

        # 检查是否有重复（相同空间类型）
        space_types = [s.context.get('space_type', 'unknown') for s in structures]
        from collections import Counter
        print(f"\n  Space type distribution:")
        for stype, count in Counter(space_types).most_common():
            print(f"    {stype}: {count}")


if __name__ == "__main__":
    # 测试1: 当前行为
    current_sizes = test_current_behavior()

    # 测试2: 理想行为
    ideal_sizes = test_ideal_behavior()

    # 测试3: 有障碍物
    test_with_obstacles()

    # 测试4: 结构相似度分析
    analyze_structure_similarity()

    print("\n" + "=" * 80)
    print("Structure Reuse Analysis Complete")
    print("=" * 80)
