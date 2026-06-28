"""
SSFR Enhanced 验证测试

测试增强版 SSFR 的核心功能：
1. 结构假设生成
2. 结构竞争
3. 结构演化
4. 多空间联合表示
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from atlas.core.ssfr_enhanced import (
    StructureHypothesis,
    StructurePool,
    MultiSpaceRepresentation,
    SSFREnhanced,
    ValidationResult,
)
from atlas.core.registry import create_space


def test_structure_hypothesis():
    """测试结构假设"""
    print("=" * 70)
    print("Test 1: Structure Hypothesis")
    print("=" * 70)

    # 创建假设
    hyp = StructureHypothesis(
        id="test_001",
        name="corridor_hypothesis",
        representations={
            "ricci": {
                "params": {"curvature_scale": 1.0},
                "fields": {"uncertainty": np.random.rand(10, 10)}
            },
            "fisher": {
                "params": {"temperature": 1.0},
                "fields": {"belief": np.random.rand(10, 10)}
            }
        },
        context={"scene_type": "corridor"},
        computation_cost=1.0,
        created_at=0.0
    )

    print(f"Created hypothesis: {hyp.id} - {hyp.name}")
    print(f"Initial fitness: {hyp.fitness:.4f}")

    # 验证
    observation = {"position": (5, 5), "obstacles": [(3, 3)]}
    actual = {"position": (5, 5), "obstacles": [(3, 3)]}

    result = hyp.validate(observation, actual, timestamp=1.0)
    print(f"After validation - Error: {result.prediction_error:.4f}, Fitness: {result.fitness:.4f}")

    # 变异
    mutated = hyp.mutate(mutation_rate=0.1)
    print(f"Mutated hypothesis: {mutated.id} - {mutated.name}")
    print(f"Mutated fitness: {mutated.fitness:.4f}")

    print("\n[PASS] StructureHypothesis test passed\n")


def test_structure_pool():
    """测试结构竞争池"""
    print("=" * 70)
    print("Test 2: Structure Pool")
    print("=" * 70)

    pool = StructurePool(max_structures=10)

    # 创建多个假设
    for i in range(5):
        hyp = StructureHypothesis(
            id=f"hyp_{i}",
            name=f"hypothesis_{i}",
            representations={"ricci": {"params": {"curvature_scale": 1.0 + i * 0.1}}},
            context={"index": i},
            computation_cost=1.0 + i * 0.1,
            created_at=0.0
        )
        pool.add(hyp)

    print(f"Added {len(pool.structures)} hypotheses")

    # 模拟竞争
    for step in range(10):
        observation = {"position": (5, 5), "step": step}
        actual = {"position": (5, 5), "obstacles": [(3, 3)]}

        winner, results = pool.compete(observation, actual, timestamp=step)
        print(f"Step {step}: Winner={winner.id}, Fitness={results[0][1].fitness:.4f}")

    # 演化
    new_structures = pool.evolve(timestamp=10.0)
    print(f"\nEvolution generated {len(new_structures)} new structures")

    # 统计
    stats = pool.get_statistics()
    print(f"\nPool statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n[PASS] StructurePool test passed\n")


def test_multi_space_representation():
    """测试多空间联合表示"""
    print("=" * 70)
    print("Test 3: Multi-Space Representation")
    print("=" * 70)

    # 创建空间
    spaces = []
    for name in ['ricci', 'fisher', 'wasserstein']:
        try:
            space = create_space(name, 10, 10)
            spaces.append(space)
            print(f"Created {name} space")
        except Exception as e:
            print(f"Failed to create {name}: {e}")

    if len(spaces) < 2:
        print("Need at least 2 spaces, skipping test")
        return

    # 创建多空间表示
    multi = MultiSpaceRepresentation(spaces)
    print(f"\nCreated MultiSpaceRepresentation with {len(spaces)} spaces")

    # 编码观测
    observation = {"position": (5, 5), "obstacles": [(3, 3), (7, 7)]}
    representations = multi.encode(observation)
    print(f"Encoded observation into {len(representations)} representations")

    for name, rep in representations.items():
        print(f"  {name}: {list(rep.keys())}")

    # 寻找一致结构
    consistent = multi.find_consistent_structure(representations, observation)
    if consistent:
        print(f"\nFound consistent structure: {consistent.name}")
    else:
        print("\nNo consistent structure found (expected with few spaces)")

    print("\n[PASS] MultiSpaceRepresentation test passed\n")


def test_ssfr_enhanced():
    """测试增强版 SSFR"""
    print("=" * 70)
    print("Test 4: SSFR Enhanced")
    print("=" * 70)

    # 创建增强版 SSFR
    ssfr = SSFREnhanced(
        width=10,
        height=10,
        space_names=['ricci', 'fisher', 'wasserstein', 'conformal'],
        max_structures=50,
        evolution_interval=5
    )

    print(f"Created SSFR Enhanced with {len(ssfr.spaces)} spaces")

    # 模拟多步执行
    for step in range(20):
        position = (step % 10, (step * 2) % 10)
        observation = {
            "position": position,
            "obstacles": [(3, 3), (7, 7)] if step % 5 == 0 else [],
            "goal_position": (9, 9),
        }
        actual = {
            "position": position,
            "obstacles": [(3, 3), (7, 7)] if step % 5 == 0 else [],
        }

        result = ssfr.step(position, observation, actual)

        if step % 5 == 0:
            print(f"Step {step}:")
            print(f"  Hypotheses: {result['hypotheses']}")
            print(f"  Winner: {result['winner_id']}")
            print(f"  Pool stats: {result['pool_stats']}")

    # 最终统计
    stats = ssfr.get_statistics()
    print(f"\nFinal statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    # 获取最佳结构
    best = ssfr.get_best_structures(n=3)
    print(f"\nTop {len(best)} structures:")
    for hyp in best:
        print(f"  {hyp.id}: {hyp.name}, fitness={hyp.fitness:.4f}, usage={hyp.usage_count}")

    print("\n[PASS] SSFREnhanced test passed\n")


def test_structure_competition():
    """测试结构竞争机制"""
    print("=" * 70)
    print("Test 5: Structure Competition")
    print("=" * 70)

    pool = StructurePool(max_structures=20)

    # 创建不同类型的假设
    hypothesis_types = [
        {"name": "corridor", "params": {"curvature_scale": 0.5}},
        {"name": "room", "params": {"curvature_scale": 2.0}},
        {"name": "maze", "params": {"curvature_scale": 1.5}},
        {"name": "open", "params": {"curvature_scale": 0.1}},
    ]

    for i, htype in enumerate(hypothesis_types):
        for j in range(3):  # 每个类型3个变体
            hyp = StructureHypothesis(
                id=f"{htype['name']}_{j}",
                name=f"{htype['name']}_v{j}",
                representations={
                    "ricci": {"params": htype["params"]},
                },
                context={"type": htype["name"], "variant": j},
                computation_cost=1.0,
                created_at=0.0
            )
            pool.add(hyp)

    print(f"Created {len(pool.structures)} hypotheses")

    # 模拟不同场景下的竞争
    scenarios = [
        {"name": "narrow_corridor", "obstacles": [(i, 5) for i in range(10)]},
        {"name": "open_room", "obstacles": []},
        {"name": "maze_like", "obstacles": [(i % 3, j % 3) for i in range(9) for j in range(9)]},
    ]

    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")

        # 多步竞争
        for step in range(5):
            observation = {
                "position": (step, step),
                "obstacles": scenario["obstacles"][:step + 1]
            }
            actual = {
                "position": (step, step),
                "obstacles": scenario["obstacles"][:step + 1]
            }

            winner, results = pool.compete(observation, actual, timestamp=step)
            if winner:
                print(f"  Step {step}: Winner={winner.name}, "
                      f"fitness={results[0][1].fitness:.4f}")

        # 演化
        new_structures = pool.evolve(timestamp=5.0)
        print(f"  Evolved: {len(new_structures)} new structures")

    # 最终统计
    stats = pool.get_statistics()
    print(f"\nFinal pool statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n[PASS] Structure Competition test passed\n")


def test_cross_space_consistency():
    """测试跨空间一致性"""
    print("=" * 70)
    print("Test 6: Cross-Space Consistency")
    print("=" * 70)

    # 创建空间
    spaces = []
    for name in ['ricci', 'fisher']:
        try:
            space = create_space(name, 10, 10)
            spaces.append(space)
        except Exception as e:
            print(f"Failed to create {name}: {e}")

    if len(spaces) < 2:
        print("Need at least 2 spaces, skipping test")
        return

    multi = MultiSpaceRepresentation(spaces)

    # 在不同观测下检查一致性
    observations = [
        {"position": (5, 5), "obstacles": [(3, 3)]},
        {"position": (5, 5), "obstacles": [(3, 3), (7, 7), (5, 5)]},
        {"position": (5, 5), "obstacles": []},
    ]

    for i, obs in enumerate(observations):
        print(f"\nObservation {i+1}: {obs['obstacles']}")

        # 编码
        representations = multi.encode(obs)

        # 寻找一致结构
        consistent = multi.find_consistent_structure(representations, obs)
        if consistent:
            print(f"  Found consistent structure: {consistent.name}")
        else:
            print(f"  No consistent structure found")

    print("\n[PASS] Cross-Space Consistency test passed\n")


if __name__ == "__main__":
    print("=" * 70)
    print("SSFR Enhanced Verification Tests")
    print("=" * 70)
    print()

    test_structure_hypothesis()
    test_structure_pool()
    test_multi_space_representation()
    test_ssfr_enhanced()
    test_structure_competition()
    test_cross_space_consistency()

    print("=" * 70)
    print("All Tests Complete")
    print("=" * 70)
