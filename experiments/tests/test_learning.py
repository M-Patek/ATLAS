"""
Test: Learning Integration
测试学习集成模块
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from atlas.learning import (
    BayesianOptimizer, SpaceOptimizer,
    MetaLearner, TaskEmbedding, SpaceSelectionPolicy,
    NeuralSpace, SpatialEncoder, MetricNetwork,
    SpaceTrainer, MetaTrainingEnvironment
)
from atlas.core import GeodesicSolver
from atlas.core.registry import create_space


def test_bayesian_optimizer():
    """测试贝叶斯优化器"""
    print("Test 1: BayesianOptimizer")

    # 定义优化问题
    def objective(params):
        # 简单的二次函数
        x, y = params['x'], params['y']
        return -(x - 0.5)**2 - (y - 0.3)**2 + 1.0

    optimizer = BayesianOptimizer(
        param_bounds={'x': (0, 1), 'y': (0, 1)},
        n_initial_points=3
    )

    result = optimizer.optimize(objective, n_iterations=10, verbose=False)

    print(f"  Best params: {result.best_params}")
    print(f"  Best score: {result.best_score:.4f}")
    print(f"  Trials: {len(result.all_trials)}")
    print("  [OK] BayesianOptimizer works")


def test_space_optimizer():
    """测试空间参数优化器"""
    print("Test 2: SpaceOptimizer")

    def evaluate_space(space):
        # 简单评估：在随机环境中规划成功率
        start = (5, 10)
        goal = (35, 10)
        obstacles = {(20, y) for y in range(5, 16)}

        try:
            space.update_from_observation(start, {'obstacles': list(obstacles)})
            solver = GeodesicSolver(space)
            result = solver.solve(start, goal, obstacles)
            return 1.0 if result.success else 0.0
        except:
            return 0.0

    optimizer = SpaceOptimizer(
        space_type="ricci",
        param_bounds={'curvature_scale': (0.1, 3.0)}
    )

    result = optimizer.optimize_for_task(
        evaluate_space,
        base_space_params={'width': 40, 'height': 20},
        n_iterations=8
    )

    print(f"  Optimal curvature_scale: {result.best_params.get('curvature_scale', 'N/A'):.3f}")
    print(f"  Best score: {result.best_score:.2f}")
    print("  [OK] SpaceOptimizer works")


def test_task_embedding():
    """测试任务嵌入"""
    print("Test 3: TaskEmbedding")

    embedding = TaskEmbedding(embedding_dim=8)

    task = {
        'obstacle_density': 0.3,
        'dynamic_ratio': 0.1,
        'goal_distance': 50,
        'required_precision': 0.8
    }

    vec = embedding.embed(task)

    assert len(vec) == 8, "Embedding dimension should be 8"
    assert all(0 <= v <= 1 for v in vec), "All values should be in [0, 1]"

    print(f"  Task embedding shape: {vec.shape}")
    print(f"  Embedding values: {[round(v, 3) for v in vec[:4]]}...")
    print("  [OK] TaskEmbedding works")


def test_meta_learner():
    """测试元学习器"""
    print("Test 4: MetaLearner")

    meta = MetaLearner(
        space_library=['euclidean', 'ricci', 'conformal'],
        embedding_dim=8
    )

    # 测试选择
    task = {
        'obstacle_density': 0.5,
        'exploration_importance': 0.8
    }

    space_name, metadata = meta.select_space_for_task(task, explore_prob=0.1)

    print(f"  Selected space: {space_name}")
    print(f"  Confidence: {max(metadata['selection_probs']):.3f}")
    print(f"  All probs: {[round(p, 3) for p in metadata['selection_probs']]}")

    # 测试推荐
    recommendation = meta.get_space_recommendation(task)
    print(f"  Top recommendation: {recommendation['recommended']}")
    print("  [OK] MetaLearner works")


def test_neural_space():
    """测试神经网络空间"""
    print("Test 5: NeuralSpace")

    space = NeuralSpace(
        width=20, height=15,
        observation_shape=(10,),
        embedding_dim=8
    )

    # 测试观测处理
    obs = {
        'position': (5, 7),
        'lidar': np.random.rand(20)
    }

    space.update_from_observation((5, 7), obs)

    # 测试距离计算（应该有启发式回退）
    d = space.compute_distance((5, 7), (10, 7))
    print(f"  Distance: {d:.2f}")

    # 测试训练
    if len(space.training_buffer) > 0:
        loss = space.train_epoch(batch_size=1)
        print(f"  Training loss: {loss:.4f}")

    print("  [OK] NeuralSpace works")


def test_training_environment():
    """测试元训练环境"""
    print("Test 6: MetaTrainingEnvironment")

    env = MetaTrainingEnvironment(
        width_range=(20, 30),
        height_range=(20, 30)
    )

    task = env.generate_task(difficulty=0.5)

    print(f"  Generated task:")
    print(f"    Size: {task['width']}x{task['height']}")
    print(f"    Obstacles: {len(task['obstacles'])}")
    print(f"    Start: {task['start']}, Goal: {task['goal']}")

    # 测试课程创建
    curriculum = env.create_curriculum(n_stages=3)
    print(f"  Curriculum stages: {len(curriculum.stages)}")
    print(f"  Current stage: {curriculum.current_stage.name}")
    print("  [OK] MetaTrainingEnvironment works")


def test_integration():
    """集成测试"""
    print("Test 7: Learning Integration")

    # 1. 用元学习器选择空间
    meta = MetaLearner(space_library=['euclidean', 'ricci', 'conformal'])

    task = {
        'obstacle_density': 0.4,
        'goal_distance': 40,
        'exploration_importance': 0.6
    }

    selected_space_type, _ = meta.select_space_for_task(task)
    print(f"  Selected: {selected_space_type}")

    # 2. 创建空间
    space = create_space(selected_space_type, width=40, height=20)

    # 3. 如果有参数，用贝叶斯优化
    if selected_space_type == 'ricci':
        def evaluate(params):
            try:
                s = create_space('ricci', width=40, height=20, **params)
                obs = {(20, y) for y in range(5, 15)}
                s.update_from_observation((5, 10), {'obstacles': list(obs)})
                solver = GeodesicSolver(s)
                result = solver.solve((5, 10), (35, 10), obs)
                return 1.0 / (1 + len(result.path)) if result.success else 0.0
            except:
                return 0.0

        optimizer = BayesianOptimizer({'curvature_scale': (0.5, 2.5)})
        result = optimizer.optimize(evaluate, n_iterations=5, verbose=False)
        print(f"  Optimized curvature_scale: {result.best_params.get('curvature_scale', 1.0):.2f}")

    print("  [OK] Integration test passed")


def main():
    print()
    print("=" * 70)
    print("ATLAS: Learning Integration Test Suite")
    print("=" * 70)
    print()

    try:
        test_bayesian_optimizer()
        test_space_optimizer()
        test_task_embedding()
        test_meta_learner()
        test_neural_space()
        test_training_environment()
        test_integration()
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print()
    print("=" * 70)
    print("All learning integration tests passed!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
